# strategy_example.py
import numpy as np
import pandas as pd
from gymnasium import spaces
import gymnasium as gym
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

# ---------- Feature Engineering ----------
def _safe_mode(x: pd.Series):
    m = x.mode()
    return m.iloc[0] if len(m) else np.nan

def _price_bins(close: pd.Series, symbol: str):
    s = symbol.upper()
    if "JPY" in s:
        mult = 1000
        return (close * mult).round().astype(int) / mult
    if any(k in s for k in ["BTC","ETH","CRYPTO"]):
        w = 5.0
        return (close / w).round() * w
    mult = 100000
    return (close * mult).round().astype(int) / mult

def add_final_features(df: pd.DataFrame, config: dict, fe_state=None):
    """
    Return df_final and FE state. If fe_state is None, fit scalers; else reuse them.
    - Ensures 'Volume' exists (fallback to 0 if missing).
    - Scales only columns that actually exist to avoid KeyError.
    - Uses safe rolling mode and symbol-aware price bins (via _safe_mode/_price_bins).
    """
    # --- basics & defaults ---
    symbol = config.get("symbol", "UNKNOWN")
    swing_n = config.get("swing_n", 20)
    poc_lookback = config.get("poc_lookback", 252)

    x = df.copy()

    # Ensure core columns exist
    required_base = ['Open', 'High', 'Low', 'Close']
    missing_base = [c for c in required_base if c not in x.columns]
    if missing_base:
        raise ValueError(f"Missing required columns: {missing_base}")

    # Ensure Volume exists (fallback to zeros)
    if 'Volume' not in x.columns:
        x['Volume'] = 0

    # Cast numerics defensively
    for col in ['Open','High','Low','Close','Volume']:
        x[col] = pd.to_numeric(x[col], errors='coerce')

    # --- swing highs/lows ---
    hi_idx, _ = find_peaks(x['High'].values, distance=swing_n)
    lo_idx, _ = find_peaks(-x['Low'].values,  distance=swing_n)

    # Map peaks onto full index, shift(1) to avoid lookahead
    x['swing_high'] = x['High'].iloc[hi_idx].reindex(x.index, method='ffill').shift(1)
    x['swing_low']  = x['Low'].iloc[lo_idx].reindex(x.index,  method='ffill').shift(1)

    # --- MSB signal ---
    x['msb_signal'] = 0
    x.loc[x['Close'] > x['swing_high'], 'msb_signal'] = 1
    x.loc[x['Close'] < x['swing_low'],  'msb_signal'] = -1

    # --- distance to rolling POC (symbol-aware binning) ---
    bins = _price_bins(x['Close'], symbol)
    poc = bins.rolling(poc_lookback).apply(lambda s: _safe_mode(pd.Series(s)), raw=False)
    x['dist_to_poc'] = x['Close'] - poc

    # Drop rows with NaNs introduced by rolling/peaks
    x = x.dropna().copy()

    # Ensure numeric dtypes for scaling
    x['msb_signal'] = pd.to_numeric(x['msb_signal'], errors='coerce').astype(float)
    x['dist_to_poc'] = pd.to_numeric(x['dist_to_poc'], errors='coerce').astype(float)

    # --- scaling (fit if fe_state is None; else reuse) ---
    cols_to_scale = ['Open','High','Low','Close','Volume','dist_to_poc','msb_signal']
    cols_present = [c for c in cols_to_scale if c in x.columns]

    if fe_state is None or 'scaler' not in fe_state or fe_state['scaler'] is None:
        scaler = StandardScaler()
        x[cols_present] = scaler.fit_transform(x[cols_present])
        fe_state = {"scaler": scaler}
    else:
        scaler = fe_state['scaler']
        x[cols_present] = scaler.transform(x[cols_present])

    return x, fe_state


# ---------- Environment ----------
class FinalForexEnv(gym.Env):
    metadata = {}

    def __init__(self, df, config, fe_state):
        super().__init__()
        self.df = df.copy()
        self.window = config["window_size"]
        self.balance0 = 1000
        self.balance = self.balance0
        self.risk_pct = config["risk_percentage"]
        self.rr = config["reward_ratio"]
        self.timezone = config.get("timezone", "Europe/Istanbul")

        self.features = [c for c in self.df.columns if c not in ['swing_high','swing_low']]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
            shape=(self.window, len(self.features) + 1), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.in_trade = False
        self.trade = {}
        self.step_i = self.window
        self.total_trades = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.balance0
        self.in_trade = False
        self.trade = {}
        self.total_trades = 0
        self.step_i = self.window
        return self._obs(), {}

    def _obs(self):
        s = self.step_i - self.window
        e = self.step_i
        frame = self.df[self.features].iloc[s:e].values
        pos_dir = 1 if (self.in_trade and self.trade.get('type')=='long') else (-1 if self.in_trade else 0)
        extra = np.full((self.window, 1), pos_dir)
        return np.concatenate([frame, extra], axis=1).astype(np.float32)

    def step(self, action):
        if self.step_i >= len(self.df) - 1:
            return self._obs(), 0.0, True, False, {}

        price = self.df['Close'].iloc[self.step_i]
        closed = None

        if self.in_trade:
            t = self.trade; pnl=0.0; done=False
            if t['type']=='long':
                if price >= t['tp']: pnl = t['profit']; self.balance += pnl; done=True
                elif price <= t['sl']: pnl = -t['risk']; self.balance += pnl; done=True
            else:
                if price <= t['tp']: pnl = t['profit']; self.balance += pnl; done=True
                elif price >= t['sl']: pnl = -t['risk']; self.balance += pnl; done=True
            if done:
                closed = {**t, "exit_price": price, "pnl": pnl}
                self.in_trade = False

        elif not self.in_trade and action != 0:
            risk = self.balance * self.risk_pct
            if action == 1:
                sl = self.df['swing_low'].iloc[self.step_i]
                d = price - sl
                if d > 1e-4:
                    tp = price + d * self.rr
                    self.trade = {"type":"long","entry":price,"sl":sl,"tp":tp,"risk":risk,"profit":risk*self.rr}
                    self.in_trade = True; self.total_trades += 1
            elif action == 2:
                sl = self.df['swing_high'].iloc[self.step_i]
                d = sl - price
                if d > 1e-4:
                    tp = price - d * self.rr
                    self.trade = {"type":"short","entry":price,"sl":sl,"tp":tp,"risk":risk,"profit":risk*self.rr}
                    self.in_trade = True; self.total_trades += 1

        self.step_i += 1
        reward = 0.0
        if closed: reward = 1.0 if closed['pnl'] > 0 else -1.0
        info = {"balance": self.balance, "closed_trade": closed, "total_trades": self.total_trades}
        done = self.step_i >= len(self.df) - 1
        return self._obs(), reward, done, False, info

def make_env(df_final, config, fe_state):
    return FinalForexEnv(df_final, config, fe_state)

# Optional hooks for training HPs or saving/loading FE state in custom ways:
def get_hyperparams(config):
    return dict(n_steps=2048, gamma=0.99, ent_coef=0.01, learning_rate=3e-5, clip_range=0.2, seed=config.get("seed", 42), verbose=1)

def save_artifacts(fe_state, out_dir: str):
    import pickle, os
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(fe_state["scaler"], f)

def load_artifacts(in_dir: str):
    import pickle, os
    with open(os.path.join(in_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return {"scaler": scaler}
