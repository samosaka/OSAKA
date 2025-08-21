import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- 1. Gelişmiş Özellik Mühendisliği Fonksiyonu ---
def add_final_features(df, swing_n=20, poc_lookback=252, pair_name="EURUSD"):
    print("Nihai özellikler hesaplanıyor...")
    
    high_peaks, _ = find_peaks(df['High'], distance=swing_n)
    low_peaks, _ = find_peaks(-df['Low'], distance=swing_n)
    df['swing_high'] = df['High'].iloc[high_peaks].reindex(df.index, method='ffill').shift(1)
    df['swing_low'] = df['Low'].iloc[low_peaks].reindex(df.index, method='ffill').shift(1)

    df['msb_signal'] = 0
    df.loc[df['Close'] > df['swing_high'], 'msb_signal'] = 1
    df.loc[df['Close'] < df['swing_low'], 'msb_signal'] = -1

    price_multiplier = 1000 if 'JPY' in pair_name.upper() else 100000
    price_bins = (df['Close'] * price_multiplier).round().astype(int)
    poc_series = price_bins.rolling(window=poc_lookback).apply(lambda x: x.mode()[0], raw=False) / price_multiplier
    df['dist_to_poc'] = (df['Close'] - poc_series)
    
    df = df.dropna()
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df.loc[:, ['dist_to_poc']] = scaler.fit_transform(df[['dist_to_poc']])

    print("Özellik mühendisliği tamamlandı.")
    return df

# --- 2. Nihai Forex Ortamı ---
class FinalForexEnv(gym.Env):
    def __init__(self, df, window_size=50, initial_balance=1000, risk_percentage=0.02, reward_ratio=2.0):
        super(FinalForexEnv, self).__init__()
        self.df = df.copy()
        self.df['Time_TR'] = self.df.index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward').tz_convert('Europe/Istanbul')
        
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.risk_percentage = risk_percentage
        self.reward_ratio = reward_ratio
        
        self.features = self.df.columns.drop(['Time_TR', 'swing_high', 'swing_low']).tolist()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window_size, len(self.features) + 1), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = self.window_size
        self.in_trade = False
        self.trade_info = {}
        self.total_trades = 0
        return self._get_observation(), {}

    def _get_observation(self):
        start = self.current_step - self.window_size
        end = self.current_step
        obs_frame = self.df[self.features].iloc[start:end].values
        pos_direction = 1 if self.in_trade and self.trade_info.get('type') == 'long' else (-1 if self.in_trade else 0)
        additional_info = np.full((self.window_size, 1), pos_direction)
        observation = np.concatenate([obs_frame, additional_info], axis=1)
        return observation.astype(np.float32)

    def step(self, action):
        if self.current_step >= len(self.df) - 1:
            return self._get_observation(), 0, True, False, {}

        current_price = self.df['Close'].iloc[self.current_step]
        current_time = self.df.index[self.current_step]
        reward = 0
        closed_trade_info = None

        if self.in_trade:
            trade_type = self.trade_info['type']
            pnl = 0
            is_closed = False
            
            if trade_type == 'long':
                if current_price >= self.trade_info['tp']:
                    pnl = self.trade_info['potential_profit']; self.balance += pnl; is_closed = True
                elif current_price <= self.trade_info['sl']:
                    pnl = -self.trade_info['risk_amount']; self.balance += pnl; is_closed = True
            elif trade_type == 'short':
                if current_price <= self.trade_info['tp']:
                    pnl = self.trade_info['potential_profit']; self.balance += pnl; is_closed = True
                elif current_price >= self.trade_info['sl']:
                    pnl = -self.trade_info['risk_amount']; self.balance += pnl; is_closed = True

            if is_closed:
                closed_trade_info = self.trade_info.copy()
                closed_trade_info['exit_time'] = current_time
                closed_trade_info['exit_price'] = current_price
                closed_trade_info['pnl'] = pnl
                self.in_trade = False

        elif not self.in_trade and action != 0:
            current_hour_tr = self.df['Time_TR'].iloc[self.current_step].hour
            is_trading_allowed = not (current_hour_tr >= 23 or current_hour_tr < 2)
            if is_trading_allowed:
                risk_amount = self.balance * self.risk_percentage
                if action == 1:
                    sl = self.df['swing_low'].iloc[self.current_step]; sl_distance = (current_price - sl)
                    if sl_distance > 0.0001:
                        tp = current_price + (sl_distance * self.reward_ratio)
                        self.in_trade = True; self.trade_info = {'type':'long', 'entry_time': current_time, 'entry_price': current_price, 'sl':sl, 'tp':tp, 'risk_amount':risk_amount, 'potential_profit': risk_amount * self.reward_ratio}; self.total_trades += 1
                elif action == 2:
                    sl = self.df['swing_high'].iloc[self.current_step]; sl_distance = (sl - current_price)
                    if sl_distance > 0.0001:
                        tp = current_price - (sl_distance * self.reward_ratio)
                        self.in_trade = True; self.trade_info = {'type':'short', 'entry_time': current_time, 'entry_price': current_price, 'sl':sl, 'tp':tp, 'risk_amount':risk_amount, 'potential_profit': risk_amount * self.reward_ratio}; self.total_trades += 1
        
        self.current_step += 1
        reward = 0
        if closed_trade_info: reward = 1 if closed_trade_info['pnl'] > 0 else -1
        info = {'balance': self.balance, 'total_trades': self.total_trades, 'closed_trade': closed_trade_info, 'risk_percentage  ': self.risk_percentage , 'reward_ratio ': self.reward_ratio, 'total_trades': self.total_trades, 'current_step': self.current_step, 'balance': self.balance, 'initial_balance': self.initial_balance }
        return self._get_observation(), reward, self.current_step >= len(self.df) - 1, False, info




# --- 3. Detaylı Backtesting Fonksiyonu ---
def run_detailed_backtest(env_class, model, df, pair_name, risk_level):
    env = env_class(df, initial_balance=1000, risk_percentage=risk_level)
    obs, info = env.reset()
    done = False
    balance_history = [env.initial_balance]
    date_history = [df.index[env.current_step]]
    trade_log = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated: break
        balance_history.append(info.get('balance'))
        date_history.append(df.index[min(env.current_step, len(df)-1)])
        if info.get('closed_trade'):
            trade = info['closed_trade'].copy()  # Start with the closed trade dictionary

            # Merge in the rest of the info dictionary
            extra_fields = [
                'balance', 'total_trades', 'risk_percentage  ', 'reward_ratio ',
                'current_step', 'initial_balance'
            ]
            
            for key in extra_fields:
                trade[key.strip()] = info.get(key)  # .strip() handles any accidental spaces

            # Optional: Add readable date
            trade['timestamp'] = str(df.index[min(env.current_step, len(df)-1)])
    
            trade_log.append(trade)
    equity_curve = pd.Series(balance_history, index=pd.to_datetime(date_history)).resample('D').last().ffill()
    returns = equity_curve.pct_change().dropna()
    total_profit = equity_curve.iloc[-1] - equity_curve.iloc[0]
    total_profit_percent = (total_profit / equity_curve.iloc[0]) * 100
    sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    high_water_mark = equity_curve.cummax(); drawdown = (equity_curve - high_water_mark) / high_water_mark
    max_dd = drawdown.min() * 100 if not drawdown.empty else 0
    
    # Yeni Metriklerin Hesaplanması
    buy_trades = [t for t in trade_log if t['type'] == 'long']
    sell_trades = [t for t in trade_log if t['type'] == 'short']
    winning_trades = [t for t in trade_log if t['pnl'] > 0]
    losing_trades = [t for t in trade_log if t['pnl'] < 0]
    
    buy_wins = [t for t in buy_trades if t['pnl'] > 0]
    sell_wins = [t for t in sell_trades if t['pnl'] > 0]
    
    buy_win_rate = len(buy_wins) / len(buy_trades) if len(buy_trades) > 0 else 0
    sell_win_rate = len(sell_wins) / len(sell_trades) if len(sell_trades) > 0 else 0
    
    gross_profit = sum(t['pnl'] for t in winning_trades)
    gross_loss = abs(sum(t['pnl'] for t in losing_trades))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
    max_win = np.max([t['pnl'] for t in winning_trades]) if winning_trades else 0
    max_loss = np.min([t['pnl'] for t in losing_trades]) if losing_trades else 0

    print("-" * 50); print(f"PERFORMANS SONUÇLARI ({pair_name} - %{risk_level*100:.0f} Risk)"); print("-" * 50)
    print(f"Başlangıç Bakiyesi: ${equity_curve.iloc[0]:,.2f}"); print(f"Bitiş Bakiyesi:     ${equity_curve.iloc[-1]:,.2f}")
    print(f"Net Kâr/Zarar:      ${total_profit:,.2f} ({total_profit_percent:.2f}%)")
    print(f"Sharpe Oranı:        {sharpe:.2f}"); print(f"Maksimum Düşüş (MDD): {max_dd:.2f}%")
    print("-" * 25); print("İşlem İstatistikleri"); print("-" * 25)
    print(f"Toplam İşlem Sayısı: {len(trade_log)}")
    print(f"Alım (Buy) İşlem Sayısı:  {len(buy_trades)}"); print(f"Satım (Sell) İşlem Sayısı: {len(sell_trades)}")
    print(f"Alım Kazanma Oranı:  %{buy_win_rate*100:.2f}"); print(f"Satım Kazanma Oranı: %{sell_win_rate*100:.2f}")
    print(f"Kâr Faktörü:         {profit_factor:.2f}")
    print("-" * 25); print("Kâr/Zarar İstatistikleri"); print("-" * 25)
    print(f"Ortalama Kârlı İşlem:  ${avg_win:,.2f}"); print(f"Ortalama Zararlı İşlem: ${avg_loss:,.2f}")
    print(f"En Büyük Kâr:         ${max_win:,.2f}"); print(f"En Büyük Zarar:        ${max_loss:,.2f}"); print("-" * 50)
    
    # İşlem Kayıt Defterini Dosyaya Kaydet
    trade_log_df = pd.DataFrame(trade_log)
    log_filename = f'trade_log_{pair_name}.txt'
    trade_log_df.to_csv(log_filename, sep='\t', index=False)
    print(f"Tüm işlemlerin detaylı kaydı '{log_filename}' adıyla kaydedildi.")

    chart_filename = f'equity_curve_{pair_name}_risk_{int(risk_level*100)}p.png'
    plt.figure(figsize=(15, 7)); plt.plot(equity_curve.index, equity_curve.values); plt.title(f'Bakiye Değişim Grafiği ({pair_name} - Risk %{risk_level*100:.0f})')
    plt.xlabel('Tarih'); plt.ylabel('Bakiye ($)'); plt.grid(True); plt.savefig(chart_filename)
    print(f"Bakiye değişim grafiği '{chart_filename}' adıyla kaydedildi.")

# --- 4. Ana Çalıştırma Bloğu ---
if __name__ == '__main__':
    PAIR_NAME = "BTCUSD" 
    DATA_FILE = 'BTCUSD_M5.csv'
    MODEL_SAVE_PATH = f'ppo_model_{PAIR_NAME}_2p_risk.zip'
    TOTAL_TIMESTEPS = 300000
    RISK_PERCENTAGE = 0.02
    
    if not os.path.exists(DATA_FILE):
        print(f"Hata: '{DATA_FILE}' dosyası bulunamadı.")
    else:
        # Kodun bu kısmı artık EURUSD ve USDJPY için ayrı veri okuma mantığı içeriyor
        print(f"Ham veri dosyası ({DATA_FILE}) okunuyor ve işleniyor...")
        try:
            if "USDJPY" in DATA_FILE:
                 df_base = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
                 df_base['Time'] = pd.to_datetime(df_base['Time'])
            elif "EURUSD" in DATA_FILE:
                 df_base = pd.read_csv(DATA_FILE, sep='\t')
                 df_base.columns = [str(col).replace('<','').replace('>','') for col in df_base.columns]
                 df_base['Time'] = pd.to_datetime(df_base['DATE'] + ' ' + df_base['TIME'])
            else: # Diğer pariteler için (ör: GBPUSD)
                 df_base = pd.read_csv(DATA_FILE, sep='\t', header=None, names=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
                 df_base['Time'] = pd.to_datetime(df_base['Time'])
                 
            df_base = df_base.set_index('Time')
            df_base = df_base[['Open', 'High', 'Low', 'Close', 'Volume']]
            df_base.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            print("Veri başarıyla okundu ve temizlendi.")
        except Exception as e:
            print(f"Veri okunurken bir hata oluştu: {e}")
            exit()



        df_final = add_final_features(df_base, pair_name=PAIR_NAME)
            
        print(f"\n{PAIR_NAME} için %{RISK_PERCENTAGE*100:.0f} risk ile eğitim başlıyor...")
        best_params = {'n_steps': 2048, 'gamma': 0.99, 'ent_coef': 0.01, 'learning_rate': 0.00003, 'clip_range': 0.2}
        
        # env_train = DummyVecEnv([lambda: FinalForexEnv(df_final, risk_percentage=RISK_PERCENTAGE)])
        # model = PPO("MlpPolicy", env_train, verbose=1, **best_params)
        
        # model.learn(total_timesteps=TOTAL_TIMESTEPS)
        # model.save(MODEL_SAVE_PATH)
        print(f"\n{PAIR_NAME} modeli eğitildi ve '{MODEL_SAVE_PATH}' olarak kaydedildi.")
            
        print(f"\nEğitilmiş {PAIR_NAME} modeli ile detaylı backtesting süreci başlatılıyor...")
        loaded_model = PPO.load(MODEL_SAVE_PATH)
        run_detailed_backtest(FinalForexEnv, loaded_model, df_final, pair_name=PAIR_NAME, risk_level=RISK_PERCENTAGE)