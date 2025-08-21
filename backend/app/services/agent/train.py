import os, time, json, pickle
from typing import Dict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from app.services.agent.utils import load_ohlcv, write_json, sha256_file, copy_file
from app.services.agent.plugin_loader import load_strategy_module

def _resolve_paths(strategy_name: str, data_filename: str, strategies_dir: str, data_dir: str, out_root: str):
    strategy_path = os.path.join(strategies_dir, f"{strategy_name}")
    data_path = os.path.join(data_dir, data_filename)
    return strategy_path, data_path, out_root

def train_model(
    *,
    strategy_name: str,
    data_filename: str,
    config: Dict,
    strategies_dir: str = "strategies",
    data_dir: str = "data",
    out_root: str = "artifacts",
    train_steps: int = 300_000,
) -> Dict:
    """Train a model with the given strategy + data; returns {'ok', 'run_dir', 'model_path', 'meta'}."""
    strategy_path, data_path, out_root = _resolve_paths(strategy_name, data_filename, strategies_dir, data_dir, out_root)
    os.makedirs(out_root, exist_ok=True)

    symbol = config.get("symbol", "BTCUSD")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, symbol, run_id)
    os.makedirs(run_dir, exist_ok=True)

    strat = load_strategy_module(strategy_path)
    df_raw = load_ohlcv(data_path)

    merged_cfg = {
        "symbol": symbol,
        "window_size": config.get("window_size", 50),
        "risk_percentage": config.get("risk_percentage", 0.02),
        "reward_ratio": config.get("reward_ratio", 2.0),
        "timezone": config.get("timezone", "Europe/Istanbul"),
        "seed": config.get("seed", 42),
        **{k: v for k, v in config.items() if k not in ["symbol","window_size","risk_percentage","reward_ratio","timezone","seed"]},
    }

    # Feature engineering (fit here)
    df_final, fe_state = strat.add_final_features(df_raw, merged_cfg, fe_state=None)

    # Env
    def _mk(): return strat.make_env(df_final, merged_cfg, fe_state)
    env = DummyVecEnv([_mk])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # Hyperparams
    hps = getattr(strat, "get_hyperparams", lambda c: dict(
        n_steps=2048, gamma=0.99, ent_coef=0.01, learning_rate=3e-5, clip_range=0.2, seed=merged_cfg["seed"], verbose=1
    ))(merged_cfg)

    model = PPO("MlpPolicy", env, **hps)
    model.learn(total_timesteps=train_steps)

    # Save artifacts
    model_path = os.path.join(run_dir, "model.zip")
    model.save(model_path)
    env.save(os.path.join(run_dir, "vecnorm.pkl"))

    fe_dir = os.path.join(run_dir, "fe_state")
    os.makedirs(fe_dir, exist_ok=True)
    if hasattr(strat, "save_artifacts"):
        strat.save_artifacts(fe_state, fe_dir)
    else:
        with open(os.path.join(fe_dir, "state.pkl"), "wb") as f:
            pickle.dump(fe_state, f)

    # Archive strategy file + hash
    archived_strategy_path = os.path.join(run_dir, "strategy.py")
    copy_file(strategy_path, archived_strategy_path)
    strat_hash = sha256_file(archived_strategy_path)

    meta = {
        "run_id": run_id,
        "symbol": symbol,
        "config": merged_cfg,
        "strategy_name": strategy_name,
        "strategy_filename": "strategy.py",
        "strategy_sha256": strat_hash,
        "data_source": os.path.abspath(data_path),
        "lib_versions": {"stable_baselines3": ">=2.3.0", "gymnasium": ">=0.29"},
        "train_steps": train_steps,
    }
    write_json(os.path.join(run_dir, "meta.json"), meta)

    return {"ok": True, "run_dir": run_dir, "model_path": model_path, "meta": meta}
