import os, json, pickle
from typing import Dict, Callable
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from app.services.agent.utils import load_ohlcv
from app.services.agent.plugin_loader import load_strategy_module

def test_model(
    *,
    run_dir: str,
    data_filename: str,
    data_dir: str = "data",
    backtest_fn: Callable = None,   # your existing function, e.g., run_detailed_backtest
) -> Dict:
    """
    Test a trained run against new data using the archived strategy.py.
    Returns {'ok', 'run_dir', 'meta', 'result'}.
    """
    if backtest_fn is None:
        return {"ok": False, "error": "backtest_fn is required"}

    run_dir = 'static\\agents\\' + run_dir
    # Load meta + archived strategy
    meta_path = os.path.join(run_dir, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    strat_path = os.path.join(run_dir, meta.get("strategy_filename", "strategy.py"))
    strat = load_strategy_module(strat_path)

    # Load eval data
    data_path = os.path.join(data_dir, data_filename)
    df_raw = load_ohlcv(data_path)

    # Load FE state
    fe_dir = os.path.join(run_dir, "fe_state")
    if hasattr(strat, "load_artifacts"):
        fe_state = strat.load_artifacts(fe_dir)
    else:
        with open(os.path.join(fe_dir, "state.pkl"), "rb") as f:
            fe_state = pickle.load(f)

    # Recompute features with SAME logic/state
    df_final, _ = strat.add_final_features(df_raw, meta["config"], fe_state=fe_state)

    # Optional VecNormalize for obs normalization during predict
    def _mk(): return strat.make_env(df_final, meta["config"], fe_state)
    env = DummyVecEnv([_mk])
    vecnorm_path = os.path.join(run_dir, "vecnorm.pkl")
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(os.path.join(run_dir, "model.zip"))

    # Adapter: many backtests accept an env_class/maker; we give a lambda
    result = backtest_fn(
        env_class=lambda df, **kw: strat.make_env(df, meta["config"], fe_state),
        model=model,
        df=df_final,
        pair_name=meta["symbol"],
        risk_level=meta["config"].get("risk_percentage", 0.02)
    )

    return {"ok": True, "run_dir": run_dir, "meta": meta, "result": result}
