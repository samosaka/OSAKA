# run_detailed_backtest.py
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any


def _to_serializable(o):
    """Helper to make numpy/pandas types JSON-serializable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def run_detailed_backtest(
    env_class,
    model,
    df: pd.DataFrame,
    pair_name: str,
    risk_level: float,
    test_output_folder_name: str | None = None
) -> dict[str, Any]:
    """
    Env-agnostic backtest runner.
    Saves artifacts under results/<folder>/ where <folder> is test_output_folder_name or pair_name.
    """
    # ---------- Paths ----------
    # If the caller passes a folder name use it; otherwise default to pair_name
    folder_name = test_output_folder_name or pair_name
    out_dir = Path("static/backtest_results") / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    default_initial_balance = 1000.0

    # Instantiate env — keep kwargs that your env actually supports
    env = env_class(df, initial_balance=default_initial_balance, risk_percentage=risk_level)

    obs, info = env.reset()

    # Robust starting balance
    start_balance = (
        getattr(env, "initial_balance", None)
        or info.get("initial_balance")
        or info.get("balance")
        or default_initial_balance
    )

    # Robust current step
    current_step = getattr(env, "current_step", 0)
    if isinstance(df.index, pd.DatetimeIndex):
        first_ts = df.index[min(current_step, len(df) - 1)]
    else:
        first_ts = pd.to_datetime(df.index[min(current_step, len(df) - 1)])

    done = False
    truncated = False
    balance_history = [float(start_balance)]
    date_history = [first_ts]
    trade_log: list[dict[str, Any]] = []

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        current_step = getattr(env, "current_step", current_step + 1)
        idx = min(current_step, len(df) - 1)
        ts = df.index[idx]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts)

        balance = info.get("balance", balance_history[-1])
        balance_history.append(float(balance))
        date_history.append(ts)

        # Closed trade details enrichment
        if info.get("closed_trade"):
            trade = dict(info["closed_trade"])  # shallow copy
            for key in (
                "balance",
                "total_trades",
                "risk_percentage",
                "reward_ratio",
                "current_step",
                "initial_balance",
            ):
                if key in info:
                    trade[key] = info[key]
            trade["timestamp"] = ts.isoformat()
            trade["pair"] = pair_name
            trade_log.append(trade)

    # --- Metrics ---
    equity_curve = (
        pd.Series(balance_history, index=pd.to_datetime(date_history))
        .resample("D")
        .last()
        .ffill()
    )

    returns = equity_curve.pct_change().dropna()
    total_profit = float(equity_curve.iloc[-1] - equity_curve.iloc[0])
    total_profit_percent = (total_profit / float(equity_curve.iloc[0])) * 100.0
    sharpe = 0.0
    if returns.std(ddof=0) != 0:
        sharpe = float(np.sqrt(252.0) * (returns.mean() / returns.std(ddof=0)))

    high_water_mark = equity_curve.cummax()
    drawdown = (equity_curve - high_water_mark) / high_water_mark
    max_dd = float(drawdown.min() * 100.0) if not drawdown.empty else 0.0

    buy_trades = [t for t in trade_log if t.get("type") in ("long", "buy")]
    sell_trades = [t for t in trade_log if t.get("type") in ("short", "sell")]
    winning_trades = [t for t in trade_log if float(t.get("pnl", 0)) > 0]
    losing_trades = [t for t in trade_log if float(t.get("pnl", 0)) < 0]

    buy_wins = [t for t in buy_trades if float(t.get("pnl", 0)) > 0]
    sell_wins = [t for t in sell_trades if float(t.get("pnl", 0)) > 0]

    buy_win_rate = len(buy_wins) / len(buy_trades) if buy_trades else 0.0
    sell_win_rate = len(sell_wins) / len(sell_trades) if sell_trades else 0.0

    gross_profit = sum(float(t.get("pnl", 0)) for t in winning_trades)
    gross_loss = abs(sum(float(t.get("pnl", 0)) for t in losing_trades))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_win = float(np.mean([float(t.get("pnl", 0)) for t in winning_trades])) if winning_trades else 0.0
    avg_loss = float(np.mean([float(t.get("pnl", 0)) for t in losing_trades])) if losing_trades else 0.0
    max_win = float(np.max([float(t.get("pnl", 0)) for t in winning_trades])) if winning_trades else 0.0
    max_loss = float(np.min([float(t.get("pnl", 0)) for t in losing_trades])) if losing_trades else 0.0

    # Console summary (optional)
    print("-" * 50)
    print(f"PERFORMANS SONUÇLARI ({pair_name} - %{risk_level*100:.0f} Risk)")
    print("-" * 50)
    print(f"Başlangıç Bakiyesi: ${equity_curve.iloc[0]:,.2f}")
    print(f"Bitiş Bakiyesi:     ${equity_curve.iloc[-1]:,.2f}")
    print(f"Net Kâr/Zarar:      ${total_profit:,.2f} ({total_profit_percent:.2f}%)")
    print(f"Sharpe Oranı:        {sharpe:.2f}")
    print(f"Maksimum Düşüş (MDD): {max_dd:.2f}%")
    print("-" * 25)
    print("İşlem İstatistikleri")
    print("-" * 25)
    print(f"Toplam İşlem Sayısı: {len(trade_log)}")
    print(f"Alım (Buy):  {len(buy_trades)} | Satım (Sell): {len(sell_trades)}")
    print(f"Alım Kazanma Oranı:  %{buy_win_rate*100:.2f}")
    print(f"Satım Kazanma Oranı: %{sell_win_rate*100:.2f}")
    print(f"Kâr Faktörü:         {profit_factor:.2f}")
    print("-" * 25)
    print("Kâr/Zarar İstatistikleri")
    print("-" * 25)
    print(f"Ortalama Kârlı İşlem:  ${avg_win:,.2f}")
    print(f"Ortalama Zararlı İşlem: ${avg_loss:,.2f}")
    print(f"En Büyük Kâr:         ${max_win:,.2f}")
    print(f"En Büyük Zarar:        ${max_loss:,.2f}")
    print("-" * 50)

    # ---------- Save artifacts ----------
    # 1) Trades as JSON
    trades_path = out_dir / "trades.json"
    with trades_path.open("w", encoding="utf-8") as f:
        json.dump(trade_log, f, ensure_ascii=False, indent=2, default=_to_serializable)
    print(f"Tüm işlemlerin JSON kaydı: '{trades_path.as_posix()}'")

    # 2) Metrics as JSON
    metrics = {
        "pair": pair_name,
        "risk_level": risk_level,
        "start_balance": float(equity_curve.iloc[0]),
        "end_balance": float(equity_curve.iloc[-1]),
        "total_profit": total_profit,
        "total_profit_percent": total_profit_percent,
        "sharpe": sharpe,
        "max_drawdown_percent": max_dd,
        "profit_factor": profit_factor,
        "buy_win_rate": buy_win_rate,
        "sell_win_rate": sell_win_rate,
        "total_trades": len(trade_log),
    }
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=_to_serializable)
    print(f"Metrix JSON kaydı: '{metrics_path.as_posix()}'")

    used_data_path = out_dir / "used_data.json"
    with used_data_path.open("w", encoding="utf-8") as f:
        json.dump(df.reset_index().to_dict(orient="records"), f, ensure_ascii=False, indent=2, default=_to_serializable)

    # 3) Equity curve PNG
    chart_filename = f"equity_curve_{pair_name}_risk_{int(risk_level*100)}p.png"
    chart_path = out_dir / chart_filename
    plt.figure(figsize=(15, 7))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title(f'Bakiye Değişim Grafiği ({pair_name} - Risk %{risk_level*100:.0f})')
    plt.xlabel("Tarih")
    plt.ylabel("Bakiye ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(chart_path.as_posix())
    plt.close()
    print(f"Bakiye değişim grafiği: '{chart_path.as_posix()}'")

    return {
        "equity_curve": equity_curve,
        "trade_log": pd.DataFrame(trade_log),
        "metrics": metrics,
        "artifacts": {
            "folder": out_dir.as_posix(),
            "trades_json": trades_path.as_posix(),
            "metrics_json": metrics_path.as_posix(),
            "equity_curve_png": chart_path.as_posix(),
        },
    }
