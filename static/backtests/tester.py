# run_detailed_backtest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any


def run_detailed_backtest(env_class, model, df: pd.DataFrame, pair_name: str, risk_level: float) -> dict[str, Any]:
    """
    Env-agnostic backtest runner.
    Expects env.reset() -> (obs, info) and env.step(a) -> (obs, reward, terminated, truncated, info).
    Tries multiple fallbacks for initial balance and current step.
    """
    # You can tweak this default if you pass a different one to the env below
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
        # If index is not datetime, convert later; or coerce to datetime here
        first_ts = pd.to_datetime(df.index[min(current_step, len(df) - 1)])

    done = False
    truncated = False
    balance_history = [float(start_balance)]
    date_history = [first_ts]
    trade_log: list[dict[str, Any]] = []

    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        # Update current_step fallback each loop
        current_step = getattr(env, "current_step", current_step + 1)
        idx = min(current_step, len(df) - 1)
        ts = df.index[idx]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.to_datetime(ts)

        # Track balance; fall back to last known if missing
        balance = info.get("balance", balance_history[-1])
        balance_history.append(float(balance))
        date_history.append(ts)

        # Closed trade details enrichment
        if info.get("closed_trade"):
            trade = dict(info["closed_trade"])  # copy
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
            trade["timestamp"] = str(ts)
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

    # Save artifacts
    trade_log_df = pd.DataFrame(trade_log)
    log_filename = f"trade_log_{pair_name}.txt"
    trade_log_df.to_csv(log_filename, sep="\t", index=False)
    print(f"Tüm işlemlerin detaylı kaydı '{log_filename}' adıyla kaydedildi.")

    chart_filename = f"equity_curve_{pair_name}_risk_{int(risk_level*100)}p.png"
    plt.figure(figsize=(15, 7))
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title(f'Bakiye Değişim Grafiği ({pair_name} - Risk %{risk_level*100:.0f})')
    plt.xlabel("Tarih")
    plt.ylabel("Bakiye ($)")
    plt.grid(True)
    plt.savefig(chart_filename)
    print(f"Bakiye değişim grafiği '{chart_filename}' adıyla kaydedildi.")

    return {
        "equity_curve": equity_curve,
        "trade_log": trade_log_df,
        "metrics": {
            "total_profit": total_profit,
            "total_profit_percent": total_profit_percent,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "profit_factor": profit_factor,
            "buy_win_rate": buy_win_rate,
            "sell_win_rate": sell_win_rate,
        },
    }
