import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from data import fetch_polygon_minute_bars, prepare_bars
from core import compute_signal, backtest, perf_stats, build_trade_log

DATA_DIR    = Path(__file__).parent / "Data"
TICKER      = "SPY"
TRAIN_START = "2022-01-01"
TRAIN_END   = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

def run_sensitivity(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Grid search over z_window × z_thresh on the training period.
    Selects optimal parameters by in-sample Sharpe ratio.

    :param df_train: prepared bar DataFrame for the training period
    :return: results DataFrame sorted by Sharpe (descending)
    """
    z_windows    = [15, 30, 60, 120, 240]
    z_thresholds = [0.5, 1.0, 1.5, 2.0]
    results = []

    for zw in z_windows:
        for zt in z_thresholds:
            df_sig = compute_signal(df_train, z_window=zw, z_thresh=zt)
            df_bt  = backtest(df_sig)
            stats  = perf_stats(df_bt["strategy_ret"], freq="min")
            stats["z_window"] = zw
            stats["z_thresh"] = zt
            results.append(stats)
            print(f"  z_window={zw:>4}, z_thresh={zt:.1f}  →  "
                  f"Sharpe={stats['sharpe']:.3f}  hit_rate={stats['hit_rate']:.2%}")

    return pd.DataFrame(results).sort_values("sharpe", ascending=False)

def run_live():
    api_key = os.environ.get("POLYGON_API_KEY", "Sy4ZQHDR_6kt6HnD8VOCJpj0qC_8PnZp")
    if not api_key:
        raise EnvironmentError(
            "Set your Polygon API key:\n  export POLYGON_API_KEY=your polygon key here"
        )

    # ── Download & prepare ──────────────────────────────────────────────
    print(f"Fetching {TICKER} 1-min bars ({TRAIN_START} → {TEST_END})...")
    raw = fetch_polygon_minute_bars(
        ticker=TICKER,
        start=TRAIN_START,
        end=TEST_END,
        api_key=api_key,
        cache_dir=DATA_DIR,
    )
    df = prepare_bars(raw)
    df_train = df[TRAIN_START:TRAIN_END]
    df_test  = df[TEST_START:TEST_END]
    print(f"Train: {len(df_train):,} bars  |  Test: {len(df_test):,} bars")

    # ── In-sample parameter search ───────────────────────────────────────
    print("\nRunning in-sample sensitivity (z_window × z_thresh grid)...")
    sens = run_sensitivity(df_train)

    print("\nTop 5 in-sample parameter combinations:")
    print(sens[["z_window", "z_thresh", "sharpe", "sortino",
                "calmar", "hit_rate", "profit_factor"]].head(5).to_string(index=False))

    best_zw = int(sens.iloc[0]["z_window"])
    best_zt = float(sens.iloc[0]["z_thresh"])
    print(f"\nOptimal: z_window={best_zw}, z_thresh={best_zt}")

    # ── Out-of-sample backtest ───────────────────────────────────────────
    print(f"\nOut-of-sample backtest ({TEST_START} → {TEST_END})...")
    df_sig = compute_signal(df_test, z_window=best_zw, z_thresh=best_zt)
    df_bt  = backtest(df_sig)
    stats  = perf_stats(df_bt["strategy_ret"], freq="min")

    print("\nOut-of-sample performance:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ── Trade log ────────────────────────────────────────────────────────
    trade_log = build_trade_log(df_bt)
    if not trade_log.empty:
        print(f"\nTrade log ({len(trade_log)} trades):")
        print(f"  Avg holding period : {trade_log['n_bars'].mean():.1f} bars")
        print(f"  Avg PnL per trade  : {trade_log['pnl'].mean():.6f}")
        print(f"  Win rate           : {(trade_log['pnl'] > 0).mean():.2%}")
        wins  = trade_log[trade_log["pnl"] > 0]["pnl"].sum()
        loss  = trade_log[trade_log["pnl"] < 0]["pnl"].sum()
        print(f"  Profit factor      : {wins / abs(loss):.3f}")

    # ── Time-of-day P&L breakdown ────────────────────────────────────────
    df_bt["hour"] = df_bt.index.hour
    tod = df_bt.groupby("hour")["strategy_ret"].sum()
    print("\nP&L by hour of day (ET):")
    print(tod.to_string())

    # ── Plots ────────────────────────────────────────────────────────────
    cum = (1 + df_bt["strategy_ret"]).cumprod() - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(cum.index, cum.values, linewidth=1.2)
    axes[0].set_title(f"Project 1: {TICKER} Intraday — Cumulative Return (OOS 2024)")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Cumulative Return")
    axes[0].grid(True, alpha=0.3)

    tod.plot(kind="bar", ax=axes[1], color="steelblue", edgecolor="white")
    axes[1].set_title("P&L by Hour of Day")
    axes[1].set_xlabel("Hour (ET)")
    axes[1].set_ylabel("Cumulative P&L")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_live()