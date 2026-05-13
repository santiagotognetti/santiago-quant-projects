import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from core import (momentum_long_short, perf_stats,
                  benchmark_long_only_equal_weight, get_risk_free_rate)
from data import get_tickers, load_prices

def lookback_sensitivity(
    prices: pd.DataFrame,
    lookback_grid: list[int] = [21, 42, 62, 126, 189, 252, 315, 378],
    topk: int = 25,
    rebalance_period: int = 21,
    tc_per_unit: float = 0.001,
    max_weight: float = 0.2,
    rf: pd.Series | None = None,
) -> pd.DataFrame:
    """

    :param prices:
    :param lookback_grid:
    :param topk:
    :param rebalance_period:
    :param tc_per_unit:
    :param max_weight:
    :param rf:
    :return:
    """
    results = []

    for lb in lookback_grid:
        print(f"  Running lookback = {lb} days")
        port_rets, _, turnover = momentum_long_short(
            prices, lb, topk,
            rebalance_period, tc_per_unit, max_weight
        )
        stats = perf_stats(
            port_rets, freq='day', rf=rf, turnover=turnover, rebalance_period=rebalance_period
        )
        stats["lookback_days"] = lb
        stats["lookback_months"] = round(lb/21)
        results.append(stats)

    return pd.DataFrame(results).set_index("lookback_days")

def plot_lookback_sensitivity(sens_df: pd.DataFrame, save_path: str = None):
    """

    :param sens_df:
    :param save_path:
    :return:
    """
    metrics = {
        "sharpe":"Sharpe Ratio",
        "annualized_return":"Annualized Return",
        "max_drawdown":"Max Drawdown",
        "annualized_vol":"Annualized Volatility",
    }

    fi, axes = plt.subplots(2, 2, figsize=(12,8))
    axes = axes.flatten()

    for ax, (col, label) in zip(axes, metrics.items()):
        ax.plot(sens_df.index, sens_df[col], marker='o', linewidth=2)
        ax.axvline(x=252, color='red', linestyle='dashed', alpha=0.6, label='Default (252 days)')
        ax.set_title(label)
        ax.set_xlabel("Lookback Period")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if col == "sharpe":
            best_lb = sens_df["sharpe"].idxmax()
            ax.axvline(x=best_lb, color='green', linestyle= 'dashed', alpha=0.6, label= 'Best Sharpe Ratio')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def run_sensitivity():
    DATA_PATH = Path(__file__).parent / "Data" / "EUMD_holdings.csv"
    start = "2010-01-01"
    end = dt.date.today()
    TOPK = 10
    tickers = get_tickers(DATA_PATH)
    prices = load_prices(tickers, start, end)
    rf = get_risk_free_rate(start=start, end=end)
    port_rets, positions, turnover = momentum_long_short(prices, lookback=252, topk=TOPK, rebalance_period=21, tc_per_unit=0.001, max_weight=0.2)

    stats = perf_stats(port_rets, freq='day', rf=rf, turnover=turnover, rebalance_period=21)

    print("Project 2 Performance:")
    for k,v in stats.items():
        print(f"  {k}: {v:.6f}")

    print("\nRunning lookback sensitivity analysis...")
    sens_df = lookback_sensitivity(prices, rf=rf, topk=TOPK, rebalance_period=21)
    print("\nSensitivity results:")
    print(sens_df[["lookback_months", "sharpe", "annualized_return",
                   "max_drawdown", "annual turnover"]].to_string())

    best_lookback = int(sens_df["sharpe"].idxmax())
    print(f"\nOptimal lookback by Sharpe: {best_lookback} days "
          f"({round(best_lookback / 21)} months)")

    plot_lookback_sensitivity(sens_df)


    # --- Main backtest with optimal lookback ---
    port_rets, positions, turnover = momentum_long_short(
        prices, lookback=best_lookback, topk=TOPK,
        rebalance_period=21, tc_per_unit=0.001, max_weight=0.20
    )
    stats = perf_stats(port_rets, freq='day', rf=rf,
                       turnover=turnover, rebalance_period=21)
    cum = (1 + port_rets).cumprod() - 1
    plt.figure(figsize=(10,4))
    plt.plot(cum.index, cum.values)
    plt.title("Project2: Cross-Sectional Momentum Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print first rebalance positions
    print("\nSample positions (first rebalance):")
    print(positions[0].loc[positions[0] != 0].sort_values(ascending=True).to_string())

if __name__ == "__main__":
    run_sensitivity()