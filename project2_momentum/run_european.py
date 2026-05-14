import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from pathlib import Path
from core import (momentum_long_short, perf_stats, factor_decomposition,
                  benchmark_long_only_equal_weight, get_risk_free_rate)
from data import get_tickers, load_prices

def run_european():
    DATA_PATH = Path(__file__).parent / "Data" / "EUMD_holdings.csv"
    start_download = "2016-06-01"
    start_eval = "2018-01-01"
    end = "2020-01-01"
    tickers = get_tickers(DATA_PATH)
    prices = load_prices(tickers, start_download, end)
    rf = get_risk_free_rate(start=start_download, end=end)

    # Momentum strategy
    port_rets, positions, turnover = momentum_long_short(prices, lookback=189, topk=10, rebalance_period=21, tc_per_unit=0.001, max_weight=0.2)

    # Long only, equal weight benchmark
    bmark_rets = benchmark_long_only_equal_weight(prices)

    # slicing for actual period of testing
    port_rets = port_rets[start_eval:]
    bmark_rets = benchmark_long_only_equal_weight(prices)[start_eval:]
    rf = rf[start_eval:]
    # Stats reporting
    stats_port = perf_stats(port_rets, freq='day',
                            rf=rf, turnover=turnover, rebalance_period=21)
    stats_bmark = perf_stats(bmark_rets, freq='day', rf=rf, turnover=[0], rebalance_period=21)
    print(f"{'Metric':<22} {'Momentum':>12} {'EW Benchmark':>14}")
    print("-" * 50)
    for k in stats_port:
        print(f" {k:<20} {stats_port[k]:>12.4f} {stats_bmark[k]:>14.4f}")

    cum_port = (1 + port_rets).cumprod() - 1
    cum_bmark = (1 + bmark_rets).cumprod() - 1

    plt.figure(figsize=(11, 4))
    plt.plot(cum_port.index, cum_port.values, label="L/S Momentum", linewidth=2)
    plt.plot(cum_bmark.index, cum_bmark.values, label="EW Benchmark", linewidth=1.5,
             linestyle="--", color="gray")
    plt.title("Project 2: Momentum vs Equal-Weight Benchmark")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    factor_decomposition(port_rets, bmark_rets, rf)

    # print first rebalance positions
    print("\nSample positions (first rebalance):")
    print(positions[0].loc[positions[0] != 0].sort_values(ascending=True).to_string())

if __name__ == "__main__":
    run_european()