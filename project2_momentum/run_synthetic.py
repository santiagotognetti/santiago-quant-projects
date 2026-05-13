import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from core import (momentum_long_short, perf_stats, factor_decomposition,
                  benchmark_long_only_equal_weight, get_risk_free_rate)

def simulate_cross_section(n_stocks=500, n_days=5200):
    """
    Simulate cross sectional price data stochastically.
    :param n_stocks: number of tickers to simulate
    :param n_days: number of trading days to simulate
    :return: prices data for specified number of stocks in specified number of days
    """
    rng = np.random.default_rng(42)
    mu = 0.0004 / 252
    sigma_market = 0.15 / np.sqrt(252)
    sigma_idio = 0.20 / np.sqrt(252)
    stock_drifts = rng.uniform(-0.0003, 0.0003, size=n_stocks)
    market = np.random.normal(loc=mu, scale=sigma_market, size=n_days)
    idios = np.random.normal(loc=0, scale=sigma_idio, size=(n_days, n_stocks))
    returns = market.reshape(-1, 1) + idios + stock_drifts.reshape(1, -1)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.date_range(start="2010-01-01", periods=n_days, freq='B')
    df_prices = pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_stocks)])
    return df_prices

def run_synthetic():
    start = "2000-01-01"
    end = dt.date.today()
    prices = simulate_cross_section(n_stocks=500, n_days=5200)
    rf = get_risk_free_rate(start=start, end=end)

    # Momentum strategy
    port_rets, positions, turnover = momentum_long_short(prices, lookback=252, topk=10, rebalance_period=21, tc_per_unit=0.001, max_weight=0.2)

    # Long only, equal weight benchmark
    bmark_rets = benchmark_long_only_equal_weight(prices)

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
    run_synthetic()