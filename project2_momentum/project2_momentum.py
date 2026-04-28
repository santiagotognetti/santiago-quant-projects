"""
Project 2: Cross-Sectional Momentum (demo script)
- Synthetic-data demo included. Replace with real price data (yfinance, exchange CSVs).
- Strategy: monthly rebalanced long-top / short-bottom on 60-day momentum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_cross_section(n_stocks=100, n_days=5200):
    mu = 0.0004 / 252
    sigma_market = 0.01 / np.sqrt(252)
    sigma_idio = 0.02 / np.sqrt(252)
    market = np.random.normal(loc=mu, scale=sigma_market, size=n_days)
    idios = np.random.normal(loc=0, scale=sigma_idio, size=(n_days, n_stocks))
    returns = market.reshape(-1,1) + idios
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.date_range(start="2023-01-02", periods=n_days, freq='B')
    df_prices = pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_stocks)])
    return df_prices

def momentum_long_short(prices, lookback=60, topk=10, rebalance_period=21, tc_bps=0.001):
    rets = prices.pct_change().fillna(0)
    momentum = prices.pct_change(periods=lookback).shift(1).fillna(0)

    rebalance_days = list(range(0, len(prices), rebalance_period))
    portfolio_rets = []
    turnover = []
    positions_store = []

    prev_pos = pd.Series(0, index=prices.columns)

    for i in rebalance_days[:-1]:
        start = i
        end = min(i+rebalance_period, len(prices)-1)
        mom_scores = momentum.iloc[start]
        top = mom_scores.nlargest(topk).index.tolist()
        bottom = mom_scores.nsmallest(topk).index.tolist()

        pos = pd.Series(0.0, index=prices.columns)
        pos[top] = 1/len(top)
        pos[bottom] = -1/len(bottom)

        # turnover proportional to changes in absolute position
        tr = (pos.subtract(prev_pos).abs()).sum() / 2.0  # fraction of portfolio turned
        turnover.append(tr)

        # apply daily returns for holding period
        period_rets = rets.iloc[start+1:end+1]
        daily_port_returns = (period_rets * pos).sum(axis=1)

        # simple transaction cost hit on rebalance (applied once per rebalance)
        tc = tr * tc_bps
        # subtract amortized cost over holding period simply (quick demo)
        daily_port_returns = daily_port_returns - tc / max(1, (end - start))

        portfolio_rets.append(daily_port_returns)
        positions_store.append(pos)
        prev_pos = pos.copy()

    portfolio_rets = pd.concat(portfolio_rets)
    return portfolio_rets, positions_store

def perf_stats(returns, freq='day'):
    if freq == 'day':
        ann_factor = 252
    else:
        ann_factor = 1
    mean = returns.mean() * ann_factor
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else np.nan
    cum = (1 + returns).cumprod() - 1
    maxdd = (cum.cummax() - cum).max()
    return {
        "annualized_return": mean,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "cumulative_return": cum.iloc[-1],
        "max_drawdown": maxdd
    }

def run_demo():
    prices = simulate_cross_section(n_stocks=100, n_days=520)
    port_rets, positions = momentum_long_short(prices, lookback=60, topk=10, rebalance_period=21, tc_bps=0.001)
    stats = perf_stats(port_rets, freq='day')

    cum = (1 + port_rets).cumprod() - 1
    print("Project 2 Performance (Synthetic):")
    for k,v in stats.items():
        print(f"  {k}: {v:.6f}")

    plt.figure(figsize=(10,4))
    plt.plot(cum.index, cum.values)
    plt.title("Project2 (Synthetic): Cross-Sectional Momentum Cumulative Return")
    plt.xlabel("Date")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # print first rebalance positions
    print("\nSample positions (first rebalance):")
    print(positions[0].loc[positions[0] != 0].sort_values(ascending=False).to_string())

if __name__ == "__main__":
    run_demo()