"""
Project 2: Cross-Sectional Momentum (demo script)
- Synthetic-data demo included. Replace with real price data (yfinance, exchange CSVs).
- Strategy: monthly rebalanced long-top / short-bottom on 60-day momentum.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web

def simulate_cross_section(n_stocks=100, n_days=5200):
    rng = np.random.default_rng(42)
    mu = 0.0004 / 252
    sigma_market = 0.01 / np.sqrt(252)
    sigma_idio = 0.02 / np.sqrt(252)
    stock_drifts = rng.uniform(-0.0003, 0.0003, size=n_stocks)
    market = np.random.normal(loc=mu, scale=sigma_market, size=n_days)
    idios = np.random.normal(loc=0, scale=sigma_idio, size=(n_days, n_stocks))
    returns = market.reshape(-1, 1) + idios + stock_drifts.reshape(1, -1)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.date_range(start="2023-01-02", periods=n_days, freq='B')
    df_prices = pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_stocks)])
    return df_prices

def momentum_long_short(prices, lookback, topk, rebalance_period, tc_per_unit, max_weight):
    rets = prices.pct_change().fillna(0)
    momentum = prices.pct_change(periods=lookback).shift(rebalance_period).fillna(0)

    rebalance_days = list(range(0, len(prices), rebalance_period))
    portfolio_rets = []
    turnover = []
    positions_store = []

    prev_pos = pd.Series(0.0, index=prices.columns)

    for i in rebalance_days[:-1]:
        start = i
        end = min(i+rebalance_period, len(prices)-1)
        mom_scores = momentum.iloc[start]
        top = mom_scores.nlargest(topk).index.tolist()
        bottom = mom_scores.nsmallest(topk).index.tolist()


        long_scores = mom_scores[top].clip(lower=0)
        short_scores = mom_scores[bottom].clip(upper=0).abs()
        long_weights = long_scores / long_scores.sum() if long_scores.sum() > 0 else pd.Series(1 / len(top), index=top)
        short_weights = short_scores / short_scores.sum() if short_scores.sum() > 0 else pd.Series(1 / len(bottom), index=bottom)
        long_weights = (long_weights.clip(upper=max_weight))
        long_weights = long_weights / long_weights.sum()

        short_weights = (short_weights.clip(upper=max_weight))
        short_weights = short_weights / short_weights.sum()

        pos = pd.Series(0.0, index=prices.columns)  # ← defined here, too late

        pos[top] = long_weights
        pos[bottom] = -short_weights



        # turnover proportional to changes in absolute position

        gross_exposure = pos.abs().sum() + prev_pos.abs().sum()
        tr = (pos.subtract(prev_pos).abs()).sum() / gross_exposure if gross_exposure > 0 else 0.0
        turnover.append(tr)

        # apply daily returns for holding period
        period_rets = rets.iloc[start+1:end+1]
        daily_port_returns = (period_rets * pos).sum(axis=1)

        # simple transaction cost hit on rebalance (applied once per rebalance)
        tc = tr * tc_per_unit
        # subtract amortized cost over holding period simply (quick demo)
        daily_port_returns = daily_port_returns - tc / max(1, (end - start))

        portfolio_rets.append(daily_port_returns)
        positions_store.append(pos)
        prev_pos = pos.copy()

    portfolio_rets = pd.concat(portfolio_rets)
    return portfolio_rets, positions_store, turnover

def get_risk_free_rate(start: str, end: str) -> pd.Series:
    """
    Fetch daily 3-month T-bill rate from FRED (annualized, in decimal).
    Ticker TB3MS.
    """
    rf = web.DataReader('TB3MS', 'fred', start, end)['TB3MS']
    rf = rf / 100
    rf = rf.resample('B').ffill()
    return rf / 252

def perf_stats(returns: pd.Series, freq: str = 'day', rf: pd.Series | None = None, turnover = 0, rebalance_period = 21):
    if freq == 'day':
        ann_factor = 252
    else:
        ann_factor = 1

    if rf is not None:
        rf_aligned = rf.reindex(returns.index).ffill().fillna(0)
    else:
        rf_aligned = pd.Series(0.0, index=returns.index)

    excess_returns = returns - rf_aligned
    mean = (1 + excess_returns.mean())**252 - 1
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else np.nan
    cum = (1 + returns).cumprod() - 1
    maxdd = (cum.cummax() - cum).max()
    ann_turnover = np.mean(turnover) * (252 / rebalance_period)
    return {
        "annualized_return": mean,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "cumulative_return": cum.iloc[-1],
        "max_drawdown": maxdd,
        "annual turnover": ann_turnover,
    }

def run_demo():
    tickers = get_sp500_tickers(n=500)
    prices = load_prices(tickers, start="2010-01-01", end="2024-12-31")
    port_rets, positions, turnover = momentum_long_short(prices, lookback=252, topk=25, rebalance_period=21, tc_per_unit=0.001, max_weight=0.2)
    rf = get_risk_free_rate(start="2010-01-01", end="2024-12-31")
    stats = perf_stats(port_rets, freq='day', rf=rf, turnover=turnover, rebalance_period=21)
    cum = (1 + port_rets).cumprod() - 1
    print("Project 2 Performance:")
    for k,v in stats.items():
        print(f"  {k}: {v:.6f}")


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
    run_demo()