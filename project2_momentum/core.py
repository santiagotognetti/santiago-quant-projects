import numpy as np
import pandas as pd
import statsmodels.api as sm
import pandas_datareader.data as web

def momentum_long_short(prices, lookback, topk, rebalance_period, tc_per_unit, max_weight):
    """
    Core function for the strategy. Detects top and bottom performers in the lookback period,
    defines a weight scheme based on performance,
    rebalances portfolio based on momemtum position, computes turnover, returns results.

    :param prices: stock prices
    :param lookback: lookback period in days
    :param topk: number of tickers at the top and bottom to be included in the portfolio (long and short positions)
    :param rebalance_period: rebalancing frequency
    :param tc_per_unit: transaction cost per unit of turnover (e.g. 0.001 = 10 bps round-trip)
    :param max_weight: maximum weight allowed to any specific ticker in the portfolio
    :return: returns, positions based on momentum and turnover stats
    """
    rets = prices.pct_change().fillna(0)
    skip_days = 21
    momentum = prices.pct_change(periods=lookback).shift(skip_days)

    rebalance_days = list(range(0, len(prices), rebalance_period))
    portfolio_rets = []
    turnover = []
    positions_store = []

    prev_pos = pd.Series(0.0, index=prices.columns)

    for i in rebalance_days[:-1]:
        start = i
        end = min(i+rebalance_period, len(prices)-1)
        mom_scores = momentum.iloc[start].dropna()

        if len(mom_scores) < 2 * topk:
            prev_pos = pd.Series(0.0, index=prices.columns)
            continue

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

        pos = pd.Series(0.0, index=prices.columns)

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

def perf_stats(returns: pd.Series, freq: str = 'day', rf: pd.Series | None = None, turnover=0, rebalance_period=21):
    """
    Creates stats study performance purposes.

    :param returns: returns in the portfolio
    :param freq: frequency, in this case trading days
    :param rf: risk-free rate
    :param turnover
    :param rebalance_period
    :return: annualized return, annualized volatility, sharpe ratio, cumulative returns, max drawdown in the period
     and annual turnover
    """
    if freq == 'day':
        ann_factor = 252
    else:
        ann_factor = 1

    if rf is not None:
        rf_aligned = rf.reindex(returns.index).ffill().fillna(0)
    else:
        rf_aligned = pd.Series(0.0, index=returns.index)

    excess_returns = returns - rf_aligned
    mean = (1 + excess_returns.mean()) ** 252 - 1
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = mean / vol if vol > 0 else np.nan

    # Sortino — downside deviation only (returns below rf)
    downside = excess_returns[excess_returns < 0]
    downside_vol = downside.std() * np.sqrt(ann_factor) if len(downside) > 1 else np.nan
    sortino = mean / downside_vol if downside_vol and downside_vol > 0 else np.nan

    cum_gross = (1 + returns).cumprod()
    maxdd = (cum_gross.cummax() - cum_gross).div(cum_gross.cummax()).max()

    cum = cum_gross - 1
    ann_turnover = np.mean(turnover) * (252 / rebalance_period)
    calmar = mean / maxdd if maxdd > 0 else np.nan

    return {
        "annualized_return": mean,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,  # ← new
        "cumulative_return": cum.iloc[-1],
        "max_drawdown": maxdd,
        "annual turnover": ann_turnover,
    }

def factor_decomposition(port_rets: pd.Series, bmark_rets: pd.Series,
                         rf: pd.Series) -> dict:
    """
    Regress strategy excess returns on market excess returns (CAPM).
    Reports alpha, beta, R-squared and t-stats.
    For a proper Fama-French decomposition, replace market_excess
    with a DataFrame of [MKT, SMB, HML] factors.
    """

    rf_aligned     = rf.reindex(port_rets.index).ffill().fillna(0)
    mkt_aligned    = bmark_rets.reindex(port_rets.index).ffill().fillna(0)

    port_excess = port_rets - rf_aligned
    mkt_excess  = mkt_aligned - rf_aligned

    X = sm.add_constant(mkt_excess)
    model = sm.OLS(port_excess, X).fit()

    alpha_daily = model.params["const"]
    beta        = model.params.iloc[1]
    alpha_ann   = (1 + alpha_daily) ** 252 - 1

    print("\n--- CAPM Factor Decomposition ---")
    print(f"  Annualized Alpha : {alpha_ann:.4f}  (t = {model.tvalues['const']:.2f}, "
          f"p = {model.pvalues['const']:.3f})")
    print(f"  Market Beta      : {beta:.4f}       (t = {model.tvalues.iloc[1]:.2f})")
    print(f"  R-squared        : {model.rsquared:.4f}")
    print(f"  Observations     : {int(model.nobs)}")
    print()
    print("  Interpretation:")
    if model.pvalues["const"] < 0.05:
        print(f"  ✓ Alpha is statistically significant — strategy adds value beyond market exposure.")
    else:
        print(f"  ✗ Alpha is NOT significant — returns are explained by market beta alone.")

    return {
        "alpha_annualized": alpha_ann,
        "beta":             beta,
        "r_squared":        model.rsquared,
        "alpha_tstat":      model.tvalues["const"],
        "alpha_pvalue":     model.pvalues["const"],
    }

def benchmark_long_only_equal_weight(prices: pd.DataFrame):
    """
    Long-only equal-weight benchmark over the same universe and date range
    as the momentum strategy. Rebalances monthly to maintain equal weights.
    No transaction costs applied (benchmark assumed frictionless).

    Returns daily portfolio returns as a pd.Series.
    """
    rets = prices.pct_change().fillna(0)
    n = prices.shape[1]
    weight = 1.0 / n
    return rets.mul(weight).sum(axis=1)

