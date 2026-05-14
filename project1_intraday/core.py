import numpy as np
import pandas as pd

def simulate_intraday(n_days: int = 5000, minutes_per_day: int = 390, seed: int = 42) -> pd.DataFrame:
    """
    Simulate intraday microstructure data with causal structure:
    imbalance(t) predicts return(t+1).
    """
    rng = np.random.default_rng(seed)
    N = n_days * minutes_per_day
    sigma = 0.0008

    imb_noise = rng.normal(scale=0.0006, size=N)
    imbalance = np.zeros(N)
    for t in range(1, N):
        imbalance[t] = 0.7 * imbalance[t - 1] + imb_noise[t]  # AR(1) persistence
    ret_noise = rng.normal(scale=sigma, size=N)
    returns = np.zeros(N)
    for t in range(1, N):
        returns[t] = 0.25 * imbalance[t - 1] + ret_noise[t]

    midprice = 100 + np.cumsum(returns)
    spread = 0.0005
    timestamps = pd.date_range(start="2025-01-02 09:00", periods=N, freq='min')

    return pd.DataFrame({
        "mid": midprice,
        "ret": returns,
        "imbalance": imbalance,
        "spread": spread
    }, index=timestamps)

def signal_and_backtest(df, z_window=60, z_thresh=1.0, round_trip_cost=0.0005):
    # feature: rolling zscore of imbalance
    df = df.copy()
    rolling_mean = df['imbalance'].rolling(z_window, min_periods=1).mean()
    rolling_std = df['imbalance'].rolling(z_window, min_periods=1).std().replace(0, 1e-8)
    df['imb_z'] = (df['imbalance'] - rolling_mean) / rolling_std

    # signal: simple mean-reversion on extreme imbalance
    df['signal'] = 0
    df.loc[df['imb_z'] > z_thresh, 'signal'] = -1
    df.loc[df['imb_z'] < -z_thresh, 'signal'] = 1

    # position is previous signal (enter next bar)
    df['pos'] = df['signal'].shift(1).fillna(0)

    # returns and transaction cost when position changes
    df['strategy_ret_raw'] = df['pos'] * df['ret']
    df['trade'] = (df['pos'] != df['pos'].shift(1)).astype(int)
    df['strategy_ret'] = df['strategy_ret_raw'] - df['trade'] * round_trip_cost

    return df

def perf_stats(returns, freq='min'):
    if freq == 'min':
        ann_factor = 252 * 6.5 * 60
    else:
        ann_factor = 252
    mean = returns.mean() * ann_factor
    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = mean/vol if vol > 0 else np.nan
    cum = (1 + returns).cumprod()
    maxdd = (cum.cummax() - cum).max()
    return {
        "annualized_return": mean,
        "annualized_vol": vol,
        "sharpe": sharpe,
        "cumulative_return": cum.iloc[-1],
        "max_drawdown": maxdd
    }

def build_trade_log():

