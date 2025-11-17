import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_intraday(n_days=8, minutes_per_day=390):
    N = n_days * minutes_per_day
    sigma = 0.0008           # synthetic per-minute volatility
    returns = np.random.normal(loc=0, scale=sigma, size=N)
    midprice = 100 + np.cumsum(returns)
    lead_return = np.concatenate([returns[1:], [0]])
    imbalance = 0.5 * lead_return + np.random.normal(scale=0.0006, size=N)
    spread = 0.0005
    timestamps = pd.date_range(start="2025-01-02 09:00", periods=N, freq='T')
    df = pd.DataFrame({
        "mid": midprice,
        "ret": returns,
        "lead_ret": lead_return,
        "imbalance": imbalance,
        "spread": spread
    }, index=timestamps)
    return df

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
    df = simulate_intraday(n_days=12)
    df = signal_and_backtest(df, z_window=60, z_thresh=1.0, round_trip_cost=0.0005)

    stats = perf_stats(df['strategy_ret'], freq='min')
    cum = (1 + df['strategy_ret']).cumprod() - 1

    print("Sample rows:")
    print(df[['mid','ret','imbalance','imb_z','signal','pos','strategy_ret']].head(12).to_string())

    print("\nPerformance (synthetic):")
    for k,v in stats.items():
        print(f"  {k}: {v:.6f}")

    # Plot cumulative returns
    plt.figure(figsize=(10,4))
    plt.plot(cum.index, cum.values)
    plt.title("Project1 (Synthetic): Intraday strategy cumulative return")
    plt.xlabel("Time")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_demo()