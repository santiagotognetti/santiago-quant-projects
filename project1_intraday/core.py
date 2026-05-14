import numpy as np
import pandas as pd

def simulate_intraday(n_days: int = 252, minutes_per_day: int = 390, seed: int = 42) -> pd.DataFrame:
    """
    Simulate intraday microstructure data with causal structure:
    imbalance at time t predicts return at time t+1 via AR(1) model for order-flow persistence.

    Imbalance resets at each day boundary — no overnight contamination.

    NOTE: pipeline validation only. Signal is planted by construction.
    Results do not constitute evidence the strategy works on real data.
    """
    rng = np.random.default_rng(seed)
    sigma = 0.0008
    all_rows = []
    price = 100.0

    for day in range(n_days):
        date = pd.Timestamp("2020-01-02") + pd.offsets.BDay(day)
        times = pd.date_range(
            start=date.replace(hour=9, minute=30),
            periods=minutes_per_day,
            freq="min"
        )
        N = minutes_per_day
        imb_noise = rng.normal(scale=0.0006, size=N)
        imbalance = np.zeros(N)
        for t in range(1, N):
            imbalance[t] = 0.7 * imbalance[t - 1] + imb_noise[t]

        ret_noise = rng.normal(scale=sigma, size=N)
        rets = np.zeros(N)
        for t in range(1, N):
            rets[t] = 0.25 * imbalance[t - 1] + ret_noise[t]

        mids = price * np.exp(np.cumsum(rets))
        price = mids[-1]  # price carries over; imbalance resets

        day_df = pd.DataFrame({
            "mid":       mids,
            "ret":       rets,
            "imbalance": imbalance,
            "spread":    0.0005,
        }, index=times)
        all_rows.append(day_df)

    return pd.concat(all_rows)

def compute_signal(df: pd.DataFrame, z_window: int = 60, z_thresh: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["imb_z"] = 0.0

    for date, group in df.groupby(df.index.date):
        idx = group.index
        rm = group["imbalance"].rolling(z_window, min_periods=1).mean()
        rs = group["imbalance"].rolling(z_window, min_periods=1).std().replace(0, 1e-8)
        df.loc[idx, "imb_z"] = ((group["imbalance"] - rm) / rs).values

    df["signal"] = 0
    df.loc[df["imb_z"] >  z_thresh, "signal"] = -1
    df.loc[df["imb_z"] < -z_thresh, "signal"] =  1
    return df

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute signal as a backtest. Position enters at next bar (shift(1)) to prevent lookahead bias
    Forced flat at the last bar of each trading day
    Transaction cost are half-spread per side of the trade, charged on every position change
    """
    df = df.copy()

    raw_pos = df["signal"].shift(1).fillna(0)

    # Force flat at end of each trading day
    eod_idx = df.groupby(df.index.date).apply(lambda g: g.index[-1]).values
    raw_pos.loc[eod_idx] = 0.0
    df["pos"] = raw_pos

    df["strategy_ret_raw"] = df["pos"] * df["ret"]

    pos_change = df["pos"].diff().abs().fillna(df["pos"].abs())
    df["trade"] = (pos_change > 0).astype(int)
    df["tc"]    = df["trade"] * (df["spread"] / 2)
    df["strategy_ret"] = df["strategy_ret_raw"] - df["tc"]

    return df

def perf_stats(returns: pd.Series, freq: str = "min", rf: float = 0.0) -> dict:
    """
    Annualised performance statistics.

    :param returns: per-bar strategy returns
    :param freq: 'min' for 1-minute bars, 'day' for daily
    :param rf: annualised risk-free rate
    """
    if len(returns) == 0:
        raise ValueError("perf_stats received an empty returns Series. "
                         "Check that df_train is not empty after slicing.")
    ann_factor = 252 * 390 if freq == "min" else 252
    rf_per_bar = (1 + rf) ** (1 / ann_factor) - 1
    excess     = returns - rf_per_bar

    mean_ann = (1 + excess.mean()) ** ann_factor - 1
    vol_ann  = returns.std() * np.sqrt(ann_factor)
    sharpe   = mean_ann / vol_ann if vol_ann > 0 else np.nan

    downside = excess[excess < 0]
    dv       = downside.std() * np.sqrt(ann_factor) if len(downside) > 1 else np.nan
    sortino  = mean_ann / dv if dv and dv > 0 else np.nan

    cum    = (1 + returns).cumprod()
    maxdd  = ((cum.cummax() - cum) / cum.cummax()).max()
    calmar = mean_ann / maxdd if maxdd > 0 else np.nan

    active        = returns[returns != 0]
    winning       = active[active > 0]
    losing        = active[active < 0]
    hit_rate      = len(winning) / len(active) if len(active) > 0 else np.nan
    profit_factor = winning.sum() / abs(losing.sum()) if len(losing) > 0 else np.nan

    return {
        "annualized_return" : mean_ann,
        "annualized_vol"    : vol_ann,
        "sharpe"            : sharpe,
        "sortino"           : sortino,
        "calmar"            : calmar,
        "cumulative_return" : cum.iloc[-1] - 1,
        "max_drawdown"      : maxdd,
        "hit_rate"          : hit_rate,
        "profit_factor"     : profit_factor,
        "n_trades"          : int(len(active)),
    }

def build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a trade level log from bar level backtest output.
    Each row is one complete trade (entry through exit).

    :param df: output of backtest(), must contain [pos, strategy_ret]
    :return: DataFrame with [entry_time, exit_time, direction, pnl, n_bars]
    """
    trades     = []
    in_trade   = False
    entry_time = None
    direction  = 0
    pnl_bars   = []

    for ts, row in df.iterrows():
        if not in_trade and row["pos"] != 0:
            in_trade   = True
            entry_time = ts
            direction  = int(row["pos"])
            pnl_bars   = [row["strategy_ret"]]
        elif in_trade and row["pos"] == direction:
            pnl_bars.append(row["strategy_ret"])
        elif in_trade and row["pos"] != direction:
            trades.append({
                "entry_time" : entry_time,
                "exit_time"  : ts,
                "direction"  : "long" if direction == 1 else "short",
                "pnl"        : sum(pnl_bars),
                "n_bars"     : len(pnl_bars),
            })
            in_trade = False
            if row["pos"] != 0:
                in_trade   = True
                entry_time = ts
                direction  = int(row["pos"])
                pnl_bars   = [row["strategy_ret"]]

    return pd.DataFrame(trades)
