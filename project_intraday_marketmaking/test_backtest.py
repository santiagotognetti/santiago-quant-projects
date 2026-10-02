"""Unit tests for intraday backtest risk controls."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import backtest


def _make_bars(index: pd.DatetimeIndex, signal: list[float], ret: list[float],
               spread: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": signal,
            "ret": ret,
            "spread": spread,
        },
        index=index,
    )


def test_stop_loss_stays_flat_until_signal_clears():
    """Stop keeps the breach-bar PnL, then stays flat until desired pos returns to 0."""
    # 8 bars, same session. Confirmed long from bars 1-2 onward via signal persistence.
    # Position enters on bar after confirmation (shift 1).
    idx = pd.date_range("2024-01-02 10:00", periods=8, freq="min")
    # Raw signals already "confirmed-looking": non-zero for several bars.
    # backtest re-applies confirmation: need two consecutive equal non-zero signals.
    signal = [0, 1, 1, 1, 1, 1, 0, 0]
    # Large negative returns so cumulative trade PnL breaches stop quickly after entry
    ret = [0.0, 0.0, -0.001, -0.001, -0.001, -0.001, 0.0, 0.0]
    df = _make_bars(idx, signal, ret)

    out = backtest(df, stop_loss=0.0015)

    # Confirmation zeros bar 1 (no prior match); bars 2-5 keep signal 1; EOD not here
    # Position = confirmed_signal.shift(1); first bar forced flat
    # Entry around bar where confirmed signal has been live
    pos = out["pos"].tolist()

    # Find first non-zero position (trade entry)
    entry_idxs = [i for i, p in enumerate(pos) if p != 0]
    assert entry_idxs, "expected at least one entry"
    entry = entry_idxs[0]

    # Accumulate PnL on consecutive non-zero bars until stop would fire
    trade_pnl = 0.0
    stop_bar = None
    for i in range(entry, len(pos)):
        if pos[i] == 0 and stop_bar is None:
            # May be flat before entry completed — skip
            continue
        if stop_bar is None:
            trade_pnl += pos[i] * ret[i]
            if trade_pnl < -0.0015:
                stop_bar = i
                # Stop bar itself must still have the position (PnL kept)
                assert pos[i] != 0, "stop bar should keep position for PnL"
                break

    assert stop_bar is not None, "stop should have triggered"

    # Bars after stop while inherited desired position would still be non-zero → flat
    for i in range(stop_bar + 1, len(pos)):
        if signal[i - 1] == 0 if i > 0 else True:
            break
        # While prior confirmed signal still non-zero, stay flat after stop
        if out["signal"].iloc[i - 1] != 0:
            assert pos[i] == 0.0, f"expected flat after stop at bar {i}, got {pos[i]}"


def test_no_overnight_position_into_next_open():
    """Last-bar signal must not reopen a position at the next session open."""
    day1 = pd.date_range("2024-01-02 15:55", periods=5, freq="min")  # 15:55..15:59
    day2 = pd.date_range("2024-01-03 09:30", periods=3, freq="min")
    idx = day1.append(day2)

    # Confirmed long into the close on day 1
    signal = [1, 1, 1, 1, 1, 1, 1, 1]
    ret = [0.0] * 7 + [0.05]  # big open gap return on day2 bar 0 if wrongly held
    df = _make_bars(idx, signal, ret)

    out = backtest(df, stop_loss=1.0)  # disable stop economically

    # First bar of day 2 must be flat (no overnight)
    assert out.loc[day2[0], "pos"] == 0.0
    # And must not earn the gap return from a leftover overnight position
    assert out.loc[day2[0], "strategy_ret_raw"] == 0.0


def test_eod_signal_zeroed_before_next_day_shift():
    day1 = pd.date_range("2024-01-02 15:57", periods=3, freq="min")
    day2 = pd.date_range("2024-01-03 09:30", periods=2, freq="min")
    idx = day1.append(day2)
    signal = [1, 1, 1, 0, 0]
    ret = [0.0] * 5
    out = backtest(_make_bars(idx, signal, ret), stop_loss=1.0)

    # EOD bar's confirmed signal must be forced to 0
    assert out.loc[day1[-1], "signal"] == 0
    assert out.loc[day2[0], "pos"] == 0.0


if __name__ == "__main__":
    test_stop_loss_stays_flat_until_signal_clears()
    test_no_overnight_position_into_next_open()
    test_eod_signal_zeroed_before_next_day_shift()
    print("All intraday backtest tests passed.")
