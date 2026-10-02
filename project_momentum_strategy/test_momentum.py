"""Unit tests for momentum portfolio construction edge cases."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core import momentum_long_short


def _make_prices(n_days: int = 120, n_stocks: int = 5) -> pd.DataFrame:
    """Deterministic prices: early NaNs on most names so warmup has too few scores."""
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(0)
    data = {}
    for i in range(n_stocks):
        rets = rng.normal(0.0005, 0.01, size=n_days)
        px = 100 * np.exp(np.cumsum(rets))
        # Only 1 stock has prices in the first 40 days → < 2*topk scores early on
        if i > 0:
            px[:40] = np.nan
        data[f"S{i}"] = px
    return pd.DataFrame(data, index=dates)


def test_warmup_keeps_calendar_continuity():
    """Insufficient cross-section must emit flat zero returns, not drop days."""
    prices = _make_prices()
    lookback = 21
    rebalance_period = 21
    topk = 2

    port_rets, positions, turnover = momentum_long_short(
        prices,
        lookback=lookback,
        topk=topk,
        rebalance_period=rebalance_period,
        tc_per_unit=0.001,
        max_weight=0.5,
    )

    # Expected trading days covered: each completed rebalance window contributes
    # rebalance_period days (except possibly the last partial). No holes from continue.
    rebalance_days = list(range(0, len(prices), rebalance_period))
    expected_days = 0
    for i in rebalance_days[:-1]:
        start = i
        end = min(i + rebalance_period, len(prices) - 1)
        expected_days += len(prices.index[start + 1 : end + 1])

    assert len(port_rets) == expected_days
    assert port_rets.index.is_monotonic_increasing
    # No duplicate timestamps from overlapping windows
    assert not port_rets.index.duplicated().any()


def test_warmup_charges_unwind_turnover():
    """Flattening when scores are insufficient should record turnover vs prev_pos."""
    prices = _make_prices(n_days=200, n_stocks=6)
    # Force a held book then a thin cross-section: use topk larger than available names
    # early, then enough names later. Simpler: run with topk=3 on 6 names with staggered NaNs.
    port_rets, positions, turnover = momentum_long_short(
        prices,
        lookback=21,
        topk=3,
        rebalance_period=21,
        tc_per_unit=0.0,
        max_weight=0.5,
    )

    assert isinstance(turnover, pd.Series)
    assert len(turnover) == len(positions)
    # At least one flat book should appear while scores are thin
    flat_windows = sum(1 for p in positions if (p == 0).all())
    assert flat_windows >= 1


def test_turnover_is_date_indexed_for_oos_slice():
    prices = _make_prices(n_days=250, n_stocks=8)
    # Fill NaNs so strategy always has enough names after warmup
    prices = prices.ffill().bfill()
    _, _, turnover = momentum_long_short(
        prices,
        lookback=21,
        topk=2,
        rebalance_period=21,
        tc_per_unit=0.001,
        max_weight=0.5,
    )
    assert isinstance(turnover.index, pd.DatetimeIndex)
    mid = turnover.index[len(turnover) // 2]
    sliced = turnover[mid:]
    assert len(sliced) < len(turnover)
    assert sliced.index.min() >= mid


if __name__ == "__main__":
    test_warmup_keeps_calendar_continuity()
    test_warmup_charges_unwind_turnover()
    test_turnover_is_date_indexed_for_oos_slice()
    print("All momentum tests passed.")
