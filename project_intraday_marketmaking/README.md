# Project 2 — Intraday Microstructure Mean-Reversion

**Status:** Research prototype (WIP)

A systematic intraday mean-reversion strategy on 1-minute bars. The signal fades
extreme deviations from the session cumulative VWAP, with explicit transaction
costs, per-trade stop-losses, and end-of-day flattening. Applied to SPY
(May–December 2024) via Polygon.io data.

---

## Core Idea

Temporary order-flow imbalances and microstructure noise can push prices away
from fair value on short horizons. If those dislocations mean-revert, a
contrarian signal — short when price is above VWAP, long when below — may
capture small, repeatable edges before costs erode them.

This project tests that hypothesis on liquid US equity ETF data using only OHLCV
1-minute bars (no Level 2 order book). The synthetic module plants a known
signal by construction and is used for pipeline validation only.

---

## Pipeline

```
Polygon 1-min bars
       │
       ▼
 prepare_bars()          ← session filter, resample, features
       │
       ▼
 compute_signal()        ← rolling z-score of VWAP deviation
       │
       ▼
 backtest()              ← confirmation filter, stop-loss, EOD flat, costs
       │
       ▼
 perf_stats() / build_trade_log()
```

### Signal logic (`compute_signal`)

1. **Feature:** intraday cumulative VWAP deviation — `(close − vwap) / vwap`,
   reset each trading day.
2. **Normalisation:** rolling z-score of `vwap_dev` within each session
   (default window: 60 bars).
3. **Entry:** fade extremes — short when `z > threshold`, long when
   `z < −threshold`.
4. **Open filter:** signals suppressed during the first 30 minutes (09:30–09:59 ET)
   to avoid directional open-auction flow.

### Execution & risk (`backtest`)

| Rule | Detail |
|---|---|
| Confirmation | Signal must persist for 2 consecutive bars before entry |
| Entry timing | Position entered on the bar after signal confirmation |
| Stop-loss | Per-trade cumulative PnL exit; stop-bar PnL kept, then flat until signal clears |
| End of day | EOD signal zeroed — no overnight carry into the next open |
| Costs | Half-spread proxy deducted on each position change |

---

## Setup

**Requirements:** Python 3.11+

```bash
pip install -r ../requirements.txt
```

**Polygon API key** (required for `run_live.py` only):

```bash
export POLYGON_API_KEY=your_key_here
```

Free-tier Polygon accounts are rate-limited to 5 requests/minute; the data
module handles pagination with automatic caching to `Data/*.parquet`.

---

## How to Run

```bash
# Synthetic demo — no API key, validates pipeline logic
python run_synthetic.py

# Live backtest on SPY — downloads/caches Polygon data, runs sensitivity + OOS
python run_live.py
```

Expected outputs: console performance tables, sensitivity grid results, trade-log
summary, time-of-day P&L breakdown, and cumulative-return plots.

---

## File Map

| File | Purpose |
|---|---|
| `core.py` | Simulation, signal generation, backtest engine, performance stats, trade log |
| `data.py` | Polygon fetch + bar cleaning / feature engineering |
| `run_synthetic.py` | Synthetic demo using `core.py` pipeline |
| `run_live.py` | Train/test split, parameter grid search, OOS evaluation, plots |
| `project1_intraday.py` | **Deprecated** — legacy imbalance z-score prototype (reference only) |
| `Data/` | Cached Polygon parquet files (created on first run) |

---

## Parameters & Methodology

### Train / test split

| Period | Dates | Bars |
|---|---|---|
| In-sample (train) | 2024-05-15 → 2024-10-31 | 46,077 |
| Out-of-sample (test) | 2024-11-01 → 2024-12-31 | 15,712 |

**Ticker:** SPY · **Bar size:** 1 minute · **Session:** 09:30–15:59 ET

### Sensitivity grid (in-sample)

| Parameter | Values tested |
|---|---|
| `z_window` | 15, 30, 60, 120, 240 |
| `z_thresh` | 0.5, 1.0, 1.5, 2.0 |
| `stop_loss` | 0.001, 0.0015, 0.002, 0.003 |

Optimal combination selected by maximum in-sample Sharpe ratio, then applied
without modification to the OOS window.

---

## Results

### In-Sample — Top 5 Parameter Combinations (2024-05-15 → 2024-10-31)

All 80 grid combinations produced **negative** Sharpe ratios. The table below
shows the five least negative (best in-sample):

| z_window | z_thresh | stop_loss | Sharpe | Sortino | Calmar | Hit Rate | Profit Factor |
|---|---|---|---|---|---|---|---|
| 240 | 1.5 | 0.003 | −6.73 | −7.47 | −1.72 | 40.7% | 0.72 |
| 240 | 1.5 | 0.002 | −6.74 | −7.83 | −1.72 | 40.5% | 0.72 |
| 240 | 1.5 | 0.0015 | −6.82 | −8.03 | −1.72 | 40.4% | 0.72 |
| 240 | 1.5 | 0.001 | −6.83 | −8.55 | −1.72 | 40.1% | 0.72 |
| 240 | 0.5 | 0.003 | −7.01 | −10.47 | −1.54 | 44.5% | 0.82 |

**Selected parameters:** `z_window=240`, `z_thresh=1.5`, `stop_loss=0.003`

### Out-of-Sample Performance (2024-11-01 → 2024-12-31)

| Metric | Value |
|---|---|
| Annualised Return | −63.95% |
| Annualised Volatility | 5.58% |
| Sharpe Ratio | −11.47 |
| Sortino | −6.77 |
| Calmar | −4.21 |
| Cumulative Return | −15.07% |
| Max Drawdown | 15.18% |
| Hit Rate | 41.4% |
| Profit Factor | 0.65 |
| Active bars (trades) | 3,311 |

### Trade-Level Summary (OOS)

| Metric | Value |
|---|---|
| Completed trades | 409 |
| Avg holding period | 7.3 bars (~7 min) |
| Win rate | 48.7% |
| Avg PnL per trade | −0.015% |
| Profit factor | 0.47 |

### P&L by Hour of Day (OOS, ET)

| Hour | Cumulative P&L |
|---|---|
| 09 | +0.81% |
| 10 | −5.23% |
| 11 | −3.19% |
| 12 | −1.51% |
| 13 | −0.84% |
| 14 | −2.20% |
| 15 | −3.31% |
| 16 | −0.84% |

Only the opening hour (post-filter) contributed positively; mid-session hours
were uniformly negative.

---

## Interpretation

The VWAP mean-reversion signal **does not appear profitable** on SPY over this
sample, net of spread costs:

1. **No in-sample edge.** Every parameter combination in the 80-cell grid
   produced negative Sharpe. Selecting the "best" in-sample parameters is
   choosing the least bad fit, not a genuine signal.
2. **OOS degradation.** Out-of-sample Sharpe (−11.47) is worse than
   in-sample (−6.73), consistent with overfitting to noise rather than
   extracting a stable microstructure premium.
3. **Costs dominate.** With ~409 trades over two months and a half-spread cost
   on each entry/exit, transaction frictions consume any raw signal. Hit rate
   near 50% with profit factor below 1 confirms the strategy loses more on
   losers than it gains on winners.
4. **Feature limitation.** The OHLC imbalance proxy and VWAP deviation derived
   from 1-minute aggregates may not capture the order-flow dynamics that
   drive short-horizon mean-reversion in the academic microstructure literature.

These results are reported honestly as a **negative finding** — the research
question was testable, the pipeline is end-to-end, and the conclusion is that
this particular signal specification does not survive realistic frictions on
SPY.

---

## Limitations

- Single ticker (SPY), short OOS window (2 months)
- No Level 2 / order book data — features are OHLCV proxies only
- Spread cost model is a bar-range proxy, not a calibrated market-impact model
- No latency, queue position, or partial-fill modelling
- In-sample selection on Sharpe when all values are negative is methodologically
  weak; a hold-out or walk-forward scheme would be more rigorous
- Survivorship and regime coverage not tested across multiple tickers or years

---

## Future Work

- Multi-ticker universe (sector ETFs, single-name large caps)
- Alternative features: order-flow imbalance from tick data, bid-ask bounce filters
- Longer evaluation window and walk-forward parameter selection
- Rank-based position sizing instead of binary ±1 signals
- Latency and realistic fill simulation
- Comparison against a naive buy-and-hold or session-VWAP benchmark

---

## Deprecated Code

`project1_intraday.py` is a legacy prototype that uses a simpler raw-imbalance
z-score signal and a combined `signal_and_backtest()` function. It is kept for
reference only — all new work should use `core.py` + `run_synthetic.py` /
`run_live.py`.
