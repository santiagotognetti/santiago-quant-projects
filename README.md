# Santiago Tognetti – Quant Research & Trading Projects

Welcome to my repository of quantitative finance research projects.  
These projects are designed to demonstrate skills relevant to **quantitative trading**, **market microstructure**, **statistical arbitrage**, and **systematic strategy development**.

I am currently completing the **Pre-Master in Econometrics & Quantitative Finance** at **Erasmus University Rotterdam**, with a background in Economics and professional experience in data analytics and financial modeling.

This repository contains two complete end-to-end quant research projects:

---

## Project 1 — Intraday Microstructure Mean-Reversion
**Folder:** `project1_intraday/`

A quantitative exploration of intraday market microstructure inefficiencies using imbalance-derived signals.  
Includes:

- Synthetic-data demo
- Microstructure imbalance feature construction
- Intraday mean-reversion model
- Rolling z-score signal generation
- Backtesting engine with cost & turnover handling
- Performance metrics (Sharpe, drawdown, annualized returns)
- Plots and cumulative results

**Core Idea:**  
Microstructure noise and temporary order-flow imbalances can cause short-lived inefficiencies.  
This project models those imbalances and tests whether they allow predictive intraday trading signals.

---

## Project 2 — Cross-Sectional Momentum Long/Short Strategy
**Folder:** `project2_momentum/`

A systematic long–short equity portfolio built on cross-sectional price momentum,
rebalanced monthly, applied to a European universe (~150 stocks from the STOXX
Europe 600). A 1-month skip is applied before scoring to avoid short-term reversal
contamination. Transaction costs of 10 bps per unit of turnover are modelled
explicitly.

Includes:

- Synthetic-data demo (500 simulated stocks, no external data required)
- Momentum factor computation with configurable lookback (default: 252 days)
- Cross-sectional ranking and score-proportional weight construction
- Long top-k / short bottom-k with max-weight cap (default: 20%)
- Monthly rebalancing with turnover tracking and transaction cost deduction
- Lookback sensitivity analysis across 8 horizons (21 → 378 days)
- CAPM factor decomposition (alpha, beta, R²)
- Equal-weight long-only benchmark comparison
- Performance analytics: Sharpe, Sortino, annualised return, max drawdown,
  annualised turnover

**Core Idea:**  
Stocks that have outperformed over the past 6–12 months tend to continue
outperforming over the next month (Jegadeesh & Titman, 1993). This project
implements a market-neutral version of that signal on European equities and
evaluates whether the premium survives realistic trading frictions.

**How to run:**

```bash
# Synthetic demo — no data download required
python project2_momentum/run_synthetic.py

# European universe — downloads via yfinance, requires EUMD_holdings.csv
python project2_momentum/run_european.py

# Lookback sensitivity analysis — runs full grid, plots diagnostics
python project2_momentum/run_sensitivity.py
```

**Key files:**

| File | Purpose |
|---|---|
| `core.py` | Strategy engine: momentum scoring, portfolio construction, performance stats, CAPM decomposition |
| `data.py` | Ticker parsing from EUMD holdings CSV, price download via yfinance |
| `run_synthetic.py` | Validates strategy logic on simulated cross-sectional data |
| `run_european.py` | Full backtest on European equity universe (2010–present) |
| `run_sensitivity.py` | Lookback grid search with Sharpe-optimal selection |

> **Note on results:** Figures and performance summaries are generated at runtime
> and printed to stdout / displayed as matplotlib charts. Backtest results
> are reported in-sample; a formal train/test split is on the roadmap.

### Out-of-Sample Results (2018–2020)

| Metric | L/S Momentum | EW Benchmark |
|---|---|---|
| Annualised Return | 0.49% | 8.68% |
| Annualised Volatility | 19.47% | 11.78% |
| Sharpe Ratio | 0.03 | 0.74 |
| Cumulative Return | 1.22% | 21.83% |
| Max Drawdown | 34.81% | 17.78% |
| Annual Turnover | 4.87× | — |

The out-of-sample period (2018–2020) coincides with two well-documented momentum
crash episodes: the Q4 2018 global equity selloff driven by Fed tightening and
trade war escalation, followed by a sharp, sudden reversal rally in Q1 2019.
Cross-sectional momentum strategies are structurally vulnerable to this pattern —
the short book is concentrated in previously underperforming stocks, which tend to
rebound most aggressively in a recovery. The strategy's 34.8% max drawdown versus
17.8% for the equal-weight benchmark reflects this exposure.

These results are consistent with the broader academic evidence on momentum crashes
(Daniel & Moskowitz, 2016). The high annual turnover of 4.87× also indicates that
transaction costs are a meaningful drag in a volatile, low-signal environment.
In-sample performance (2010–2018) is reported separately in the sensitivity
analysis section.

### Lookback Sensitivity Analysis (In-Sample: 2010–2018)

The strategy was evaluated across eight lookback horizons to identify the
optimal momentum formation period before the out-of-sample test window.

| Lookback (days) | Lookback (months) | Sharpe | Ann. Return | Max Drawdown | Ann. Turnover |
|---|---|---|---|---|---|
| 21 | 1 | –0.11 | –2.59% | 75.60% | 11.01× |
| 42 | 2 | 0.56 | 14.12% | 89.00% | 8.35× |
| 62 | 3 | 0.80 | 20.56% | 79.91% | 7.02× |
| 126 | 6 | 1.05 | 29.51% | 155.43% | 4.98× |
| **189** | **9** | **1.08** | **30.02%** | **91.17%** | **3.93×** |
| 252 | 12 | 1.01 | 27.18% | 156.86% | 3.19× |
| 315 | 15 | 0.71 | 18.30% | 97.92% | 2.90× |
| 378 | 18 | 0.45 | 11.50% | 63.52% | 2.56× |

The **9-month (189-day) lookback** maximises the in-sample Sharpe ratio at 1.08,
consistent with the Jegadeesh & Titman (1993) finding that intermediate-horizon
momentum (6–12 months) dominates short and long-term horizons. The 1-month
lookback produces negative returns, capturing short-term reversal rather than
momentum — a well-known empirical regularity. Turnover declines monotonically
with lookback length, as longer formation windows generate more stable rankings
and require fewer position changes at each rebalance.

The 6-month and 12-month lookbacks show suspiciously high max drawdown values
(>150%), which warrants investigation — this may reflect a period of concentrated
position flipping during a single drawdown episode rather than a structural
weakness of those horizons.

The optimal lookback (189 days) was selected purely on in-sample Sharpe and
applied without modification to the out-of-sample test period (2018–2020).


--



Contact details:
LinkedIn: https://www.linkedin.com/in/santiago-tognetti-57022a122/
Email: tognettisantiago@gmail.com
GitHub: https://github.com/gomerfield/


**Future Work**

Implement market microstructure models using real LOB or tick data.
Build intraday volatility forecasting models (HAR, GARCH).
Experiment with optimization-based portfolio construction (risk parity, volatility targeting).

If you find this repository useful or would like to discuss quant internships, feel free to reach out.


