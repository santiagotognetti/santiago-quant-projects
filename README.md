#### Python 3.11+

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
- Performance metrics (Sharpe, drawdown, Sortino, Calmar, annualized returns)
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

- Synthetic-data demo (500 simulated stocks, no external data required). Results from this module arise from noise, and was only placed for testing logic
- Momentum factor computation with configurable lookback (default: 189 days, selected via in-sample sensitivity analysis)
- Cross-sectional ranking and score-proportional weight construction
- Long top-k / short bottom-k with max-weight cap (default: 20%)
- Monthly rebalancing with turnover tracking and transaction cost deduction
- Lookback sensitivity analysis across 8 horizons (21 → 378 days)
- CAPM factor decomposition (alpha, beta, R²)
- Equal-weight long-only benchmark comparison
- Performance analytics: Sharpe, Sortino, Calmar, annualised return, max drawdown,
  annualised turnover

**Core Idea:**  
Stocks that have outperformed over the past 6–12 months tend to continue
outperforming over the next month (Jegadeesh & Titman, 1993). This project implements a dollar-neutral (but not beta-hedged) 
version of that signal on European equities and evaluates whether the premium survives realistic trading frictions.
Note: universe is subject to survivorship bias — only current STOXX Europe 600 constituents are included.


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


### Out-of-Sample Results (2018–2020)

| Metric | L/S Momentum | EW Benchmark |
|---|---|---|
| Annualised Return | 2.43% | 11.25% |
| Annualised Volatility | 19.46% | 11.78% |
| Sharpe Ratio | 0.12 | 0.95 |
| Sortino | 0.18 | 1.33 |
| Calmar | 0.08 | 0.69 |
| Cumulative Return | 0.33% | 21.83% |
| Max Drawdown | 28.69% | 16.37% |
| Annual Turnover | 9.39× | — |

The out-of-sample period (2018–2020) coincides with two well-documented momentum
crash episodes: the Q4 2018 global equity selloff driven by Fed tightening and
trade war escalation, followed by a sharp, sudden reversal rally in Q1 2019.
Cross-sectional momentum strategies are structurally vulnerable to this pattern —
the short book is concentrated in previously underperforming stocks, which tend to
rebound most aggressively in a recovery. The strategy's 28.7% max drawdown versus
16.4% for the equal-weight benchmark reflects this exposure.

The CAPM decomposition yields an annualised alpha of 4.6% (t = 0.33, p = 0.739),
which is not statistically significant — returns over this period are not
explained by market exposure either, given a market beta of –0.19. The low
R-squared of 0.013 confirms the strategy is largely market-neutral, as intended.

These results are consistent with the broader academic evidence on momentum crashes
(Daniel & Moskowitz, 2016). The high annual turnover of 9.39× also indicates that
transaction costs are a meaningful drag in a volatile, low-signal environment.
In-sample performance (2010–2018) is reported separately in the sensitivity
analysis section.

### Lookback Sensitivity Analysis (In-Sample: 2010–2018)

The strategy was evaluated across eight lookback horizons to identify the
optimal momentum formation period before the out-of-sample test window.
All lookbacks are evaluated over a common window starting from the first
date at which the longest formation period (378 days + 21-day skip) has a
valid signal, ensuring metrics are directly comparable across horizons.
In-sample, the strategy achieved a Sharpe of 0.89 at a 189-day lookback
with 24.1% annualised return net of costs.

| Lookback (days) | Months | Sharpe | Sortino | Calmar | Ann. Return | Max Drawdown | Ann. Turnover |
|---|---|---|---|---|---|---|---|
| 21 | 1 | –0.20 | –0.29 | –0.07 | –4.65% | 63.82% | 21.90× |
| 42 | 2 | 0.40 | 0.60 | 0.25 | 9.92% | 39.24% | 16.58× |
| 62 | 3 | 0.70 | 1.03 | 0.55 | 17.37% | 31.66% | 13.91× |
| 126 | 6 | 0.81 | 1.21 | 0.72 | 22.21% | 31.05% | 9.83× |
| **189** | **9** | **0.89** | **1.31** | **0.72** | **24.11%** | **33.46%** | **7.72×** |
| 252 | 12 | 0.80 | 1.15 | 0.62 | 21.22% | 34.30% | 6.25× |
| 315 | 15 | 0.51 | 0.75 | 0.45 | 13.15% | 29.04% | 5.66× |
| 378 | 18 | 0.44 | 0.65 | 0.35 | 11.29% | 32.28% | 4.97× |

The **9-month (189-day) lookback** maximises Sharpe at 0.89 and Sortino at
1.31, consistent with Jegadeesh & Titman (1993). The 1-month lookback
produces negative returns across all three risk-adjusted metrics, capturing
short-term reversal rather than momentum — a well-known empirical
regularity. Turnover declines monotonically with lookback length, as longer
formation windows generate more stable rankings.

The decline from an in-sample Sharpe of 0.89 to 0.12 out-of-sample is consistent 
with both overfitting to a benign 2010–2018 training environment and the known
fragility of momentum strategies during sharp reversals.

Note that the Calmar ratio peaks jointly at 6 months (0.72) and 9 months
(0.72), reflecting near-identical drawdown control at those two horizons.
The 9-month lookback was selected on Sharpe as the primary criterion.

The optimal lookback (189 days) was selected purely on in-sample Sharpe and
applied without modification to the out-of-sample test period (2018–2020).




**Future Work**
- Beta-weighted position sizing to reduce the observed market beta of –0.19
- Rank-based weights instead of score-proportional weights to eliminate
  the clipping discontinuity at zero momentum
- Extension of the evaluation window beyond 2020 to cover the post-COVID
  momentum recovery period




--



## Contact details:
### LinkedIn: https://www.linkedin.com/in/santiago-tognetti-57022a122/
### Email: tognettisantiago@gmail.com
### GitHub: https://github.com/santiagotognetti/


--



