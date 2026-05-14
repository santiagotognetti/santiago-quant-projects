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
outperforming over the next month (Jegadeesh & Titman, 1993). This project
implements a market-neutral (dollar-neutral; beta not explicitly hedged) version of that signal on European equities and
evaluates whether the premium survives realistic trading frictions.
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
(Daniel & Moskowitz, 2016). In-sample performance (2010–2018) is reported
separately in the sensitivity analysis section.

These results are consistent with the broader academic evidence on momentum crashes
(Daniel & Moskowitz, 2016). The high annual turnover of 9.39× also indicates that
transaction costs are a meaningful drag in a volatile, low-signal environment.
In-sample performance (2010–2018) is reported separately in the sensitivity
analysis section.

### Lookback Sensitivity Analysis (In-Sample: 2010–2018)

The strategy was evaluated across eight lookback horizons to identify the
optimal momentum formation period before the out-of-sample test window.
In-sample (2010–2018), the strategy achieved a Sharpe of 1.06 at a 189-day
lookback with 29.4% annualised return net of costs.

| Lookback (days) | Months | Sharpe | Sortino | Calmar | Ann. Return | Max Drawdown | Ann. Turnover |
|---|---|---|---|---|---|---|---|
| 21 | 1 | –0.16 | –0.22 | –0.06 | –3.76% | 63.82% | 21.90× |
| 42 | 2 | 0.52 | 0.74 | 0.33 | 13.06% | 39.24% | 16.58× |
| 62 | 3 | 0.77 | 1.06 | 0.62 | 19.60% | 31.66% | 13.91× |
| 126 | 6 | 1.02 | 1.46 | 0.93 | 28.77% | 31.05% | 9.83× |
| **189** | **9** | **1.06** | **1.53** | **0.88** | **29.44%** | **33.46%** | **7.72×** |
| 252 | 12 | 1.00 | 1.42 | 0.78 | 26.74% | 34.30% | 6.25× |
| 315 | 15 | 0.70 | 1.02 | 0.62 | 17.97% | 29.04% | 5.66× |
| 378 | 18 | 0.44 | 0.65 | 0.35 | 11.29% | 32.28% | 4.97× |

The **9-month (189-day) lookback** maximises the in-sample Sharpe ratio at 1.06
and Sortino at 1.53, consistent with the Jegadeesh & Titman (1993) finding that
intermediate-horizon momentum (6–12 months) dominates short and long-term
horizons. The 1-month lookback produces negative returns across all three
risk-adjusted metrics, capturing short-term reversal rather than momentum — a
well-known empirical regularity. Turnover declines monotonically with lookback
length, as longer formation windows generate more stable rankings and require
fewer position changes at each rebalance.

Note that the Calmar ratio peaks at 6 months (0.93) rather than 9 months (0.88),
reflecting a slightly lower max drawdown at that horizon. The 9-month lookback
was selected on Sharpe as the primary criterion; a practitioner weighting
drawdown control more heavily might prefer 6 months.

The optimal lookback (189 days) was selected purely on in-sample Sharpe and
applied without modification to the out-of-sample test period (2018–2020).



**Future Work**

Beta-weighting positions



--



## Contact details:
### LinkedIn: https://www.linkedin.com/in/santiago-tognetti-57022a122/
### Email: tognettisantiago@gmail.com
### GitHub: https://github.com/santiagotognetti/


--



