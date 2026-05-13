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


--
Project 1:
python project1_intraday/project1_intraday.py

Project 2:
python project2_momentum/project2_momentum.py

Contact details:
LinkedIn: https://www.linkedin.com/in/santiago-tognetti-57022a122/
Email: tognettisantiago@gmail.com
GitHub: https://github.com/gomerfield/


**Future Work**

Implement market microstructure models using real LOB or tick data.
Build intraday volatility forecasting models (HAR, GARCH).
Experiment with optimization-based portfolio construction (risk parity, volatility targeting).

If you find this repository useful or would like to discuss quant internships, feel free to reach out.


