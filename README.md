# Santiago Tognetti – Quant Research & Trading Projects

Welcome to my repository of quantitative finance research projects.  
These projects are designed to demonstrate skills relevant to **quantitative trading**, **market microstructure**, **statistical arbitrage**, and **systematic strategy development**.

I am currently completing the **Pre-Master in Econometrics & Quantitative Finance** at **Erasmus University Rotterdam**, with a background in Economics (UBA) and professional experience in data analytics and financial modeling.

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

A systematic long–short portfolio based on 60-day momentum, rebalanced monthly, on a universe of equities.

Includes:

- Synthetic-data demo (replaceable with real data)
- Factor computation (momentum)
- Cross-sectional ranking and portfolio building
- Long top-k / short bottom-k construction
- Turnover, transaction cost, and rebalance logic
- Performance analytics (Sharpe, drawdown)
- Visual results and diagnostics

**Core Idea:**  
Momentum persists cross-sectionally across equities.  
This project simulates a market-neutral long-short momentum strategy and evaluates performance under realistic frictions.

---

## 🧱 Repository Structure

santiago-quant-projects/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── project1_intraday/
│ ├── project1_intraday.py
│ ├── notebooks/
│ │ ├── 01_intraday_features.ipynb
│ │ └── 02_intraday_backtest.ipynb
│ └── results/
│ ├── figures/
│ └── summary_project1.pdf
│
├── project2_momentum/
│ ├── project2_momentum.py
│ ├── notebooks/
│ │ ├── 01_momentum_features.ipynb
│ │ └── 02_backtest_momentum.ipynb
│ └── results/
│ ├── figures/
│ └── summary_project2.pdf
│
└── data/
├── raw/
└── processed/

## Installation

Clone the repository:

```bash
git clone https://github.com/gomerfield/santiago-quant-projects.git
cd santiago-quant-projects

pip install -r requirements.txt

**## How to run the projects**

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
Add volatility-adjusted momentum factors.
Build intraday volatility forecasting models (HAR, GARCH).
Experiment with optimization-based portfolio construction (risk parity, volatility targeting).

If you find this repository useful or would like to discuss quant internships, feel free to reach out.


