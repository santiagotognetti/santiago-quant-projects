# Quant Project Repo

This repository contains two demo projects intended to show how to convert econometric ideas into reproducible trading research:

- Project1: Intraday microstructure mean-reversion (minute data).
- Project2: Cross-sectional momentum (daily data).

Each script contains a synthetic-data demo section so code runs immediately. Replace the synthetic generators with real market data (yfinance / exchange CSVs / provider API) for actual research.

See README for methodology, results, and robustness checks to add:
- transaction-cost modeling
- slippage and capacity assumptions
- walk-forward and out-of-sample testing
- Monte-Carlo resampling and multiple-testing adjustments
