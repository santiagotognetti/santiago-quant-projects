import pandas as pd
import yfinance as yf



def get_tickers(filepath: str) -> list[str]:
    """
    Scrape tickers.
    Uses a browser user-agent to avoid blocks.
    Note: survivorship bias applies — current constituents only.
    """
    EXCHANGE_SUFFIX = {
        "Xetra": ".DE",
        "Deutsche Boerse Xetra": ".DE",
        "London Stock Exchange": ".L",
        "Nyse Euronext - Euronext Paris": ".PA",
        "Euronext Amsterdam": ".AS",
        "SIX Swiss Exchange": ".SW",
        "Borsa Italiana": ".MI",
        "Nasdaq Omx Helsinki Ltd.": ".HE",
        "Omx Nordic Exchange Copenhagen A/S": ".CO",
        "Nasdaq Omx Nordic": ".ST",
        "Oslo Bors Asa": ".OL",
        "Bolsa De Madrid": ".MC",
        "Nyse Euronext - Euronext Brussels": ".BR",
        "Nyse Euronext - Euronext Lisbon": ".LS",
        "Irish Stock Exchange - All Market": ".IR",
        "Wiener Boerse Ag": ".VI",
        "New York Stock Exchange Inc.": "",
        "Eurex Deutschland": "",
    }
    df = pd.read_csv(filepath,skiprows=2)

    df = df[df["Asset Class"] == "Equity"].copy()

    tickers = []
    skipped = []

    for _, row in df.iterrows():
        raw_ticker = str (row["Ticker"]).strip()
        exchange = str(row["Exchange"]).strip()

        clean_ticker = raw_ticker.replace(" ", "-")

        suffix = EXCHANGE_SUFFIX.get(exchange)

        if suffix is None:
            skipped.append(f"{raw_ticker} ({exchange})")
            continue

        tickers.append(clean_ticker + suffix)

    if skipped:
        print(f"Warning: {len(skipped)} tickers skipped (unkown exchange):")

        for t in skipped:
            print(f"  {t}")

    print(f"Loaded {len(tickers)} equity tickers from {filepath}")
    return tickers

def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.
    Drops any stock with more than 10% missing data.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    prices = raw.dropna(axis=1, thresh=int(0.9 * len(raw)))
    print(f"Loaded {prices.shape[1]} stocks, {prices.shape[0]} trading days.")
    return prices

def benchmark_long_only_equal_weight(prices: pd.DataFrame):
    """
       Long-only equal-weight benchmark over the same universe and date range
       as the momentum strategy. Rebalances monthly to maintain equal weights.
       No transaction costs applied (benchmark assumed frictionless).

       Returns daily portfolio returns as a pd.Series.
       """
    rets = prices.pct_change().fillna(0)
    n = prices.shape[1]
    weight = 1.0 / n
    return rets.mul(weight).sum(axis=1)
