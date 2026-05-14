import numpy as np
import os
import time
import requests
import pandas as pd
from pathlib import Path

POLYGON_BASE = "https://api.polygon.io/v2/aggs/ticker"

def fetch_polygon_minute_bars(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    cache_dir: Path = Path("Data"),
    multiplier: int = 1,
    timespan: str = "minute",
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV bars from Polygon.io and cache to parquet.

    Free tier rate limit: 5 requests/minute — pagination is handled
    automatically with a 12-second sleep between pages.
    Cached files are reused on subsequent runs.

    :param ticker: e.g. 'SPY', 'AAPL'
    :param start: 'YYYY-MM-DD'
    :param end: 'YYYY-MM-DD'
    :param api_key: Polygon.io API key
    :param cache_dir: directory for parquet cache files
    :param multiplier: bar size multiplier (1 = 1-minute bars)
    :param timespan: 'minute', 'hour', or 'day'
    :return: DataFrame with DatetimeIndex (America/New_York) and columns
             [open, high, low, close, volume, vwap]
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{ticker}_{start}_{end}_{multiplier}{timespan}.parquet"

    if cache_file.exists():
        print(f"Loading cached data: {cache_file}")
        return pd.read_parquet(cache_file)

    key = api_key or os.environ.get("POLYGON_API_KEY")
    if not key:
        raise ValueError(
            "Polygon API key required. Pass api_key= or set POLYGON_API_KEY env variable."
        )

    url = f"{POLYGON_BASE}/{ticker}/range/{multiplier}/{timespan}/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}

    all_results = []
    page = 0

    while url:
        page += 1
        print(f"  Fetching page {page}: {ticker} {start} → {end} ...")
        resp = requests.get(url, params=params if page == 1 else {"apiKey": key})
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break
        all_results.extend(results)

        # Polygon pagination: next_url provided if more pages exist
        url = data.get("next_url", None)
        if url:
            time.sleep(12)   # free tier: 5 req/min → 12s between pages

    if not all_results:
        raise ValueError(f"No data returned for {ticker} between {start} and {end}.")

    df = pd.DataFrame(all_results)
    df["timestamp"] = (
        pd.to_datetime(df["t"], unit="ms", utc=True)
        .dt.tz_convert("America/New_York")
    )
    df = df.set_index("timestamp").sort_index()
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap"
    })
    df = df[["open", "high", "low", "close", "volume", "vwap"]]

    df.to_parquet(cache_file)
    print(f"Saved {len(df)} bars → {cache_file}")
    return df

def prepare_bars(
    df: pd.DataFrame,
    session_start: str = "09:30",
    session_end: str = "15:59",
) -> pd.DataFrame:
    """
    Clean raw Polygon bars for backtesting:
    - Restrict to regular session (09:30–15:59 ET); excludes the 16:00 close bar to avoid corner cases
    - Drop zero-volume bars (halts, auction prints)
    - Resample to 1-min grid; forward-fill max 2 consecutive missing bars
    - Compute log returns
    - Compute bar-level order-flow imbalance: (close - open) / (high - low)
    - Compute spread proxy: (high - low) / close

    :param df: raw DataFrame from fetch_polygon_minute_bars
    :param session_start: inclusive session open
    :param session_end: inclusive session close
    :return: cleaned DataFrame with [open, high, low, close, volume, vwap,
             ret, imbalance, spread]
    """
    # Regular session only
    df = df.between_time(session_start, session_end).copy()

    # Drop zero-volume bars (halts, auction prints)
    df = df[df["volume"] > 0].copy()

    # Fill gaps up to 2 consecutive missing bars; drop anything longer
    df = df.resample("min").last()
    df = df.ffill(limit=2)
    df = df.dropna(subset=["close"])

    # Log returns
    df["ret"] = np.log(df["close"] / df["close"].shift(1)).fillna(0)

    # Bar-level order-flow imbalance proxy: (close - open) / (high - low)
    bar_range       = (df["high"] - df["low"]).replace(0, np.nan)
    df["imbalance"] = ((df["close"] - df["open"]) / bar_range).fillna(0)

    # Spread proxy: (high - low) / close
    df["spread"] = (df["high"] - df["low"]) / df["close"]


    return df[["open", "high", "low", "close", "volume", "vwap",
               "ret", "imbalance", "spread"]]

