"""
Fetch and cache company fundamentals via yfinance.

Fundamentals change slowly (quarterly), so results are cached to disk
and refreshed only when stale (default: 1 day).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import yfinance as yf

from config import META_DIR

FUNDAMENTALS_CACHE_DIR = META_DIR / "fundamentals"

FUNDAMENTALS_FIELDS = [
    "shortName", "sector", "industry",
    "currentPrice", "marketCap", "beta",
    "trailingPE", "forwardPE", "priceToBook",
    "trailingEps", "forwardEps",
    "dividendYield", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    "returnOnEquity", "profitMargins", "grossMargins", "revenueGrowth",
    "totalRevenue", "freeCashflow",
    "recommendationKey", "targetMeanPrice", "numberOfAnalystOpinions",
]


def fetch_fundamentals(ticker: str, max_age_seconds: int = 86400, use_cache: bool = True) -> dict:
    """Fetch key fundamentals for a ticker, cached for `max_age_seconds`."""
    FUNDAMENTALS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = FUNDAMENTALS_CACHE_DIR / f"{ticker.replace('.', '_')}.json"

    if use_cache and cache_path.exists():
        age = time.time() - cache_path.stat().st_mtime
        if age < max_age_seconds:
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)

    info = yf.Ticker(ticker).info
    data = {field: info.get(field) for field in FUNDAMENTALS_FIELDS}
    data["ticker"] = ticker
    data["fetched_at"] = time.time()

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    from config import DEFAULT_TICKERS

    for ticker in DEFAULT_TICKERS:
        data = fetch_fundamentals(ticker, use_cache=False)
        print(f"[fundamentals] {ticker}: {data.get('shortName')} — "
              f"P/E {data.get('trailingPE')}, mktcap {data.get('marketCap')}")
