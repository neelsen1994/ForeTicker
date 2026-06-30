"""
Earnings dates and macro event flags.

Leakage rule (see DEVELOPMENT_GUIDE.md): the *date* of an earnings
announcement is known in advance and safe to flag. The *EPS surprise*
(reported vs estimated) is only known once the report is released —
using it on the announcement day itself would leak the future outcome,
so it is shifted to become visible starting the next trading day.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yfinance as yf

from config import META_DIR

EARNINGS_CACHE_DIR = META_DIR / "earnings"


def fetch_earnings_calendar(ticker: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch earnings dates + EPS estimate/actual/surprise from yfinance.
    Cached locally since yfinance only returns a limited recent history per call.
    """
    EARNINGS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EARNINGS_CACHE_DIR / f"{ticker.replace('.', '_')}_earnings.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    t = yf.Ticker(ticker)
    ed = t.earnings_dates
    if ed is None or ed.empty:
        print(f"[events] {ticker}: no earnings dates available")
        return pd.DataFrame(columns=["date", "eps_estimate", "eps_reported", "eps_surprise_pct"])

    ed = ed.reset_index().rename(columns={
        "Earnings Date": "date",
        "EPS Estimate": "eps_estimate",
        "Reported EPS": "eps_reported",
        "Surprise(%)": "eps_surprise_pct",
    })
    ed["date"] = pd.to_datetime(ed["date"]).dt.tz_localize(None).dt.normalize()
    ed = ed.sort_values("date").reset_index(drop=True)

    ed.to_parquet(cache_path, index=False)
    return ed


def add_event_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge earnings calendar onto a price-indexed DataFrame (DatetimeIndex).
    Adds:
      - is_earnings_day: 1 on the scheduled announcement date (known in advance, safe)
      - eps_surprise_pct: reported vs estimated surprise, visible only from the
        NEXT trading day onward (avoids leaking same-day outcome)
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    earnings = fetch_earnings_calendar(ticker)
    df["is_earnings_day"] = 0
    df["eps_surprise_pct"] = 0.0

    if earnings.empty:
        return df

    for _, row in earnings.iterrows():
        ann_date = row["date"]
        if ann_date in df.index:
            df.loc[ann_date, "is_earnings_day"] = 1

        surprise = row.get("eps_surprise_pct")
        if pd.isna(surprise):
            continue

        # Surprise becomes known the next trading day after announcement
        future_dates = df.index[df.index > ann_date]
        if len(future_dates) > 0:
            next_day = future_dates[0]
            df.loc[next_day, "eps_surprise_pct"] = surprise

    return df


if __name__ == "__main__":
    from config import RAW_PRICES_DIR, DEFAULT_TICKERS

    for ticker in DEFAULT_TICKERS:
        path = RAW_PRICES_DIR / f"{ticker.replace('.', '_')}.parquet"
        if not path.exists():
            print(f"[events] {ticker}: no price data, skip")
            continue
        df = pd.read_parquet(path)
        out = add_event_features(df, ticker)
        n_earnings = out["is_earnings_day"].sum()
        print(f"[events] {ticker}: {n_earnings} earnings days flagged")
