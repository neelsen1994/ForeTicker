"""
Merge price + technical indicators + event flags + daily sentiment
into a single feature matrix per ticker.

Output: data/processed/features/{TICKER}_features.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import RAW_PRICES_DIR, PROCESSED_SENTIMENTS_DIR, PROCESSED_FEATURES_DIR
from features.technical import add_technical_features
from features.events import add_event_features

SENTIMENT_COLS = ["sentiment_mean", "sentiment_std", "article_count", "bullish_ratio", "bearish_ratio"]


def build_feature_matrix(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge technical-indicator price data with daily sentiment and event flags.
    Both price_df and sentiment_df are indexed/keyed by date.
    """
    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)

    # Technical indicators (drops the warm-up NaN rows internally)
    merged = add_technical_features(price_df)

    # Event flags (earnings days, EPS surprise — leakage-safe)
    merged = add_event_features(merged, ticker)

    # Sentiment — left join, forward-fill short gaps (market open, no news that day)
    sentiment_df = sentiment_df.copy()
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    sentiment_df = sentiment_df.set_index("date")

    merged = merged.join(sentiment_df[SENTIMENT_COLS], how="left")
    merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].ffill(limit=3).fillna(0)

    return merged


def build_ticker(ticker: str) -> pd.DataFrame:
    price_path = RAW_PRICES_DIR / f"{ticker.replace('.', '_')}.parquet"
    sentiment_path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_daily_sentiment.parquet"

    if not price_path.exists():
        raise FileNotFoundError(f"No price data for {ticker} — run price_fetcher.py first")
    if not sentiment_path.exists():
        raise FileNotFoundError(f"No sentiment data for {ticker} — run sentiment_aggregator.py first")

    price_df = pd.read_parquet(price_path)
    sentiment_df = pd.read_parquet(sentiment_path)

    features = build_feature_matrix(price_df, sentiment_df, ticker)

    PROCESSED_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    features.to_parquet(out_path)
    print(f"[builder] {ticker}: {len(features)} rows, {features.shape[1]} columns → {out_path}")
    return features


def build_all(tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = build_ticker(ticker)
        except Exception as e:
            print(f"[builder] ERROR {ticker}: {e}")
    return results


def validate_features(ticker: str) -> bool:
    """No-NaN-gap check per the Phase 3 goal."""
    path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    if not path.exists():
        print(f"[WARN] {ticker}: feature matrix missing")
        return False

    df = pd.read_parquet(path)
    nan_counts = df.isna().sum()
    bad_cols = nan_counts[nan_counts > 0]

    if not bad_cols.empty:
        print(f"[WARN] {ticker}: NaN gaps found —\n{bad_cols}")
        return False

    print(f"[OK] {ticker}: {len(df)} rows, no NaN gaps")
    return True


if __name__ == "__main__":
    from config import DEFAULT_TICKERS

    build_all(DEFAULT_TICKERS)
    for ticker in DEFAULT_TICKERS:
        validate_features(ticker)
