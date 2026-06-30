"""
Aggregate article-level FinBERT scores → daily sentiment time-series per ticker.

Output: data/processed/sentiments/{TICKER}_daily_sentiment.parquet
Columns: date, ticker, sentiment_mean, sentiment_std, article_count,
         bullish_ratio, bearish_ratio
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import PROCESSED_SENTIMENTS_DIR


def aggregate_daily_sentiment(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    articles_df: columns = [date, ticker, positive, negative, neutral, ...]
    Returns daily aggregated sentiment per ticker, aligned to trading calendar.
    """
    df = articles_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["net_sentiment"] = df["positive"] - df["negative"]

    daily = (
        df.groupby(["date", "ticker"])
        .agg(
            sentiment_mean=("net_sentiment", "mean"),
            sentiment_std=("net_sentiment", "std"),
            article_count=("net_sentiment", "count"),
            bullish_ratio=("positive", "mean"),
            bearish_ratio=("negative", "mean"),
        )
        .reset_index()
    )

    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    daily = daily.sort_values("date").reset_index(drop=True)
    return daily


def load_finbert_scores(ticker: str) -> pd.DataFrame:
    path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_finbert.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No FinBERT scores for {ticker} — run finbert_scorer.py first")
    return pd.read_parquet(path)


def aggregate_ticker(ticker: str) -> pd.DataFrame:
    """Load FinBERT scores, aggregate to daily, save parquet, return DataFrame."""
    scores_df = load_finbert_scores(ticker)
    daily = aggregate_daily_sentiment(scores_df)

    out_path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_daily_sentiment.parquet"
    daily.to_parquet(out_path, index=False)
    print(f"[sentiment_aggregator] {ticker}: {len(daily)} trading days → {out_path}")
    return daily


def aggregate_all_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = aggregate_ticker(ticker)
        except Exception as e:
            print(f"[sentiment_aggregator] ERROR {ticker}: {e}")
    return results


def validate_sentiment(ticker: str, min_days: int = 30) -> bool:
    """Quick sanity check on the daily sentiment output."""
    path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_daily_sentiment.parquet"
    if not path.exists():
        print(f"[WARN] {ticker}: daily sentiment file missing")
        return False

    df = pd.read_parquet(path)
    n = len(df)
    if n < min_days:
        print(f"[WARN] {ticker}: only {n} sentiment days (need {min_days})")
        return False

    missing = df["sentiment_mean"].isna().sum()
    if missing > 0:
        print(f"[WARN] {ticker}: {missing} NaN sentiment_mean values")

    print(f"[OK] {ticker}: {n} days of sentiment, mean={df['sentiment_mean'].mean():.4f}")
    return True


if __name__ == "__main__":
    from config import DEFAULT_TICKERS
    results = aggregate_all_tickers(DEFAULT_TICKERS)
    for ticker in DEFAULT_TICKERS:
        validate_sentiment(ticker)
