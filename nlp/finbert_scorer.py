"""
Batch FinBERT sentiment scoring for scraped articles.

Reads article text files listed in articles_meta.json,
scores each one, and saves results to:
  data/processed/sentiments/{TICKER}_finbert.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

from config import (
    FINBERT_MODEL,
    RAW_ARTICLES_DIR,
    PROCESSED_SENTIMENTS_DIR,
    ARTICLES_META_FILE,
)


class FinBERTScorer:
    def __init__(self, model_name: str = FINBERT_MODEL):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None,  # return all 3 scores: positive, negative, neutral
        )

    def score_text(self, text: str) -> dict:
        # FinBERT is capped at 512 tokens — chunk long articles into ~2000 char pieces
        chunks = [text[i: i + 2000] for i in range(0, min(len(text), 8000), 2000)]
        results = self.pipe(chunks, truncation=True, max_length=512)

        agg = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for chunk_result in results:
            for item in chunk_result:
                agg[item["label"].lower()] += item["score"]

        n = len(results)
        return {k: v / n for k, v in agg.items()}

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        scores = [self.score_text(t) for t in tqdm(df[text_col], desc="FinBERT scoring")]
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)


def _load_articles_df(ticker: str) -> pd.DataFrame:
    """Load all successfully scraped articles for a ticker from meta + text files."""
    if not ARTICLES_META_FILE.exists():
        raise FileNotFoundError("articles_meta.json not found — run news_collector.py first")

    with open(ARTICLES_META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    rows = []
    for record in meta:
        if record.get("ticker") != ticker:
            continue
        if record.get("scrape_status") != "done":
            continue
        fname = record.get("file")
        if not fname:
            continue

        text_path = RAW_ARTICLES_DIR / fname
        if not text_path.exists():
            continue

        text = text_path.read_text(encoding="utf-8")
        rows.append({
            "url": record.get("url", ""),
            "date": record.get("date", ""),
            "ticker": ticker,
            "source": record.get("source", ""),
            "title": record.get("title", ""),
            "text": text,
        })

    if not rows:
        raise ValueError(f"No scraped articles found for {ticker}")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


def score_ticker(ticker: str, scorer: FinBERTScorer | None = None) -> pd.DataFrame:
    """Score all articles for a ticker and save to parquet."""
    if scorer is None:
        scorer = FinBERTScorer()

    df = _load_articles_df(ticker)
    print(f"[finbert_scorer] Scoring {len(df)} articles for {ticker}")

    scored = scorer.score_dataframe(df)
    scored = scored.drop(columns=["text"])  # don't store full text in parquet

    PROCESSED_SENTIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_finbert.parquet"
    scored.to_parquet(out_path, index=False)
    print(f"[finbert_scorer] Saved → {out_path}")
    return scored


def score_all_tickers(tickers: list[str]) -> dict[str, pd.DataFrame]:
    scorer = FinBERTScorer()  # load model once, reuse across tickers
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = score_ticker(ticker, scorer)
        except Exception as e:
            print(f"[finbert_scorer] ERROR {ticker}: {e}")
    return results


if __name__ == "__main__":
    from config import DEFAULT_TICKERS
    score_all_tickers(DEFAULT_TICKERS)
