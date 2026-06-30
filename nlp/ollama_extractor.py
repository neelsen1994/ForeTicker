"""
Deep structured extraction via Ollama LLM.

Used on high-signal articles (top/bottom sentiment decile from FinBERT)
to extract richer features: impact horizon, key factors, macro event flag.

Saves results to:
  data/processed/sentiments/{TICKER}_ollama.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import requests
import pandas as pd
from tqdm import tqdm

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAW_ARTICLES_DIR,
    PROCESSED_SENTIMENTS_DIR,
    ARTICLES_META_FILE,
)

EXTRACTION_PROMPT = """Analyze this financial news article and return ONLY a JSON object with this exact structure:
{{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": 0.0-1.0,
  "impact_horizon": "intraday" | "short_term" | "long_term",
  "key_factors": ["factor1", "factor2"],
  "affected_tickers": ["TICKER1"],
  "macro_event": true | false
}}

Article:
{article_text}

Return only the JSON. No explanation."""


def _parse_ollama_response(raw: str) -> dict | None:
    raw = raw.strip()
    # Strip markdown code fences if the model adds them
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def extract_structured(article_text: str, retries: int = 2) -> dict | None:
    """Send one article to Ollama and return parsed structured output."""
    prompt = EXTRACTION_PROMPT.format(article_text=article_text[:3000])

    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            result = _parse_ollama_response(raw)
            if result is not None:
                return result
        except (requests.RequestException, KeyError) as e:
            if attempt == retries:
                print(f"[ollama_extractor] Failed after {retries+1} attempts: {e}")
            else:
                time.sleep(2)

    return None


def _load_finbert_scores(ticker: str) -> pd.DataFrame:
    path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_finbert.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run finbert_scorer first for {ticker}")
    return pd.read_parquet(path)


def _read_article_text(url: str) -> str | None:
    """Look up article text from meta file by URL."""
    if not ARTICLES_META_FILE.exists():
        return None
    with open(ARTICLES_META_FILE, encoding="utf-8") as f:
        meta = json.load(f)
    for record in meta:
        if record.get("url") == url and record.get("scrape_status") == "done":
            fname = record.get("file")
            if fname:
                path = RAW_ARTICLES_DIR / fname
                if path.exists():
                    return path.read_text(encoding="utf-8")
    return None


def extract_ticker(
    ticker: str,
    top_pct: float = 0.10,
    delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Run Ollama extraction on the top and bottom `top_pct` articles by FinBERT net sentiment.
    Saves to {ticker}_ollama.parquet and returns the DataFrame.
    """
    scores_df = _load_finbert_scores(ticker)
    scores_df["net_sentiment"] = scores_df["positive"] - scores_df["negative"]

    n = len(scores_df)
    k = max(1, int(n * top_pct))

    top_bull = scores_df.nlargest(k, "net_sentiment")
    top_bear = scores_df.nsmallest(k, "net_sentiment")
    candidates = pd.concat([top_bull, top_bear]).drop_duplicates(subset=["url"])

    print(f"[ollama_extractor] {ticker}: running Ollama on {len(candidates)} articles")

    rows = []
    for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc=f"Ollama {ticker}"):
        text = _read_article_text(row["url"])
        if not text:
            continue

        result = extract_structured(text)
        if result is None:
            continue

        rows.append({
            "url": row["url"],
            "date": row["date"],
            "ticker": ticker,
            "title": row.get("title", ""),
            "finbert_net": row["net_sentiment"],
            "ollama_sentiment": result.get("sentiment"),
            "ollama_confidence": result.get("confidence"),
            "impact_horizon": result.get("impact_horizon"),
            "key_factors": json.dumps(result.get("key_factors", [])),
            "affected_tickers": json.dumps(result.get("affected_tickers", [])),
            "macro_event": result.get("macro_event", False),
        })

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    if not rows:
        print(f"[ollama_extractor] No results extracted for {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    PROCESSED_SENTIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_ollama.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[ollama_extractor] Saved → {out_path}")
    return df


def extract_all_tickers(tickers: list[str], top_pct: float = 0.10) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = extract_ticker(ticker, top_pct=top_pct)
        except Exception as e:
            print(f"[ollama_extractor] ERROR {ticker}: {e}")
    return results


if __name__ == "__main__":
    from config import DEFAULT_TICKERS
    extract_all_tickers(DEFAULT_TICKERS)
