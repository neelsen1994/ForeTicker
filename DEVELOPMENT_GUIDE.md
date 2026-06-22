# ForeTicker — Development Guide

Remote server setup: GPU available, Ollama running in Docker.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                              │
│  Alpha Vantage / yfinance         NewsAPI / RSS feeds               │
│  (OHLCV + fundamentals)           (raw article URLs)                │
└────────────────┬────────────────────────┬───────────────────────────┘
                 │                        │
                 ▼                        ▼
┌───────────────────────┐   ┌─────────────────────────────────────────┐
│   TIME-SERIES STORE   │   │           NLP PIPELINE                  │
│   (Parquet / SQLite)  │   │  Scraper → FinBERT OR Ollama LLM        │
│   OHLCV, indicators,  │   │  → structured JSON per article:         │
│   macro (VIX, rates)  │   │    {sentiment, confidence, entities,    │
│   earnings calendar   │   │     key_factors, impact_horizon}        │
└───────────┬───────────┘   └───────────────────┬─────────────────────┘
            │                                   │
            └──────────────────┬────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│  Technical indicators (TA-Lib) + Sentiment time-series +            │
│  Event flags (earnings, macro announcements) + Rolling aggregations │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FORECASTING MODEL                               │
│  Temporal Fusion Transformer (TFT) via PyTorch Forecasting          │
│  Inputs: past OHLCV + indicators + sentiment scores                 │
│  Output: predicted return direction + quantile intervals            │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       BACKTESTER                                    │
│  Walk-forward validation → Sharpe ratio, max drawdown, win rate     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack Decisions

| Component | Choice | Why |
|---|---|---|
| Price data | `yfinance` (primary), Alpha Vantage (backup) | yfinance is free, no key needed, covers global tickers |
| News collection | `NewsAPI` + `feedparser` (RSS) | NewsAPI for historical, RSS for real-time |
| Article scraping | `trafilatura` | More robust than newspaper3k, better boilerplate removal |
| Sentiment / NLP | `FinBERT` (HuggingFace) for fast batch scoring; Ollama LLM for deep structured extraction | FinBERT = domain-tuned BERT, fast GPU inference; Ollama = richer output when needed |
| Time-series model | Temporal Fusion Transformer (`pytorch-forecasting`) | Built-in support for static + time-varying covariates, interpretable attention |
| Technical indicators | `pandas-ta` | Pure Python, no TA-Lib C build headache on Linux |
| Data storage | Parquet files + SQLite (metadata) | Parquet = columnar, fast reads; SQLite = zero infra |
| Experiment tracking | `MLflow` (local) | Track hyperparams, metrics, model versions |
| Backtesting | Custom walk-forward loop (simple) | Avoids look-ahead bugs in off-the-shelf backtesting libs |
| API / serving | `FastAPI` | Async, easy to add websocket for live data later |

---

## Environment Setup

### 1. Python Environment

```bash
# On the remote server
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install \
    yfinance \
    newsapi-python \
    feedparser \
    trafilatura \
    transformers \
    torch torchvision \
    pytorch-forecasting \
    pytorch-lightning \
    pandas-ta \
    pandas \
    pyarrow \
    scikit-learn \
    mlflow \
    fastapi \
    uvicorn \
    python-dotenv \
    requests \
    tqdm \
    schedule
```

> If you want FinBERT on GPU, make sure `torch` installs with CUDA support:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Ollama Model Setup

```bash
# Pull a good reasoning model for structured extraction
ollama pull llama3.1:8b          # good balance of speed/quality
ollama pull mistral:7b-instruct  # alternative, fast

# For finance-specific tasks (if available on your Ollama version)
ollama pull command-r:35b        # better for RAG/structured output, needs more VRAM
```

Test Ollama is reachable:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "What is the sentiment of: Apple beats earnings expectations by 15%?",
  "stream": false
}'
```

### 3. Project `.env`

```env
NEWS_API_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alphavantage_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
FINBERT_MODEL=ProsusAI/finbert
DATA_DIR=./data
MLFLOW_TRACKING_URI=./mlruns
```

---

## Project Structure

```
ForeTicker/
├── .env
├── config.py                    # Load env, shared constants
├── DEVELOPMENT_GUIDE.md
│
├── data/
│   ├── raw/
│   │   ├── prices/              # {TICKER}.parquet
│   │   └── articles/            # {date}_{source}_{hash}.txt
│   ├── processed/
│   │   ├── features/            # {TICKER}_features.parquet
│   │   └── sentiments/          # {TICKER}_sentiments.parquet
│   └── meta/
│       └── articles_meta.json   # URL, date, ticker, scrape status
│
├── ingestion/
│   ├── price_fetcher.py         # yfinance wrapper
│   ├── news_collector.py        # NewsAPI + RSS
│   └── article_scraper.py       # trafilatura-based scraper
│
├── nlp/
│   ├── finbert_scorer.py        # Batch FinBERT sentiment scoring
│   ├── ollama_extractor.py      # Deep structured extraction via Ollama
│   └── sentiment_aggregator.py  # Roll article-level → day-level scores
│
├── features/
│   ├── technical.py             # pandas-ta indicators
│   ├── events.py                # Earnings dates, macro event flags
│   └── builder.py               # Merge price + sentiment + events → feature matrix
│
├── models/
│   ├── tft_model.py             # TFT definition and training loop
│   ├── train.py                 # Walk-forward training entry point
│   └── evaluate.py              # Sharpe, drawdown, win rate
│
├── backtest/
│   └── walkforward.py           # Walk-forward validation logic
│
├── api/
│   └── main.py                  # FastAPI serving endpoint
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_sentiment_quality_check.ipynb
    └── 03_model_baseline.ipynb
```

---

## Phase-by-Phase Development Plan

### Phase 1 — Data Pipeline (Week 1)

**Goal**: Reliable, reproducible data for at least 3 tickers over 2+ years.

#### `ingestion/price_fetcher.py`
```python
import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_prices(ticker: str, start: str, end: str, data_dir: str = "./data/raw/prices") -> pd.DataFrame:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(data_dir) / f"{ticker.replace('.', '_')}.parquet"

    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.index = pd.to_datetime(df.index)
    df.to_parquet(out_path)
    print(f"Saved {len(df)} rows for {ticker} → {out_path}")
    return df
```

#### `ingestion/article_scraper.py`
```python
import trafilatura
import hashlib
import json
from pathlib import Path
from datetime import datetime

def scrape_url(url: str, ticker: str, article_date: str, data_dir: str = "./data") -> dict | None:
    raw_dir = Path(data_dir) / "raw" / "articles"
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text or len(text) < 100:
        return None

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    fname = f"{article_date}_{ticker}_{url_hash}.txt"
    (raw_dir / fname).write_text(text, encoding="utf-8")

    meta = {
        "url": url, "ticker": ticker, "date": article_date,
        "file": fname, "char_count": len(text)
    }
    return meta
```

**Validation checklist before moving to Phase 2:**
- [ ] At least 500 trading days of OHLCV per ticker, no gaps
- [ ] At least 200 articles per ticker with clean extracted text
- [ ] Article dates correctly aligned to trading calendar (no weekend articles assigned to wrong day)

---

### Phase 2 — NLP Sentiment Pipeline (Week 2)

**Two-track approach**: FinBERT for speed (batch all articles), Ollama for depth (sample or high-impact articles).

#### `nlp/finbert_scorer.py`
```python
from transformers import pipeline
import pandas as pd
from pathlib import Path
import torch

class FinBERTScorer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-classification",
            model=model_name,
            device=device,
            top_k=None  # return all 3 scores (positive, negative, neutral)
        )

    def score_text(self, text: str) -> dict:
        # FinBERT has 512 token limit — chunk long articles
        chunks = [text[i:i+2000] for i in range(0, min(len(text), 8000), 2000)]
        results = self.pipe(chunks, truncation=True, max_length=512)

        # Average scores across chunks
        agg = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for chunk_result in results:
            for item in chunk_result:
                agg[item["label"].lower()] += item["score"]

        n = len(results)
        return {k: v / n for k, v in agg.items()}

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        scores = [self.score_text(t) for t in df[text_col]]
        scores_df = pd.DataFrame(scores)
        return pd.concat([df, scores_df], axis=1)
```

#### `nlp/ollama_extractor.py`
```python
import requests
import json
import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

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

def extract_structured(article_text: str) -> dict | None:
    prompt = EXTRACTION_PROMPT.format(article_text=article_text[:3000])

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        raw = resp.json()["response"].strip()

        # Strip markdown fences if model adds them
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw)
    except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
        print(f"Extraction failed: {e}")
        return None
```

#### `nlp/sentiment_aggregator.py`
```python
import pandas as pd

def aggregate_daily_sentiment(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    articles_df: columns = [date, ticker, positive, negative, neutral]
    Returns daily aggregated sentiment per ticker.
    """
    articles_df["date"] = pd.to_datetime(articles_df["date"])
    articles_df["net_sentiment"] = articles_df["positive"] - articles_df["negative"]

    daily = articles_df.groupby(["date", "ticker"]).agg(
        sentiment_mean=("net_sentiment", "mean"),
        sentiment_std=("net_sentiment", "std"),
        article_count=("net_sentiment", "count"),
        bullish_ratio=("positive", "mean"),
        bearish_ratio=("negative", "mean"),
    ).reset_index()

    # Fill missing trading days with 0 (no news = neutral)
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0)
    return daily
```

**Critical rule**: Article publication time matters. A news article published after market close belongs to the **next** trading day's feature set — not the same day. Add this alignment in the aggregator.

---

### Phase 3 — Feature Engineering (Week 3)

#### `features/technical.py`
```python
import pandas as pd
import pandas_ta as ta

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: Open, High, Low, Close, Volume"""
    df = df.copy()

    # Trend
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.macd(append=True)

    # Momentum
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)

    # Volatility
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)

    # Volume
    df.ta.obv(append=True)
    df.ta.vwap(append=True)

    # Target: next-day return (what we're predicting)
    df["return_1d"] = df["Close"].pct_change().shift(-1)
    df["target_direction"] = (df["return_1d"] > 0).astype(int)

    return df.dropna()
```

#### `features/builder.py`
```python
import pandas as pd

def build_feature_matrix(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price+technical features with daily sentiment.
    Both indexed by date.
    """
    price_df.index = pd.to_datetime(price_df.index)
    sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
    sentiment_df = sentiment_df.set_index("date")

    merged = price_df.join(sentiment_df, how="left")

    # Forward-fill sentiment on days with no news (market was open, no articles)
    sentiment_cols = ["sentiment_mean", "sentiment_std", "article_count", "bullish_ratio", "bearish_ratio"]
    merged[sentiment_cols] = merged[sentiment_cols].fillna(method="ffill", limit=3).fillna(0)

    return merged
```

---

### Phase 4 — Temporal Fusion Transformer (Week 4)

#### `models/tft_model.py`
```python
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
import pandas as pd

def create_tft_dataset(df: pd.DataFrame, ticker: str, max_encoder_length: int = 60, max_prediction_length: int = 5):
    df = df.copy()
    df["time_idx"] = range(len(df))
    df["ticker"] = ticker

    time_varying_known = []  # future-known: earnings dates, macro events
    time_varying_unknown = [
        "return_1d", "RSI_14", "MACD_12_26_9",
        "sentiment_mean", "article_count", "bullish_ratio",
        "ATRr_14", "BBP_5_2_0"
    ]

    dataset = TimeSeriesDataSet(
        df[:-max_prediction_length],
        time_idx="time_idx",
        target="return_1d",
        group_ids=["ticker"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=time_varying_unknown,
        time_varying_known_categoricals=time_varying_known,
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    return dataset


def train_tft(dataset: TimeSeriesDataSet, max_epochs: int = 30):
    train_loader = dataset.to_dataloader(train=True, batch_size=64, num_workers=4)

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=1e-3,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        log_interval=10,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=0.1,
    )
    trainer.fit(tft, train_dataloaders=train_loader)
    return tft
```

---

### Phase 5 — Walk-Forward Backtesting (Week 5)

#### `backtest/walkforward.py`
```python
import pandas as pd
import numpy as np
from typing import Callable

def walk_forward_evaluate(
    df: pd.DataFrame,
    train_fn: Callable,
    predict_fn: Callable,
    train_months: int = 18,
    test_months: int = 3,
) -> pd.DataFrame:
    """
    Rolls a training window forward in time.
    Returns a DataFrame of predictions vs actuals.

    NO data from the test window ever touches training — this is the key invariant.
    """
    df = df.sort_index()
    all_preds = []

    dates = df.index
    start = dates[0]
    end = dates[-1]

    current = start + pd.DateOffset(months=train_months)

    while current + pd.DateOffset(months=test_months) <= end:
        train_end = current
        test_end = current + pd.DateOffset(months=test_months)

        train_data = df[df.index < train_end]
        test_data = df[(df.index >= train_end) & (df.index < test_end)]

        model = train_fn(train_data)
        preds = predict_fn(model, test_data)

        test_data = test_data.copy()
        test_data["predicted"] = preds
        all_preds.append(test_data[["return_1d", "predicted"]])

        current += pd.DateOffset(months=test_months)

    return pd.concat(all_preds)


def compute_metrics(results_df: pd.DataFrame, threshold: float = 0.0) -> dict:
    """Signal-based metrics — more meaningful than RMSE for trading."""
    df = results_df.dropna()

    # Direction accuracy
    df["pred_direction"] = (df["predicted"] > threshold).astype(int)
    df["actual_direction"] = (df["return_1d"] > 0).astype(int)
    accuracy = (df["pred_direction"] == df["actual_direction"]).mean()

    # Strategy returns (long when predicted positive, flat otherwise)
    df["strategy_return"] = df["return_1d"] * df["pred_direction"]

    # Sharpe ratio (annualized, assuming daily returns)
    sharpe = (df["strategy_return"].mean() / df["strategy_return"].std()) * np.sqrt(252)

    # Max drawdown
    cumulative = (1 + df["strategy_return"]).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "direction_accuracy": round(accuracy, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_drawdown, 4),
        "total_trades": int(df["pred_direction"].sum()),
        "annualized_return": round(df["strategy_return"].mean() * 252, 4),
    }
```

---

## Data Leakage Checklist

These mistakes will inflate your metrics and make the model useless in production. Check each one.

- [ ] **No future price in features**: `pct_change().shift(-1)` is your target — never use it as an input feature
- [ ] **Sentiment shifted correctly**: Articles published after 4PM ET belong to the *next* trading day
- [ ] **Scaler fit only on train data**: `StandardScaler().fit(train)` then `.transform(test)` — never `fit(all_data)`
- [ ] **Walk-forward, not random split**: Never `train_test_split(shuffle=True)` on time-series
- [ ] **No hyperparameter tuning on test fold**: Tune on validation period inside the train window only
- [ ] **Earnings dates excluded from inputs**: If you use earnings surprise as a feature, it must be the *announced* EPS vs *estimated* EPS at time T, not the actual outcome known at T+1

---

## Suggested Tickers to Start With

| Ticker | Exchange | Why |
|---|---|---|
| `AAPL` | NASDAQ | High news volume, liquid, lots of FinBERT training examples |
| `SAP.DE` | XETRA | You already have price data for this |
| `MSFT` | NASDAQ | Stable, high analyst coverage |
| `NVDA` | NASDAQ | High volatility = more signal to capture |

Start with **AAPL** — it has the most English-language news coverage, which means more training data for your NLP pipeline.

---

## Development Order

```
Week 1  →  price_fetcher.py + news_collector.py + article_scraper.py
           Goal: 2 tickers × 2 years of prices + 500 clean articles each

Week 2  →  finbert_scorer.py + sentiment_aggregator.py
           Goal: All articles scored, daily sentiment timeseries in Parquet

Week 3  →  technical.py + builder.py
           Goal: Single merged feature matrix per ticker, no NaN gaps

Week 4  →  tft_model.py + train.py
           Goal: TFT trains end-to-end without errors on GPU

Week 5  →  walkforward.py + evaluate.py
           Goal: Sharpe ratio and direction accuracy reported per ticker

Week 6  →  ollama_extractor.py (replace/supplement FinBERT)
           Compare metrics: FinBERT-only vs Ollama-only vs ensemble

Week 7+ →  FastAPI serving, add more tickers, tune model
```

---

## Useful Commands (Quick Reference)

```bash
# Fetch price data for a ticker
python -c "from ingestion.price_fetcher import fetch_prices; fetch_prices('AAPL', '2022-01-01', '2024-12-31')"

# Score all articles with FinBERT
python -c "from nlp.finbert_scorer import FinBERTScorer; ..."

# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Check Ollama status
curl http://localhost:11434/api/tags

# Monitor GPU
watch -n 1 nvidia-smi
```

---

## Baseline to Beat

Before training any model, compute the **naive baseline**:
- "Market always goes up" → predict `return > 0` every day
- This gives ~54% accuracy on long-term equity indices

If your model doesn't beat ~58% direction accuracy consistently across walk-forward folds, the signal isn't real. Most academic papers claim 60-65% — that's the realistic ceiling for public data + public news.

---

## Notes on Ollama vs FinBERT

| | FinBERT | Ollama (Llama 3.1 8B) |
|---|---|---|
| Speed | ~200 articles/min on GPU | ~10-20 articles/min |
| Output | 3 probability scores | Rich JSON (factors, horizon, entities) |
| Finance domain | Fine-tuned on finance text | General, prompt-guided |
| Best for | Batch scoring all articles | Deep extraction on important articles |
| Recommended use | Phase 1 signal | Phase 2 enrichment |

**Practical approach**: Score everything with FinBERT first. Then use Ollama on the top-10% most-negative and top-10% most-positive articles to extract structured factors. This gives you both speed and depth.
