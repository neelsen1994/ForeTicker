# ForeTicker — Complete Project Documentation

This document is the full record of what was built, why, and what's left. It's written so you (or a future session) can pick this project back up without re-deriving any of the decisions made along the way. It assumes you invest and read stock charts, but aren't deep into the technical/quant side — every indicator and metric gets a plain-English explanation, not just a formula.

**Companion file:** [DEVELOPMENT_GUIDE.md](DEVELOPMENT_GUIDE.md) is the *original plan* written before any code existed. This document is the *as-built* record — where reality matched the plan, where it diverged, and why.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Architecture](#2-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Development Log — Phase by Phase](#4-development-log--phase-by-phase)
5. [Current Data & Model Status](#5-current-data--model-status)
6. [How to Run Everything](#6-how-to-run-everything)
7. [Known Issues & Limitations](#7-known-issues--limitations)
8. [Stock Market & Technical Analysis Glossary](#8-stock-market--technical-analysis-glossary)
9. [Future Roadmap](#9-future-roadmap)

---

## 1. What This Project Does

ForeTicker is a personal stock analysis and forecasting system for 4 tickers (AAPL, MSFT, NVDA, SAP.DE). It:

- Pulls historical + live price data and financial news
- Scores news sentiment with FinBERT (fast) and optionally extracts richer structure with a local LLM via Ollama (slow but deeper)
- Builds a technical + fundamental + sentiment feature matrix per ticker
- Trains a Temporal Fusion Transformer (TFT) to forecast next-day returns
- Backtests that model honestly (walk-forward, no look-ahead bias)
- Serves everything through a Streamlit dashboard and a FastAPI backend
- Runs a real-time-ish alert engine that watches for sentiment spikes, earnings surprises, and volume/price anomalies — **not** based on the TFT, because the TFT doesn't yet beat a naive baseline (see [Section 7](#7-known-issues--limitations))

**Your stated top priority** was the alert engine — specifically because you've personally lost money by not getting news fast enough. That shaped a key decision: alerts are built on directly interpretable signals (sentiment/volume/price/earnings), not on the model, because a confidently-wrong prediction is worse than no prediction for exactly the failure mode you described.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                              │
│  yfinance (prices)      Alpha Vantage / GDELT / NewsAPI / RSS (news)│
└────────────────┬────────────────────────┬───────────────────────────┘
                 │                        │
                 ▼                        ▼
┌───────────────────────┐   ┌─────────────────────────────────────────┐
│   TIME-SERIES STORE    │   │           NLP PIPELINE                  │
│   Parquet files         │   │  Scraper → FinBERT (fast, all articles) │
│   OHLCV, earnings cal.  │   │           → Ollama (slow, top/bottom 10%)│
└───────────┬───────────┘   └───────────────────┬─────────────────────┘
            │                                   │
            └──────────────────┬────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                            │
│  Technical indicators (`ta` lib) + Daily sentiment + Earnings flags  │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                 ┌───────────────┴────────────────┐
                 ▼                                 ▼
┌───────────────────────────────┐   ┌──────────────────────────────────┐
│   FORECASTING (TFT)            │   │   ALERT ENGINE                    │
│   Walk-forward backtested       │   │   Sentiment/volume/price/earnings │
│   Doesn't beat naive baseline   │   │   rules — NOT model-based         │
└───────────────┬────────────────┘   └────────────────┬───────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  Streamlit Dashboard + FastAPI  │
                    └───────────────────────────────┘
```

---

## 3. Repository Structure

```
ForeTicker/
├── config.py                    # All shared constants, thresholds, paths, API keys
├── .env                         # Your actual secrets (gitignored)
├── .env.example                 # Template — safe to commit
├── requirements.txt              # All dependencies
├── DEVELOPMENT_GUIDE.md          # Original plan (Phases 1-7, written before any code)
├── DOCUMENTATION.md              # This file — the as-built record
│
├── data/                         # All gitignored — local only
│   ├── raw/prices/               # {TICKER}.parquet — OHLCV from yfinance
│   ├── raw/articles/              # {date}_{ticker}_{hash}.txt — scraped article text
│   ├── processed/features/        # {TICKER}_features.parquet — final model input
│   ├── processed/sentiments/       # FinBERT scores, daily aggregates, Ollama extractions
│   └── meta/
│       ├── articles_meta.json     # URL/date/ticker/scrape-status registry
│       ├── alerts.json            # Alert feed (persistent log)
│       ├── earnings/               # Cached earnings calendars per ticker
│       └── newsapi_last_fetch.json # NewsAPI cooldown state for the live watcher
│
├── ingestion/
│   ├── price_fetcher.py          # yfinance OHLCV fetch + validation
│   ├── news_collector.py         # Alpha Vantage + GDELT + RSS + NewsAPI collectors
│   └── article_scraper.py        # trafilatura-based full-text scraping
│
├── nlp/
│   ├── finbert_scorer.py         # Batch FinBERT sentiment scoring (GPU)
│   ├── sentiment_aggregator.py    # Article-level → daily sentiment time series
│   └── ollama_extractor.py       # Deep structured extraction on high-signal articles
│
├── features/
│   ├── technical.py              # EMA/MACD/RSI/Stochastic/Bollinger/ATR/OBV/VWAP
│   ├── events.py                 # Earnings calendar + leakage-safe event flags
│   ├── fundamentals.py           # yfinance fundamentals (P/E, market cap, etc.), cached
│   └── builder.py                # Merges price + technical + events + sentiment
│
├── models/
│   ├── tft_model.py               # TFT dataset construction + training loop
│   ├── train.py                   # Single train/val split entry point
│   ├── evaluate.py                # Sharpe/drawdown/accuracy metrics + predict_window()
│   ├── predict.py                 # Forward-looking inference (real future forecast)
│   └── checkpoints/                # Saved .ckpt + dataset_params.pkl per ticker
│
├── backtest/
│   └── walkforward.py            # Rolling walk-forward evaluation (the real accuracy test)
│
├── alerts/
│   ├── rules.py                  # Sentiment/volume/price/earnings trigger functions
│   └── watcher.py                # Live polling loop + alert feed persistence
│
├── dashboard/
│   └── app.py                    # Streamlit UI — charts, fundamentals, forecast, alerts
│
└── api/
    └── main.py                   # FastAPI serving layer (8 endpoints)
```

---

## 4. Development Log — Phase by Phase

### Phase 1 — Data Pipeline

**Goal (from the guide):** reliable, reproducible price + news data for 3+ tickers over 2+ years.

**What was built:**
- [ingestion/price_fetcher.py](ingestion/price_fetcher.py) — wraps `yfinance`, validates trading-day counts and flags date gaps
- [ingestion/news_collector.py](ingestion/news_collector.py) — **four** collectors, more than the original guide called for:
  - `collect_alphavantage()` — historical news via Alpha Vantage's `NEWS_SENTIMENT` endpoint, chunked into yearly windows
  - `collect_gdelt()` — free, no-key historical news via GDELT's Doc API, chunked into 2-week windows with exponential backoff on rate limits
  - `collect_rss()` — Yahoo Finance RSS feeds, real-time-ish, no rate limit
  - `collect_newsapi()` — NewsAPI, but its free tier only allows the last 30 days, so it's more useful for live polling than historical backfill
- [ingestion/article_scraper.py](ingestion/article_scraper.py) — `trafilatura`-based full-text extraction, batches of 50 with progress checkpointing so a crash doesn't lose work

**Why four news sources?** The original guide only planned NewsAPI + RSS. NewsAPI's free tier turned out to only permit articles from the last 30 days — useless for building 2+ years of historical training data. GDELT was added as a free historical alternative, then Alpha Vantage was added when GDELT's rate limiting became a problem (see below). This ended up being the right call: Alpha Vantage became the primary historical source, RSS/NewsAPI became the live-polling sources, and GDELT is currently unusable (blocked — see [Section 7](#7-known-issues--limitations)).

**Key technical decision — trading-day alignment:** Articles published after 4 PM ET are assigned to the *next* trading day, not the day they were published. This matters because a stock's reaction to after-hours news shows up in the next session's price, not the same day's. Getting this wrong is one of the most common sources of "the model looks great in backtest, useless in production" — it silently leaks the market's *reaction* into the *same day* as the news, which the model then learns as if the news predicted same-day movement.

**Issues encountered & fixed:**
- Alpha Vantage's free key caps at 25 requests/day. The collector fetches one request per ticker per calendar year, so 4 tickers × 5 years = 20 requests — should fit, but partial runs (debugging, retries) ate into the quota, leaving a **data gap: no articles for all of 2025 and early-to-mid 2026** for most tickers. See [Section 7](#7-known-issues--limitations) for the current status.
- GDELT got rate-limited/blocked during heavy interactive testing early on, and **remains blocked** even with conservative 10-second pacing — this looks like a longer-duration IP-level throttle, not the rolling window GDELT's docs describe.
- `pandas-ta` (the guide's originally planned indicator library) is unmaintained and doesn't build on Python 3.11+. Switched to the `ta` library in Phase 3 — see below.

---

### Phase 2 — NLP Sentiment Pipeline

**Goal:** FinBERT for fast batch scoring of all articles, Ollama for deep structured extraction on the highest-signal articles.

**What was built:**
- [nlp/finbert_scorer.py](nlp/finbert_scorer.py) — `ProsusAI/finbert` via HuggingFace `transformers`, chunks articles into 2000-character pieces (FinBERT's hard limit is 512 tokens) and averages scores across chunks. Runs on GPU — **4,465 articles scored in under a minute** on the server's L40S GPU.
- [nlp/sentiment_aggregator.py](nlp/sentiment_aggregator.py) — rolls article-level FinBERT scores into daily `sentiment_mean`, `sentiment_std`, `article_count`, `bullish_ratio`, `bearish_ratio` per ticker, respecting the trading-day alignment rule from Phase 1.
- [nlp/ollama_extractor.py](nlp/ollama_extractor.py) — runs only on the top/bottom 10% of articles by FinBERT net sentiment (280 articles for AAPL, for example, not all 1400+). For each, prompts a local Ollama model (`llama3.1:8b`) to extract: `sentiment` (bullish/bearish/neutral), `confidence`, `impact_horizon` (intraday/short_term/long_term), `key_factors`, `macro_event` flag.

**Why only 10% for Ollama?** FinBERT processes ~70 articles/second on GPU; Ollama took 12-25 seconds *per article* (it's a much heavier LLM call, not a lightweight classifier). Running it on everything would have taken days. The 10% sample deliberately targets the most extreme sentiment articles — the ones most likely to matter for both model training and (later) for alerts.

**Result:** All 4 tickers' Ollama extraction completed (some individual requests timed out and were skipped gracefully — the pipeline doesn't crash on a single failed extraction). Output saved to `data/processed/sentiments/{TICKER}_ollama.parquet` with fields like `impact_horizon` and `macro_event` that are genuinely useful signals but **not yet wired into anything** — a good candidate for the alert engine or as additional model features later.

---

### Phase 3 — Feature Engineering

**Goal:** merge technical indicators + sentiment + event flags into one clean feature matrix per ticker, no NaN gaps.

**What was built:**
- [features/technical.py](features/technical.py) — trend (EMA 20/50, MACD), momentum (RSI 14, Stochastic Oscillator), volatility (Bollinger Bands, ATR), volume (OBV, VWAP). See [Section 8](#8-stock-market--technical-analysis-glossary) for what each of these means.
- [features/events.py](features/events.py) — earnings calendar via `yfinance`, with a leakage-safe design: `is_earnings_day` (the announcement *date* is known in advance, safe to use) vs `eps_surprise_pct` (the *actual reported* surprise is only known after the report — shifted to appear starting the *next* trading day, never the announcement day itself).
- [features/fundamentals.py](features/fundamentals.py) — market cap, P/E, beta, dividend yield, 52-week range, analyst consensus, etc., cached for 24 hours since these change slowly.
- [features/builder.py](features/builder.py) — merges everything, forward-fills sentiment gaps (max 3 days — a stock with no news for a few days keeps its last-known sentiment; longer gaps reset to neutral).

**Why switch from `pandas-ta` to `ta`?** `pandas-ta`'s PyPI release only supports Python <3.11 and its GitHub repo is largely abandoned. `ta` is actively maintained and has feature parity for everything the guide needed. The API is slightly more verbose (explicit indicator classes instead of chained `.ta.` accessors) but otherwise equivalent.

**Data leakage discipline (the guide's own checklist, honored throughout):**
- `return_1d` (the model's target) is `pct_change().shift(-1)` — the *next* day's return — and is never used as an input feature.
- The very last row of each feature matrix is always dropped, because its target (tomorrow's return) is unknown. This matters later: it means the standard feature matrix is always one day "behind" for live use, which the alert engine had to specifically work around (see Phase 6 below).

**Result:** all 4 tickers build cleanly — 28 columns, zero NaN gaps, currently ~1075-1093 rows each (2022 through today).

---

### Phase 4 — Temporal Fusion Transformer (TFT)

**Goal:** train a TFT that ingests OHLCV + technical indicators + sentiment + event flags and outputs a next-day return forecast with quantile intervals.

**What was built:**
- [models/tft_model.py](models/tft_model.py) — `create_tft_dataset()` builds a `pytorch_forecasting.TimeSeriesDataSet`. Inputs are split into:
  - `time_varying_unknown_reals` — features whose *future* values aren't known (return, RSI, MACD, sentiment, etc.) — the model must infer these
  - `time_varying_known_categoricals` — `is_earnings_day`, since a scheduled earnings date is known in advance
  - `train_tft()` fits the model with `QuantileLoss` (predicts a distribution of outcomes, not just a point estimate)
- [models/train.py](models/train.py) — single train/validation split entry point, logs to MLflow
- [models/predict.py](models/predict.py) — genuine forward-looking inference: constructs synthetic "future" rows beyond the last known date (since TFT's decoder needs *something* there) and forecasts the next 1-5 trading days
- [models/evaluate.py](models/evaluate.py) — `compute_metrics()` (direction accuracy, Sharpe ratio, max drawdown, annualized return — see [Section 8](#8-stock-market--technical-analysis-glossary)) and `naive_baseline_metrics()` (the "market always goes up" baseline every model must beat to be worth using)

**Why the TFT architecture specifically?** (from the original guide) It natively supports mixing static, known-future, and unknown-future covariates in one model, and its attention mechanism is at least partially interpretable (which variables the model weighted most for a given prediction) — unlike a black-box LSTM.

**Environment issues fixed along the way** (worth remembering if you rebuild this environment):
1. `pytorch_forecasting`'s `TemporalFusionTransformer` inherits from `lightning.pytorch.LightningModule`, **not** the legacy `pytorch_lightning` package — using the wrong import causes a cryptic `TypeError` at `trainer.fit()`.
2. MLflow 3.x dropped the plain filesystem tracking backend (`./mlruns`) — switched to `sqlite:///mlflow.db`.
3. `tft.predict()` spawns its own `Trainer` that auto-detects **all** GPUs on a shared multi-GPU server and tries (and fails) distributed NCCL sync. Fix: always pass `trainer_kwargs={"devices": 1}`, and prefix training commands with `CUDA_VISIBLE_DEVICES=0`.
4. Saving the full model object via `torch.save(tft, path)` fails with an unpicklable-local-object error. Fix: use Lightning's own `trainer.save_checkpoint()` + `TemporalFusionTransformer.load_from_checkpoint()` — the documented pattern for this library.
5. `torch` defaulted to CUDA 13 wheels on install, but the server's driver (550.54.14) only supports up to CUDA 12.4. Fixed by explicitly installing `torch==2.6.0+cu124`.

**A critical bug found and fixed later (during the "improve model quality" pass):** the original training loop passed **no validation set** to the trainer at all. This meant:
- `EarlyStopping` was silently monitoring *training* loss (which rarely plateaus early, since the model can always keep memorizing) instead of *validation* loss
- The TFT's built-in `ReduceLROnPlateau` scheduler (conditioned on `val_loss`) silently never fired
- `train.py`'s reported accuracy was computed on only **5 days** of held-out data regardless of the configured validation window size, because the validation dataset was built with `predict=True` (which only yields the dataframe's last 5 rows) instead of scanning across the whole intended window

All three were fixed: `create_train_val_datasets()` now carves an internal validation slice out of every training window, `train_tft()` selects the best-validation checkpoint (not just the last epoch) via `ModelCheckpoint`, and evaluation uses `predict_window()` (in `models/evaluate.py`) which correctly scans every day in a held-out window via `min_prediction_idx`, not just the last few rows.

**Learning rate tuning:** Lightning's `lr_find` (via `lightning.pytorch.tuner.Tuner`) showed the original default learning rate (`1e-3`) was about **8x too high**, causing the model to converge prematurely to a mediocre solution. The new default is `1.3e-4`.

---

### Phase 5 — Walk-Forward Backtesting

**Goal:** honestly measure whether the model's predictions are actually worth anything, using a methodology that can't fool itself with look-ahead bias.

**What was built:** [backtest/walkforward.py](backtest/walkforward.py) — rolls an 18-month training window forward in 3-month steps. At each step:
1. Train a **fresh** TFT only on data strictly before the test window (no leakage)
2. Predict 1-day-ahead returns for every day in the following 3-month test window
3. Roll forward 3 months and repeat

This produces 11 independent test windows per ticker spanning 2022–2026, giving a much more honest accuracy estimate than a single train/test split (which can get lucky or unlucky depending on which period you happened to pick).

**Why walk-forward instead of a normal train/test split?** Financial time series aren't i.i.d. — a random shuffle-based split would let the model "see the future" (train on data chronologically after some test points), producing inflated backtest accuracy that evaporates in live trading. Walk-forward guarantees every prediction is made using only information that would have actually been available at that point in time.

**Results (after the validation/early-stopping/learning-rate fixes described in Phase 4):**

| Ticker | Model accuracy | Model Sharpe | Baseline accuracy | Baseline Sharpe |
|---|---|---|---|---|
| AAPL | 52.0% | 0.67 | 53.9% | 0.93 |
| MSFT | 49.8% | 0.14 | 52.2% | 0.25 |
| NVDA | 51.6% | 1.15 | 55.3% | 1.28 |
| SAP.DE | 47.4% | 0.15 | 53.7% | 0.73 |

**Honest conclusion:** none of the 4 tickers beat the naive "always predict up" baseline. This matches the original dev guide's own stated bar — *"if your model doesn't beat ~58% direction accuracy consistently, the signal isn't real"* — we're at 47-52%, meaningfully short of that. This is not a failure of engineering; short-horizon single-stock direction prediction from public data is genuinely one of the hardest problems in quantitative finance, and the guide itself set expectations accordingly (*"most academic papers claim 60-65% — that's the realistic ceiling"*).

**Decision made because of this result:** the real-time alert engine (Phase 6) was deliberately built on interpretable sentiment/event rules, **not** on the TFT's predictions, specifically because a weak model producing confident-looking alerts would recreate the exact problem you said you're trying to solve (missing/misreading signals and losing money).

---

### Dashboard & API (parallel to Phases 4-7)

**[dashboard/app.py](dashboard/app.py)** — Streamlit, chosen over Dash/React because the whole stack is already Python and a personal analytics tool doesn't need a separate frontend/backend split yet. Sections, top to bottom:
1. **🔔 Recent Alerts** — the most time-sensitive information, placed first
2. Fundamentals header (price, market cap, P/E, beta, dividend yield, 52-week range, analyst consensus)
3. Price chart — candlestick + EMA/Bollinger overlays + volume
4. **TFT Forecast** — next 1-5 day predicted returns, with an explicit caption pointing at the backtest panel so a user doesn't mistake the forecast for something proven
5. RSI/MACD panel
6. Sentiment panel — daily net sentiment + article count, earnings days flagged
7. Backtest performance — pulls live from MLflow, shows model vs. baseline

**[api/main.py](api/main.py)** — FastAPI, 8 endpoints: `/health`, `/tickers`, `/prices/{ticker}`, `/features/{ticker}`, `/sentiment/{ticker}`, `/fundamentals/{ticker}`, `/backtest/{ticker}`, `/predict/{ticker}`. Auto-generated docs at `/docs`.

**Environment note:** installing `streamlit` pulled in a newer `anyio`/`starlette` that conflicted with an old pinned `fastapi==0.104.1`. Fixed by upgrading to `fastapi>=0.110`.

---

### Phase 6 — Real-Time Alert Engine

**Goal (yours, not the original guide's):** get notified the moment something changes, instead of finding out after the market has already moved. You were explicit that this is the most important feature and that weak signal is worse than no signal.

**Decision:** built on interpretable, already-validated signals — not the TFT (see Phase 5's honest results above).

**What was built:**
- [alerts/rules.py](alerts/rules.py) — four independent trigger functions, each returning a structured alert dict (severity, message, value, z-score, date):
  - `check_sentiment_spike()` — today's net FinBERT sentiment vs. its own 30-day rolling mean/std, z-score ≥ 2.0
  - `check_volume_anomaly()` — today's trading volume vs. 30-day rolling baseline, z-score ≥ 3.0 (only flags spikes, not unusually quiet days)
  - `check_price_move()` — single-day close-to-close move ≥ 3%
  - `check_earnings_surprise()` — |EPS surprise| ≥ 5%, using the leakage-safe next-day-visible value from `features/events.py`
- [alerts/watcher.py](alerts/watcher.py) — the orchestration layer:
  - `build_alert_view()` — **deliberately bypasses** the model-training feature pipeline (`features/technical.py`), because that pipeline always drops the most recent row (its target is unknown until tomorrow) — exactly the row an alert needs to see *today*. Instead it does a lighter merge of raw price + sentiment + events that keeps every row.
  - `refresh_ticker()` — pulls fresh price (yfinance), RSS headlines (no rate limit), NewsAPI (rate-limited, see below), scrapes new articles, re-scores with FinBERT, re-aggregates daily sentiment
  - `poll_once()` / `run_forever()` — `run_forever()` uses the `schedule` package, checking every `ALERT_POLL_INTERVAL_MINUTES` (default 30)
  - `backfill_alerts()` — scans recent history (default 90 days) to seed the alerts feed on first use, so the dashboard isn't empty before any live triggers happen
  - Alerts are deduplicated by `(ticker, rule, date)` so a re-poll on the same trading day doesn't spam the same alert repeatedly

**NewsAPI rate limiting:** NewsAPI's free tier caps at 100 requests/day total. Polling every 30 minutes × 4 tickers would blow through that fast, so NewsAPI calls run on their **own** 90-minute cooldown (tracked per-ticker in `data/meta/newsapi_last_fetch.json`, persisted across restarts), independent of the 30-minute RSS/price refresh — keeping total usage to roughly 64 requests/day.

**Validated:** rules correctly fired on a real historical event in the data (AAPL, 2026-06-26: a genuine +13σ volume spike, +3.14% price move, and +2.88σ sentiment swing — all three rules triggered as expected). Backfill populated 93 real historical alerts across all 4 tickers.

**Your choice on notifications:** dashboard-only feed for now — no push (email/Slack/desktop) yet. The trigger logic in `alerts/rules.py` is decoupled from delivery, so adding a push channel later only requires a new step inside `watcher.py`'s `_log_alerts()`.

**Threshold tuning note:** at the current 3% threshold, `price_move` fired on ~17% of ticker-days in the backfill — that's closer to "normal volatility" than "alert-worthy" for a volatile stock like NVDA. All thresholds live in `config.py` (`ALERT_*` constants) and are worth tightening based on your own risk tolerance once you've watched the feed for a while.

---

## 5. Current Data & Model Status

*(snapshot as of this writing — will drift as you keep running things)*

**Articles:** 6,423 total in `data/meta/articles_meta.json` — 4,506 successfully scraped to full text, 1,917 failed (dead links, paywalls, etc. — normal attrition, not a bug).

| Ticker | Article count | Date coverage |
|---|---|---|
| AAPL | 2,196 | 2022-01-03 → 2026-07-01 (gap: all of 2025 – mid-2026 from historical sources; live RSS/NewsAPI cover current dates) |
| MSFT | 2,542 | 2022-01-03 → 2026-06-23 (same gap) |
| NVDA | 1,518 | 2022-01-04 → 2026-06-23 (same gap) |
| SAP.DE | 167 | 2022-01-11 → 2026-06-22 (same gap, and much thinner coverage overall — SAP has far less English-language financial news volume) |

**Feature matrices:** all 4 tickers built cleanly, ~1075-1093 rows, 28 columns, zero NaN gaps.

**Model checkpoints:** all 4 tickers trained and saved to `models/checkpoints/` (`.ckpt` + `dataset_params.pkl` pairs) — `/predict` and the dashboard's forecast panel work for all of them.

**Backtest results:** see the table in Phase 5 above — none currently beat the naive baseline.

**Alerts:** 93 alerts backfilled across all 4 tickers' last 90 days, live in the dashboard.

**MLflow experiments:** `foreticker_tft` (single train/val split runs) and `foreticker_walkforward` (the honest rolling backtest) — tracked in `mlflow.db` (SQLite). View with `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

---

## 6. How to Run Everything

```bash
# Environment setup (once)
conda create -n foreticker python=3.11
conda activate foreticker
pip install -r requirements.txt
cp .env.example .env   # then fill in NEWS_API_KEY, ALPHA_VANTAGE_KEY

# --- Phase 1: Data ---
python ingestion/price_fetcher.py                    # OHLCV for all DEFAULT_TICKERS
python ingestion/news_collector.py                    # historical news (Alpha Vantage + RSS)
python ingestion/article_scraper.py                   # scrape full text for pending articles

# --- Phase 2: NLP ---
python nlp/finbert_scorer.py                          # score all scraped articles (GPU)
python nlp/sentiment_aggregator.py                    # roll up to daily sentiment
python nlp/ollama_extractor.py                        # deep extraction on top/bottom 10% (slow, needs Ollama running)

# --- Phase 3: Features ---
python features/builder.py                            # merge everything into final feature matrix

# --- Phase 4: Train ---
CUDA_VISIBLE_DEVICES=0 python models/train.py --ticker AAPL --epochs 50

# --- Phase 5: Backtest ---
CUDA_VISIBLE_DEVICES=0 python backtest/walkforward.py --epochs 50   # all tickers
CUDA_VISIBLE_DEVICES=0 python backtest/walkforward.py --ticker AAPL --epochs 50   # single ticker

# --- Predict ---
CUDA_VISIBLE_DEVICES=0 python models/predict.py --ticker AAPL --days 5

# --- Alerts ---
python -m alerts.watcher --backfill                    # seed feed with recent history
python -m alerts.watcher --once                        # single check, good for cron
python -m alerts.watcher                                # run forever, polls every 30 min

# --- Serving ---
streamlit run dashboard/app.py --server.port 8510
uvicorn api.main:app --host 0.0.0.0 --port 8011

# --- Experiment tracking ---
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Important environment notes:**
- Always prefix training/prediction commands with `CUDA_VISIBLE_DEVICES=0` on this multi-GPU server — otherwise `pytorch_forecasting`'s internal `Trainer.predict()` calls try to use all GPUs and crash with an NCCL error.
- This server shares ports with other unrelated projects — 8501 (Streamlit's default) is occupied; we use 8510 for the dashboard and 8011 for the API.

---

## 7. Known Issues & Limitations

**1. 2025 – mid-2026 news data gap.** Alpha Vantage's free-tier daily quota (25 requests/day) was exhausted partway through the historical backfill. AAPL/MSFT got through 2022-2024; NVDA got 2022-2023; SAP.DE got almost nothing. Two ways to close it:
- Get a real (non-placeholder) Alpha Vantage key and re-run `collect_alphavantage()` for just the 2025–2026 window (only ~8 more requests needed — well under the daily cap)
- Retry GDELT — but see the next point.

**2. GDELT is currently blocked.** Early heavy interactive testing triggered what looks like a longer-duration throttle (not the rolling window their docs describe) — even a conservative 10-second delay between requests still gets immediately rate-limited. Worth retrying after more time has passed, or just accepting Alpha Vantage/NewsAPI/RSS as the ongoing sources.

**3. The TFT model doesn't beat the naive baseline.** Walk-forward direction accuracy is 47-52% across all 4 tickers, against a baseline of 52-55%. This is the honest, bug-free result after fixing a real validation/early-stopping bug and tuning the learning rate — it's not an artifact of broken methodology, the signal from this feature set + architecture genuinely isn't there yet at the 58% "real signal" bar the original guide set. Possible next steps if you want to keep pushing on this:
   - Hyperparameter search (Optuna via `pytorch_forecasting`'s tuning utilities) — no guarantee of success, short-horizon single-stock direction is a genuinely hard problem
   - Feature pruning — 11 unknown-real features against ~300-450 rows per training window risks overfitting to noise
   - Wire in the Ollama-extracted features (`impact_horizon`, `macro_event`) that are currently computed but unused
   - Try a different target formulation (e.g. classification directly instead of quantile regression on returns)

**4. Alert thresholds haven't been tuned against real usage yet.** `price_move` at 3% fires on about 1-in-6 ticker-days — reasonable for flagging genuine volatility but likely too loose if you want alerts to mean "this is unusual," not "this is a normal Tuesday for NVDA." Tune `ALERT_*` constants in `config.py` as you see false positives/negatives.

**5. No push notifications yet.** Alerts are dashboard-only by your choice. Email/Slack/desktop push would need to be added to `alerts/watcher.py`'s `_log_alerts()`.

**6. SAP.DE has thin data.** Only 167 articles total vs. 1,500-2,500+ for the US tickers — much less English-language financial news volume for a German company. Its sentiment signal is noisier as a result.

---

## 8. Stock Market & Technical Analysis Glossary

This section explains every indicator, metric, and term used in this project in plain language — written for someone who reads stock charts but hasn't gone deep on the math behind them.

### Price data basics

**OHLCV** — Open, High, Low, Close, Volume. The four price points and trading volume for one time period (in this project, one trading day). "Close" is the most-quoted number; "Volume" is how many shares changed hands.

**Candlestick chart** — a chart where each day is drawn as a "candle": a thick body showing the open-to-close range (green/white if the close was higher than the open, red/black if lower), with thin "wicks" above and below showing the day's high and low. More information-dense than a simple line chart.

**Return** — the percentage change in price over some period. "1-day return" = (today's close − yesterday's close) / yesterday's close. This project's model predicts this number for tomorrow.

**Direction accuracy** — out of all days predicted, what fraction did the model correctly call as "up" or "down"? A coin flip gets 50%. This project's naive baseline (always guessing "up," since stocks drift upward over time on average) gets 52-55%.

### Technical indicators (used in `features/technical.py`)

**EMA — Exponential Moving Average.** A moving average that weights recent prices more heavily than older ones (unlike a simple average, which weights every day equally). EMA 20 and EMA 50 (20-day and 50-day) are used here. When the shorter EMA crosses above the longer one, that's traditionally read as a bullish signal ("golden cross" territory); crossing below is bearish.

**MACD — Moving Average Convergence Divergence.** The difference between a fast EMA and a slow EMA (typically 12-day and 26-day), plus a "signal line" (a 9-day EMA of that difference itself). When MACD crosses above its signal line, it's often read as bullish momentum building; crossing below, bearish. The `macd_diff` column is MACD minus its signal line — the "histogram" you often see plotted as bars under a MACD chart.

**RSI — Relative Strength Index.** A momentum oscillator from 0-100 measuring the speed and size of recent price changes. Above 70 is traditionally considered "overbought" (may be due for a pullback); below 30 is "oversold" (may be due for a bounce). It's a lagging, mean-reversion-flavored signal, not a crystal ball — strong trends can stay "overbought" for a long time.

**Stochastic Oscillator (`stoch_k`, `stoch_d`)** — another momentum indicator, comparing today's close to the recent high-low range (over 14 days by default) rather than to past closes like RSI does. `stoch_k` is the raw calculation; `stoch_d` is a smoothed 3-day average of it. Same overbought/oversold interpretation as RSI (above 80 / below 20 are common thresholds), just computed differently.

**Bollinger Bands (`bb_high`, `bb_low`, `bb_pct`)** — a moving average (20-day) with two bands drawn 2 standard deviations above and below it. Prices tend to stay within the bands most of the time; touching or breaking a band is read as an extreme move. `bb_pct` expresses where the current price sits between the bands as a 0-1 percentage (0 = at the lower band, 1 = at the upper band, >1 or <0 = broken through a band).

**ATR — Average True Range.** A pure volatility measure (14-day) — how much a stock typically moves in a day, regardless of direction. Higher ATR = more volatile stock. Used here as a model input, not a directional signal by itself.

**OBV — On-Balance Volume.** A running total that adds a day's volume when the price closes up and subtracts it when the price closes down. The idea: volume often leads price — if OBV is rising while price is flat, that can suggest accumulation (buying pressure building) before the price itself moves.

**VWAP — Volume-Weighted Average Price.** The average price a stock traded at, weighted by how much volume traded at each price level. Institutional traders often use VWAP as a benchmark — "did I get a better or worse price than the day's VWAP?"

### Fundamentals (used in `features/fundamentals.py` and the dashboard header)

**Market Cap** — total value of all outstanding shares (share price × number of shares). The standard measure of "how big is this company."

**P/E Ratio (trailing / forward)** — Price-to-Earnings. Share price divided by earnings per share. "Trailing" uses the last 12 months of actual earnings; "forward" uses analysts' estimated *next* 12 months. A high P/E generally means the market expects strong future growth (or the stock is expensive relative to current profits); a low P/E can mean the market is skeptical, or the stock is undervalued — context (industry, growth rate) matters a lot here.

**EPS — Earnings Per Share.** Net income divided by number of shares outstanding. The per-share profit number that P/E is built from.

**EPS Surprise** — the difference between a company's *actual* reported EPS and what analysts had *estimated* beforehand, as a percentage. A big positive surprise ("beat") often moves the stock up sharply; a big miss often moves it down. This project treats the surprise as only "knowable" the trading day *after* the earnings announcement (see the leakage note in Phase 3) — using the actual reported number on the announcement day itself, before it's public, would be looking into the future.

**Beta** — a measure of how much a stock moves relative to the overall market. Beta of 1.0 = moves in line with the market; beta > 1 = more volatile than the market (amplified moves both directions); beta < 1 = less volatile.

**Dividend Yield** — annual dividend per share divided by share price, as a percentage. How much cash income you get relative to what you paid, ignoring price appreciation.

**52-Week High/Low** — the highest and lowest price the stock has traded at over the past year. Commonly used as reference points ("trading near its 52-week high" is often read as strong momentum).

**Return on Equity (ROE) / Profit Margins / Revenue Growth** — profitability and growth metrics from the company's financials, included in the fundamentals panel for context but not currently fed into the model.

**Analyst Consensus / Target Price** — the aggregated "buy/hold/sell" recommendation and average 12-month price target from Wall Street analysts covering the stock.

### Sentiment analysis

**FinBERT** — a version of the BERT language model fine-tuned specifically on financial text, so it understands finance-specific language better than a general-purpose sentiment model (e.g., it knows "beat estimates" is positive and "missed guidance" is negative, in ways a generic model might not). Outputs three probabilities per piece of text: positive, negative, neutral.

**Net sentiment** — `positive − negative` from FinBERT's output. Ranges from -1 (maximally negative) to +1 (maximally positive). This project's `sentiment_mean` column is the average net sentiment across all articles for a ticker on a given day.

**Sentiment z-score** — how many standard deviations today's sentiment is from its own recent (30-day) average. Used by the alert engine: a z-score of +2.88 (as seen in the real AAPL example from this session) means sentiment that day was unusually more positive than the stock's own recent normal — a much more meaningful signal than looking at the raw sentiment number alone, since "normal" sentiment differs a lot between companies.

### Machine learning & backtesting metrics

**Sharpe Ratio** — the classic risk-adjusted-return metric: average return divided by the volatility (standard deviation) of those returns, annualized. A higher Sharpe means more return per unit of risk taken. A Sharpe of 1.0 is generally considered good, 2.0+ excellent, negative means you're losing money after adjusting for risk taken. Comparing a model's Sharpe against the naive baseline's Sharpe is more informative than comparing raw returns, since it accounts for how bumpy the ride was to get there.

**Max Drawdown** — the largest peak-to-trough decline in cumulative returns over the test period, as a percentage. If you'd started with $100 and at some point your equity dipped to $70 before recovering, that's a 30% max drawdown — a measure of "how bad could it have gotten if you'd bought at the worst possible time."

**Annualized Return** — the average daily strategy return, scaled up to a yearly figure (`× 252`, the typical number of US trading days in a year) for easier comparison against things like savings account interest or index fund returns.

**Walk-forward validation** — the backtesting method used in Phase 5: repeatedly train on data up to some date, test on the following period, then roll the whole window forward and repeat. This is the financial-time-series-appropriate alternative to randomly shuffling data into train/test sets (which would let a model "see the future," producing fake, inflated accuracy).

**Look-ahead bias / data leakage** — using information in training or feature construction that wouldn't actually have been available at the time a real prediction would have been made. The most dangerous kind of bug in this domain because it makes backtests look great while being completely useless in live trading. This project's leakage-safe design choices (trading-day alignment for news, next-day-only EPS surprise visibility, walk-forward validation, dropping the last row whose target is unknown) all exist specifically to prevent this.

**Naive Baseline** — the simplest possible "model": always predict the same thing (here, "the stock will go up," since markets drift upward over most multi-year periods). Any real model has to beat this consistently to be worth using — if it can't, it isn't adding information, just noise dressed up as a forecast.

**Quantile Loss** — the loss function this project's TFT is trained with. Instead of predicting a single number (e.g., "tomorrow's return will be +0.5%"), it predicts a *range* of outcomes at different confidence levels (e.g., "there's a 10% chance the return is below X, a 50% chance it's below the median, a 90% chance it's below Y"). More honest than a single point estimate, since it also communicates *uncertainty*, not just a guess.

**Temporal Fusion Transformer (TFT)** — the specific neural network architecture used for forecasting. Built on the Transformer/attention mechanism from modern NLP, but adapted for time series — it can mix static features (like which ticker this is), features whose future is known (like a scheduled earnings date), and features whose future is unknown (like tomorrow's RSI, which depends on tomorrow's unknowable price) in one coherent model, and its attention weights offer some interpretability into which time steps and features mattered most for a given prediction.

**Encoder length / Prediction length** — the TFT is configured with a 60-day "encoder length" (how much history it looks back on to make a prediction) and a 5-day "prediction length" (how far into the future it forecasts in one shot). This project only actually *uses* the first of those 5 predicted days (the "1-day-ahead" prediction) for evaluation and alerts, though `models/predict.py` can surface all 5.

---

## 9. Future Roadmap

Roughly in priority order, based on what's been discussed:

1. **Tune alert thresholds** against real, ongoing usage — the current `price_move` threshold in particular fires too often to feel like a genuine "alert."
2. **Add push notifications** (email/Slack/desktop) once you've validated the dashboard feed is catching the right things.
3. **Decide on the model's fate**: invest further in hyperparameter tuning / feature pruning to try to clear the 58% bar, or accept it as a secondary signal and lean fully on the sentiment/event-based alert engine.
4. **Wire in the unused Ollama-extracted signals** (`impact_horizon`, `macro_event`) — these are already computed and sitting in `data/processed/sentiments/{TICKER}_ollama.parquet` but not used by either the model or the alert engine yet.
5. **Close the 2025 data gap** — either a real Alpha Vantage key or another GDELT attempt once its throttle likely clears.
6. **More tickers** — the pipeline is fully parameterized by `DEFAULT_TICKERS` in `config.py`; adding a ticker is mostly a matter of running the pipeline phases for it, though each new ticker needs its own price fetch, news backfill, and model training/backtest.
7. **Testing & hardening** — no automated tests exist yet anywhere in the codebase; this is a personal-scale project, but if it grows, worth adding at least smoke tests for the data pipeline and rule functions.
