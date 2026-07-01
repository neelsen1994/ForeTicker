"""
Real-time-ish alert watcher: periodically refreshes data for each ticker and
checks it against alerts/rules.py, logging any triggers to a persistent feed
(data/meta/alerts.json) that the dashboard reads.

Deliberately does NOT use the TFT model — see backtest/walkforward.py results,
where the model doesn't beat the naive baseline. Alerts are built on directly
interpretable signals: sentiment swings, earnings surprises, volume/price moves.

Run once:      python -m alerts.watcher --once
Run forever:   python -m alerts.watcher
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
from datetime import datetime, timezone

import pandas as pd

from config import (
    RAW_PRICES_DIR,
    PROCESSED_SENTIMENTS_DIR,
    ALERTS_LOG_FILE,
    META_DIR,
    DEFAULT_TICKERS,
    DEFAULT_START,
    DEFAULT_END,
    ALERT_POLL_INTERVAL_MINUTES,
    NEWSAPI_POLL_INTERVAL_MINUTES,
    NEWSAPI_LOOKBACK_DAYS,
    NEWSAPI_LAST_FETCH_FILE,
)
from alerts.rules import run_all_rules


def _load_newsapi_last_fetch() -> dict:
    if NEWSAPI_LAST_FETCH_FILE.exists():
        with open(NEWSAPI_LAST_FETCH_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_newsapi_last_fetch(state: dict) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(NEWSAPI_LAST_FETCH_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _newsapi_due(ticker: str) -> bool:
    """NewsAPI's free tier caps at 100 req/day — throttle independently of the
    30-min RSS/price poll so 4 tickers don't blow through the quota."""
    state = _load_newsapi_last_fetch()
    last = state.get(ticker)
    if last is None:
        return True
    elapsed = datetime.now(timezone.utc) - datetime.fromisoformat(last)
    return elapsed.total_seconds() >= NEWSAPI_POLL_INTERVAL_MINUTES * 60


def _mark_newsapi_fetched(ticker: str) -> None:
    state = _load_newsapi_last_fetch()
    state[ticker] = datetime.now(timezone.utc).isoformat()
    _save_newsapi_last_fetch(state)

SENTIMENT_COLS = ["sentiment_mean", "sentiment_std", "article_count", "bullish_ratio", "bearish_ratio"]


def build_alert_view(ticker: str) -> pd.DataFrame:
    """
    Lightweight merge of price + sentiment + events for alert-checking.

    Deliberately does NOT go through features/technical.py's pipeline: that
    pipeline drops the most recent row (its return_1d target is unknown until
    the next trading day), which is correct for model training but would hide
    today's price/volume/sentiment from the alert engine — exactly the data
    an alert needs to see immediately.
    """
    from features.events import add_event_features

    price_path = RAW_PRICES_DIR / f"{ticker.replace('.', '_')}.parquet"
    if not price_path.exists():
        raise FileNotFoundError(f"No price data for {ticker} — run ingestion/price_fetcher.py first")

    price_df = pd.read_parquet(price_path)
    price_df.index = pd.to_datetime(price_df.index)
    price_df = price_df.sort_index()

    sentiment_path = PROCESSED_SENTIMENTS_DIR / f"{ticker.replace('.', '_')}_daily_sentiment.parquet"
    if sentiment_path.exists():
        sentiment_df = pd.read_parquet(sentiment_path)
        sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])
        sentiment_df = sentiment_df.set_index("date")
        merged = price_df.join(sentiment_df[SENTIMENT_COLS], how="left")
        merged[SENTIMENT_COLS] = merged[SENTIMENT_COLS].ffill(limit=3).fillna(0)
    else:
        merged = price_df.copy()
        for col in SENTIMENT_COLS:
            merged[col] = 0.0

    merged = add_event_features(merged, ticker)
    return merged


def refresh_ticker(ticker: str) -> pd.DataFrame:
    """Pulls the latest price + news + sentiment for one ticker, returns the alert view."""
    from ingestion.price_fetcher import fetch_prices
    from ingestion.news_collector import collect_rss, collect_newsapi, _load_meta, _save_meta, _dedup
    from ingestion.article_scraper import scrape_pending
    from nlp.finbert_scorer import score_ticker
    from nlp.sentiment_aggregator import aggregate_ticker

    fetch_prices(ticker, DEFAULT_START, DEFAULT_END)

    new_records = collect_rss(ticker)

    if _newsapi_due(ticker):
        from_date = (datetime.now(timezone.utc) - pd.Timedelta(days=NEWSAPI_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
        to_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        new_records += collect_newsapi(ticker, ticker, from_date, to_date, max_pages=1)
        _mark_newsapi_fetched(ticker)

    all_records = _dedup(_load_meta() + new_records)
    _save_meta(all_records)

    scrape_pending(batch_size=20, delay_seconds=1.0, ticker_filter=ticker)

    try:
        score_ticker(ticker)
        aggregate_ticker(ticker)
    except (FileNotFoundError, ValueError):
        pass  # no articles yet for this ticker — sentiment columns default to 0 in build_alert_view

    return build_alert_view(ticker)


def check_ticker(ticker: str) -> list[dict]:
    df = refresh_ticker(ticker)
    return run_all_rules(ticker, df)


def _load_alerts() -> list[dict]:
    if ALERTS_LOG_FILE.exists():
        with open(ALERTS_LOG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_alerts(alerts: list[dict]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, default=str)


def _log_alerts(new_alerts: list[dict]) -> int:
    """Appends new alerts, deduped by (ticker, rule, date) so a re-poll on the
    same trading day doesn't spam duplicate entries. Returns count added."""
    existing = _load_alerts()
    seen = {(a["ticker"], a["rule"], a["date"]) for a in existing}

    added = 0
    for alert in new_alerts:
        key = (alert["ticker"], alert["rule"], alert["date"])
        if key in seen:
            continue
        alert["fired_at"] = datetime.now(timezone.utc).isoformat()
        existing.append(alert)
        seen.add(key)
        added += 1

    if added:
        existing.sort(key=lambda a: a["fired_at"], reverse=True)
        _save_alerts(existing)

    return added


def backfill_alerts(tickers: list[str] = DEFAULT_TICKERS, window: int = 90) -> int:
    """
    Runs the rules across each ticker's full history (not just "today") so the
    alerts feed isn't empty on first use. Only scans the last `window` trading
    days per ticker to keep this fast — older history is less actionable anyway.
    """
    from config import ALERT_ROLLING_WINDOW_DAYS

    all_triggered = []
    for ticker in tickers:
        try:
            df = build_alert_view(ticker)
        except FileNotFoundError as e:
            print(f"[watcher] backfill skip {ticker}: {e}")
            continue

        start = max(ALERT_ROLLING_WINDOW_DAYS + 1, len(df) - window)
        for i in range(start, len(df)):
            subset = df.iloc[: i + 1]
            all_triggered.extend(run_all_rules(ticker, subset))

        print(f"[watcher] backfill {ticker}: scanned {len(df) - start} days")

    added = _log_alerts(all_triggered)
    print(f"[watcher] Backfill logged {added} alert(s)")
    return added


def poll_once(tickers: list[str] = DEFAULT_TICKERS) -> list[dict]:
    all_triggered = []
    for ticker in tickers:
        try:
            triggered = check_ticker(ticker)
            all_triggered.extend(triggered)
            status = f"{len(triggered)} alert(s)" if triggered else "no triggers"
            print(f"[watcher] {ticker}: {status}")
        except Exception as e:
            print(f"[watcher] ERROR {ticker}: {e}")

    added = _log_alerts(all_triggered)
    if added:
        print(f"[watcher] Logged {added} new alert(s)")
    return all_triggered


def run_forever(tickers: list[str] = DEFAULT_TICKERS, interval_minutes: int = ALERT_POLL_INTERVAL_MINUTES):
    import schedule

    def job():
        print(f"[watcher] Polling {len(tickers)} ticker(s) at {datetime.now().isoformat()}")
        poll_once(tickers)

    job()
    schedule.every(interval_minutes).minutes.do(job)
    print(f"[watcher] Running every {interval_minutes} minutes. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="run a single poll cycle and exit")
    parser.add_argument("--backfill", action="store_true", help="scan recent history to seed the alerts feed, then exit")
    parser.add_argument("--ticker", default=None, help="target a single ticker instead of all defaults")
    args = parser.parse_args()

    target_tickers = [args.ticker] if args.ticker else DEFAULT_TICKERS

    if args.backfill:
        backfill_alerts(target_tickers)
    elif args.once:
        result = poll_once(target_tickers)
        print(f"[watcher] Done — {len(result)} alert(s) this cycle")
    else:
        run_forever(target_tickers)
