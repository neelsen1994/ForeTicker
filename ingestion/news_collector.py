"""
Collect news article metadata from three sources:
  1. NewsAPI  — historical search by keyword/ticker (requires API key)
  2. RSS feeds — real-time / recent headlines via feedparser
  3. GDELT    — free, no key, years of historical coverage

Output is a list of metadata dicts saved to data/meta/articles_meta.json.
Actual article text is fetched separately by article_scraper.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import (
    NEWS_API_KEY,
    META_DIR,
    ARTICLES_META_FILE,
    RSS_FEEDS,
    GDELT_QUERIES,
    MARKET_CLOSE_HOUR_ET,
)

try:
    from newsapi import NewsApiClient
    _NEWSAPI_AVAILABLE = True
except ImportError:
    _NEWSAPI_AVAILABLE = False

# ET offset from UTC (standard time; DST would be -4 but conservative -5 is safer)
ET_UTC_OFFSET = timedelta(hours=-5)

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_meta() -> list[dict]:
    if ARTICLES_META_FILE.exists():
        with open(ARTICLES_META_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_meta(records: list[dict]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTICLES_META_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def _dedup(records: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = r.get("url", "")
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _align_to_trading_day(pub_dt: datetime) -> str:
    """
    Articles published after 4 PM ET belong to the NEXT trading day's feature set.
    """
    et_dt = pub_dt.astimezone(timezone(ET_UTC_OFFSET))
    if et_dt.hour >= MARKET_CLOSE_HOUR_ET:
        et_dt += timedelta(days=1)
    return et_dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------------------

def collect_newsapi(
    ticker: str,
    query: str,
    from_date: str,
    to_date: str,
    page_size: int = 100,
    max_pages: int = 5,
) -> list[dict]:
    """Fetch article metadata from NewsAPI (requires API key, 30-day limit on free tier)."""
    if not _NEWSAPI_AVAILABLE:
        print("[news_collector] newsapi-python not installed — skipping NewsAPI")
        return []
    if not NEWS_API_KEY:
        print("[news_collector] NEWS_API_KEY not set — skipping NewsAPI")
        return []

    client = NewsApiClient(api_key=NEWS_API_KEY)
    records: list[dict] = []

    for page in range(1, max_pages + 1):
        try:
            resp = client.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language="en",
                sort_by="publishedAt",
                page_size=page_size,
                page=page,
            )
        except Exception as e:
            print(f"[news_collector] NewsAPI error (page {page}): {e}")
            break

        articles = resp.get("articles", [])
        if not articles:
            break

        for art in articles:
            pub_str = art.get("publishedAt", "")
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                trading_date = _align_to_trading_day(pub_dt)
            except ValueError:
                trading_date = pub_str[:10]

            records.append({
                "url": art.get("url", ""),
                "ticker": ticker,
                "date": trading_date,
                "published_at": pub_str,
                "source": art.get("source", {}).get("name", "newsapi"),
                "title": art.get("title", ""),
                "scrape_status": "pending",
                "file": None,
            })

        if page * page_size >= resp.get("totalResults", 0):
            break

    print(f"[news_collector] NewsAPI: {len(records)} articles for {ticker}")
    return records


# ---------------------------------------------------------------------------
# RSS
# ---------------------------------------------------------------------------

def collect_rss(ticker: str, feeds: Optional[list[str]] = None) -> list[dict]:
    """Parse RSS feeds for a ticker and return article metadata."""
    if feeds is None:
        feeds = RSS_FEEDS.get(ticker, [])

    records: list[dict] = []

    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"[news_collector] RSS parse error ({feed_url}): {e}")
            continue

        for entry in parsed.entries:
            url = entry.get("link", "")
            if not url:
                continue

            pub_struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub_struct:
                pub_dt = datetime(*pub_struct[:6], tzinfo=timezone.utc)
                trading_date = _align_to_trading_day(pub_dt)
            else:
                trading_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            records.append({
                "url": url,
                "ticker": ticker,
                "date": trading_date,
                "published_at": entry.get("published", ""),
                "source": parsed.feed.get("title", "rss"),
                "title": entry.get("title", ""),
                "scrape_status": "pending",
                "file": None,
            })

    print(f"[news_collector] RSS: {len(records)} articles for {ticker}")
    return records


# ---------------------------------------------------------------------------
# GDELT
# ---------------------------------------------------------------------------

def _gdelt_windows(from_date: str, to_date: str, window_days: int = 14):
    """Yield (start, end) datetime strings in GDELT format: YYYYMMDDHHMMSS."""
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=window_days), end)
        yield (
            cursor.strftime("%Y%m%d%H%M%S"),
            window_end.strftime("%Y%m%d%H%M%S"),
        )
        cursor = window_end


def collect_gdelt(
    ticker: str,
    query: str,
    from_date: str,
    to_date: str,
    max_records: int = 250,
    window_days: int = 14,
    delay_seconds: float = 2.0,
) -> list[dict]:
    """
    Fetch article metadata from GDELT Doc API v2.
    Free, no key needed, covers years of history.
    Paginates by sliding date windows since each request caps at 250 results.
    """
    records: list[dict] = []
    windows = list(_gdelt_windows(from_date, to_date, window_days))
    total_windows = len(windows)

    for i, (win_start, win_end) in enumerate(windows):
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": max_records,
            "startdatetime": win_start,
            "enddatetime": win_end,
            "sort": "DateDesc",
            "format": "json",
        }

        try:
            resp = requests.get(GDELT_API, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[news_collector] GDELT error (window {i+1}/{total_windows}): {e}")
            time.sleep(delay_seconds)
            continue

        articles = data.get("articles") or []
        for art in articles:
            seen_raw = art.get("seendate", "")
            try:
                # GDELT format: 20220115T143000Z
                pub_dt = datetime.strptime(seen_raw, "%Y%m%dT%H%M%SZ").replace(
                    tzinfo=timezone.utc
                )
                trading_date = _align_to_trading_day(pub_dt)
            except ValueError:
                trading_date = seen_raw[:8]
                if len(trading_date) == 8:
                    trading_date = f"{trading_date[:4]}-{trading_date[4:6]}-{trading_date[6:]}"

            url = art.get("url", "")
            if not url:
                continue

            records.append({
                "url": url,
                "ticker": ticker,
                "date": trading_date,
                "published_at": seen_raw,
                "source": art.get("domain", "gdelt"),
                "title": art.get("title", ""),
                "scrape_status": "pending",
                "file": None,
            })

        print(
            f"[news_collector] GDELT {ticker} window {i+1}/{total_windows} "
            f"({win_start[:8]}→{win_end[:8]}): {len(articles)} articles"
        )

        # Be polite — GDELT is a shared public resource
        if i < total_windows - 1:
            time.sleep(delay_seconds)

    print(f"[news_collector] GDELT total: {len(records)} articles for {ticker}")
    return records


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def collect_all(
    tickers: list[str],
    from_date: str,
    to_date: str,
    newsapi_queries: Optional[dict[str, str]] = None,
    gdelt_queries: Optional[dict[str, str]] = None,
    use_gdelt: bool = True,
    use_rss: bool = True,
    use_newsapi: bool = True,
) -> list[dict]:
    """
    Collect metadata from GDELT + RSS + NewsAPI for each ticker.
    Merges with any existing records in articles_meta.json (deduped by URL).
    """
    if newsapi_queries is None:
        newsapi_queries = {t: t for t in tickers}
    if gdelt_queries is None:
        gdelt_queries = GDELT_QUERIES

    existing = _load_meta()
    new_records: list[dict] = []

    for ticker in tickers:
        if use_gdelt:
            query = gdelt_queries.get(ticker, ticker)
            new_records += collect_gdelt(ticker, query, from_date, to_date)
        if use_rss:
            new_records += collect_rss(ticker)
        if use_newsapi:
            query = newsapi_queries.get(ticker, ticker)
            new_records += collect_newsapi(ticker, query, from_date, to_date)

    all_records = _dedup(existing + new_records)
    _save_meta(all_records)
    print(f"[news_collector] Total unique articles in meta: {len(all_records)}")
    return all_records


if __name__ == "__main__":
    from config import DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END

    collect_all(DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END)
