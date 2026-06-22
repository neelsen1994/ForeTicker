"""
Scrape full article text from URLs collected by news_collector.py.
Uses trafilatura for robust boilerplate removal.
Updates scrape_status in articles_meta.json after each batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import trafilatura

from config import RAW_ARTICLES_DIR, META_DIR, ARTICLES_META_FILE


def _load_meta() -> list[dict]:
    if ARTICLES_META_FILE.exists():
        with open(ARTICLES_META_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_meta(records: list[dict]) -> None:
    META_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTICLES_META_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def scrape_url(
    url: str,
    ticker: str,
    article_date: str,
    data_dir: Path = RAW_ARTICLES_DIR,
    min_chars: int = 100,
) -> Optional[dict]:
    """
    Fetch and extract clean text from a single URL.
    Returns a metadata dict on success, None on failure.
    """
    raw_dir = Path(data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return None

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
    )
    if not text or len(text) < min_chars:
        return None

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    fname = f"{article_date}_{ticker}_{url_hash}.txt"
    (raw_dir / fname).write_text(text, encoding="utf-8")

    return {
        "url": url,
        "ticker": ticker,
        "date": article_date,
        "file": fname,
        "char_count": len(text),
    }


def scrape_pending(
    batch_size: int = 50,
    delay_seconds: float = 1.0,
    ticker_filter: Optional[str] = None,
) -> dict[str, int]:
    """
    Read articles_meta.json, scrape all 'pending' entries, update statuses.
    Returns counts: {scraped, failed, skipped}.
    """
    records = _load_meta()
    counts = {"scraped": 0, "failed": 0, "skipped": 0}

    pending = [
        (i, r) for i, r in enumerate(records)
        if r.get("scrape_status") == "pending"
        and (ticker_filter is None or r.get("ticker") == ticker_filter)
    ]

    print(f"[article_scraper] {len(pending)} pending articles to scrape")

    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start: batch_start + batch_size]

        for idx, (record_idx, record) in enumerate(batch):
            url = record.get("url", "")
            ticker = record.get("ticker", "UNKNOWN")
            date = record.get("date", "1970-01-01")

            if not url:
                records[record_idx]["scrape_status"] = "failed"
                counts["failed"] += 1
                continue

            result = scrape_url(url, ticker, date)

            if result:
                records[record_idx]["scrape_status"] = "done"
                records[record_idx]["file"] = result["file"]
                records[record_idx]["char_count"] = result["char_count"]
                counts["scraped"] += 1
            else:
                records[record_idx]["scrape_status"] = "failed"
                counts["failed"] += 1

            # Polite delay between requests
            if idx < len(batch) - 1:
                time.sleep(delay_seconds)

        # Save progress after each batch so partial work isn't lost on crash
        _save_meta(records)
        done_so_far = batch_start + len(batch)
        print(f"[article_scraper] Progress: {done_so_far}/{len(pending)} "
              f"(scraped={counts['scraped']}, failed={counts['failed']})")

    return counts


def validate_articles(ticker: str, min_articles: int = 200) -> bool:
    """Check that we have enough clean scraped articles for a ticker."""
    records = _load_meta()
    done = [r for r in records if r.get("ticker") == ticker and r.get("scrape_status") == "done"]
    n = len(done)
    if n < min_articles:
        print(f"[WARN] {ticker}: only {n} scraped articles (need {min_articles})")
        return False
    print(f"[OK] {ticker}: {n} clean articles available")
    return True


if __name__ == "__main__":
    counts = scrape_pending(batch_size=50, delay_seconds=1.0)
    print(f"[article_scraper] Done: {counts}")
