from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()

# API keys
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")

# Ollama
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# NLP models
FINBERT_MODEL: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

# Paths
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./data"))
RAW_PRICES_DIR: Path = DATA_DIR / "raw" / "prices"
RAW_ARTICLES_DIR: Path = DATA_DIR / "raw" / "articles"
PROCESSED_FEATURES_DIR: Path = DATA_DIR / "processed" / "features"
PROCESSED_SENTIMENTS_DIR: Path = DATA_DIR / "processed" / "sentiments"
META_DIR: Path = DATA_DIR / "meta"
ARTICLES_META_FILE: Path = META_DIR / "articles_meta.json"
ALERTS_LOG_FILE: Path = META_DIR / "alerts.json"

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Default tickers and date range
DEFAULT_TICKERS: list[str] = ["AAPL", "MSFT", "NVDA", "SAP.DE"]
DEFAULT_START: str = "2022-01-01"
DEFAULT_END: str = datetime.today().strftime("%Y-%m-%d")

# Market close hour in ET (24h) — articles after this hour go to next trading day
MARKET_CLOSE_HOUR_ET: int = 16

# GDELT search queries per ticker (company name works better than ticker symbol)
GDELT_QUERIES: dict[str, str] = {
    "AAPL":  '"Apple" OR "AAPL" stock',
    "MSFT":  '"Microsoft" OR "MSFT" stock',
    "NVDA":  '"NVIDIA" OR "NVDA" stock',
    "SAP.DE": '"SAP" software stock',
}

# Alert thresholds — tune these as you see false positives/negatives in practice
ALERT_SENTIMENT_ZSCORE: float = 2.0       # sentiment_mean move vs its own rolling mean/std
ALERT_VOLUME_ZSCORE: float = 3.0          # volume vs rolling mean/std
ALERT_PRICE_MOVE_PCT: float = 3.0         # daily |return| in percent
ALERT_EARNINGS_SURPRISE_PCT: float = 5.0  # |EPS surprise| in percent
ALERT_ROLLING_WINDOW_DAYS: int = 30       # window used to compute rolling mean/std baselines
ALERT_POLL_INTERVAL_MINUTES: int = 30

# NewsAPI free tier caps at 100 requests/day total (shared across all tickers).
# A separate, slower cooldown than the 30-min RSS/price poll keeps well under that:
# 4 tickers x 1 request every 90 min = ~64 requests/day.
NEWSAPI_POLL_INTERVAL_MINUTES: int = 90
NEWSAPI_LOOKBACK_DAYS: int = 3
NEWSAPI_LAST_FETCH_FILE: Path = META_DIR / "newsapi_last_fetch.json"

# RSS feeds per ticker (extend as needed)
RSS_FEEDS: dict[str, list[str]] = {
    "AAPL": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL&region=US&lang=en-US",
    ],
    "MSFT": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT&region=US&lang=en-US",
    ],
    "NVDA": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=NVDA&region=US&lang=en-US",
    ],
    "SAP.DE": [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SAP.DE&region=US&lang=en-US",
    ],
}
