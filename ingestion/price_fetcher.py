import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
import pandas as pd

from config import RAW_PRICES_DIR, DEFAULT_START, DEFAULT_END, DEFAULT_TICKERS


def fetch_prices(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    data_dir: Path = RAW_PRICES_DIR,
) -> pd.DataFrame:
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(data_dir) / f"{ticker.replace('.', '_')}.parquet"

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({start} → {end})")

    # yfinance multi-level columns when downloading single ticker can vary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    df.to_parquet(out_path)
    print(f"[price_fetcher] {ticker}: {len(df)} trading days saved → {out_path}")
    return df


def load_prices(ticker: str, data_dir: Path = RAW_PRICES_DIR) -> pd.DataFrame:
    path = Path(data_dir) / f"{ticker.replace('.', '_')}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No cached data for {ticker}. Run fetch_prices first.")
    return pd.read_parquet(path)


def validate_prices(df: pd.DataFrame, ticker: str, min_trading_days: int = 500) -> bool:
    n = len(df)
    if n < min_trading_days:
        print(f"[WARN] {ticker}: only {n} trading days (need {min_trading_days})")
        return False

    # Check for unexpected gaps (more than 5 calendar days between consecutive rows)
    gaps = df.index.to_series().diff().dt.days.dropna()
    large_gaps = gaps[gaps > 5]
    if not large_gaps.empty:
        print(f"[WARN] {ticker}: {len(large_gaps)} large date gaps detected")
        for date, gap in large_gaps.items():
            print(f"  {date.date()}: {gap}-day gap")

    missing_close = df["Close"].isna().sum()
    if missing_close > 0:
        print(f"[WARN] {ticker}: {missing_close} missing Close values")
        return False

    print(f"[OK] {ticker}: {n} trading days, no missing Close values")
    return True


def fetch_all(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker in tickers:
        try:
            df = fetch_prices(ticker, start, end)
            validate_prices(df, ticker)
            results[ticker] = df
        except Exception as e:
            print(f"[ERROR] {ticker}: {e}")
    return results


if __name__ == "__main__":
    fetch_all()
