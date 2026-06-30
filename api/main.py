"""
FastAPI serving layer for ForeTicker.

Run with:  uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from config import PROCESSED_FEATURES_DIR, RAW_PRICES_DIR, PROCESSED_SENTIMENTS_DIR, DEFAULT_TICKERS
from features.fundamentals import fetch_fundamentals

app = FastAPI(title="ForeTicker API", version="0.1.0")


def _safe(ticker: str) -> str:
    return ticker.replace(".", "_")


def _available_tickers() -> list[str]:
    return [t for t in DEFAULT_TICKERS if (PROCESSED_FEATURES_DIR / f"{_safe(t)}_features.parquet").exists()]


def _require_ticker(ticker: str) -> str:
    if ticker not in _available_tickers():
        raise HTTPException(status_code=404, detail=f"No data for ticker '{ticker}'")
    return ticker


def _filter_dates(df: pd.DataFrame, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tickers")
def list_tickers():
    return {"tickers": _available_tickers()}


@app.get("/prices/{ticker}")
def get_prices(ticker: str, start: Optional[date] = None, end: Optional[date] = None):
    ticker = _require_ticker(ticker)
    path = RAW_PRICES_DIR / f"{_safe(ticker)}.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = _filter_dates(df, start, end)
    out = df.reset_index().rename(columns={"index": "date", "Date": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@app.get("/features/{ticker}")
def get_features(ticker: str, start: Optional[date] = None, end: Optional[date] = None,
                  limit: int = Query(default=500, le=5000)):
    ticker = _require_ticker(ticker)
    path = PROCESSED_FEATURES_DIR / f"{_safe(ticker)}_features.parquet"
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = _filter_dates(df, start, end)
    df = df.tail(limit)
    out = df.reset_index().rename(columns={"index": "date", "Date": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@app.get("/sentiment/{ticker}")
def get_sentiment(ticker: str, start: Optional[date] = None, end: Optional[date] = None):
    ticker = _require_ticker(ticker)
    path = PROCESSED_SENTIMENTS_DIR / f"{_safe(ticker)}_daily_sentiment.parquet"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"No sentiment data for '{ticker}'")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = _filter_dates(df, start, end)
    out = df.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@app.get("/fundamentals/{ticker}")
def get_fundamentals(ticker: str):
    ticker = _require_ticker(ticker)
    try:
        return fetch_fundamentals(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch fundamentals: {e}")


@app.get("/backtest/{ticker}")
def get_backtest(ticker: str):
    ticker = _require_ticker(ticker)
    try:
        import mlflow
        from config import MLFLOW_TRACKING_URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("foreticker_walkforward")
        if exp is None:
            raise HTTPException(status_code=404, detail="No backtest experiment found")
        runs = client.search_runs(
            exp.experiment_id,
            filter_string=f"params.ticker = '{ticker}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise HTTPException(status_code=404, detail=f"No backtest results for '{ticker}'")
        return runs[0].data.metrics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to query backtest results: {e}")


@app.get("/predict/{ticker}")
def get_prediction(ticker: str, days: int = Query(default=5, ge=1, le=5)):
    ticker = _require_ticker(ticker)
    try:
        from models.predict import predict_next
        forecast = predict_next(ticker, n_days=days)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prediction failed: {e}")

    out = forecast.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")
