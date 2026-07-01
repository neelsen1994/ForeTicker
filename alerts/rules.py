"""
Alert trigger rules — sentiment/event-based, not TFT-based.

The TFT's walk-forward accuracy doesn't beat the naive baseline (see
backtest/walkforward.py results), so alerts are built on signals that are
directly interpretable and already validated: sentiment swings, earnings
surprises, and volume/price anomalies.

Each rule takes a ticker's feature DataFrame (most recent row = "today")
and returns a list of triggered alert dicts. Empty list = no trigger.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import (
    ALERT_SENTIMENT_ZSCORE,
    ALERT_VOLUME_ZSCORE,
    ALERT_PRICE_MOVE_PCT,
    ALERT_EARNINGS_SURPRISE_PCT,
    ALERT_ROLLING_WINDOW_DAYS,
)


def _zscore(today: float, baseline: pd.Series) -> float | None:
    mean, std = baseline.mean(), baseline.std()
    if std == 0 or pd.isna(std):
        return None
    return (today - mean) / std


def check_sentiment_spike(
    ticker: str, df: pd.DataFrame,
    window: int = ALERT_ROLLING_WINDOW_DAYS, z_thresh: float = ALERT_SENTIMENT_ZSCORE,
) -> list[dict]:
    """Flags when today's net sentiment is an outlier vs its own recent history."""
    if len(df) < window + 1:
        return []

    baseline = df["sentiment_mean"].iloc[-(window + 1):-1]
    today_val = df["sentiment_mean"].iloc[-1]
    z = _zscore(today_val, baseline)
    if z is None or abs(z) < z_thresh:
        return []

    direction = "bullish" if z > 0 else "bearish"
    severity = "high" if abs(z) >= z_thresh * 1.5 else "medium"
    return [{
        "ticker": ticker,
        "rule": "sentiment_spike",
        "severity": severity,
        "message": f"{ticker}: news sentiment swung {direction} ({z:+.2f}σ vs {window}-day baseline)",
        "value": round(float(today_val), 4),
        "zscore": round(float(z), 2),
        "date": str(df.index[-1].date()),
    }]


def check_volume_anomaly(
    ticker: str, df: pd.DataFrame,
    window: int = ALERT_ROLLING_WINDOW_DAYS, z_thresh: float = ALERT_VOLUME_ZSCORE,
) -> list[dict]:
    """Flags when today's trading volume is an outlier vs its own recent history."""
    if len(df) < window + 1:
        return []

    baseline = df["Volume"].iloc[-(window + 1):-1]
    today_val = df["Volume"].iloc[-1]
    z = _zscore(today_val, baseline)
    if z is None or z < z_thresh:  # only alert on spikes, not unusually quiet days
        return []

    severity = "high" if z >= z_thresh * 1.5 else "medium"
    return [{
        "ticker": ticker,
        "rule": "volume_anomaly",
        "severity": severity,
        "message": f"{ticker}: trading volume spiked to {int(today_val):,} ({z:+.2f}σ vs {window}-day baseline)",
        "value": int(today_val),
        "zscore": round(float(z), 2),
        "date": str(df.index[-1].date()),
    }]


def check_price_move(
    ticker: str, df: pd.DataFrame, pct_thresh: float = ALERT_PRICE_MOVE_PCT,
) -> list[dict]:
    """Flags a large single-day price move (close-to-close)."""
    if len(df) < 2:
        return []

    prev_close = df["Close"].iloc[-2]
    today_close = df["Close"].iloc[-1]
    if prev_close == 0:
        return []

    pct_change = (today_close - prev_close) / prev_close * 100
    if abs(pct_change) < pct_thresh:
        return []

    direction = "up" if pct_change > 0 else "down"
    severity = "high" if abs(pct_change) >= pct_thresh * 1.5 else "medium"
    return [{
        "ticker": ticker,
        "rule": "price_move",
        "severity": severity,
        "message": f"{ticker}: price moved {direction} {abs(pct_change):.2f}% in a day",
        "value": round(float(pct_change), 2),
        "date": str(df.index[-1].date()),
    }]


def check_earnings_surprise(
    ticker: str, df: pd.DataFrame, pct_thresh: float = ALERT_EARNINGS_SURPRISE_PCT,
) -> list[dict]:
    """Flags a large EPS surprise, visible the trading day after the announcement."""
    if len(df) < 1:
        return []

    today_val = df["eps_surprise_pct"].iloc[-1]
    if pd.isna(today_val) or today_val == 0 or abs(today_val) < pct_thresh:
        return []

    direction = "beat" if today_val > 0 else "missed"
    severity = "high" if abs(today_val) >= pct_thresh * 2 else "medium"
    return [{
        "ticker": ticker,
        "rule": "earnings_surprise",
        "severity": severity,
        "message": f"{ticker}: earnings {direction} estimates by {abs(today_val):.2f}%",
        "value": round(float(today_val), 2),
        "date": str(df.index[-1].date()),
    }]


ALL_RULES = [
    check_sentiment_spike,
    check_volume_anomaly,
    check_price_move,
    check_earnings_surprise,
]


def run_all_rules(ticker: str, df: pd.DataFrame) -> list[dict]:
    """Runs every rule against a ticker's feature DataFrame, returns all triggered alerts."""
    alerts = []
    for rule_fn in ALL_RULES:
        alerts.extend(rule_fn(ticker, df))
    return alerts
