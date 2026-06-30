"""
Technical indicators using the `ta` library (pandas-ta is unmaintained / Python <3.11 only).

add_technical_features(df) expects columns: Open, High, Low, Close, Volume
and adds trend, momentum, volatility, and volume indicators, plus the
prediction target (next-day return / direction).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import ta


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: Open, High, Low, Close, Volume"""
    df = df.copy()

    close, high, low, volume = df["Close"], df["High"], df["Low"], df["Volume"]

    # Trend
    df["ema_20"] = ta.trend.EMAIndicator(close, window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    macd = ta.trend.MACD(close)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Momentum
    df["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()
    df["atr_14"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

    # Volume
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(high, low, close, volume).volume_weighted_average_price()

    # Target: next-day return (what we're predicting) — never use as an input feature
    df["return_1d"] = df["Close"].pct_change().shift(-1)
    df["target_direction"] = (df["return_1d"] > 0).astype(int)

    return df.dropna()


if __name__ == "__main__":
    from config import RAW_PRICES_DIR, DEFAULT_TICKERS

    for ticker in DEFAULT_TICKERS:
        path = RAW_PRICES_DIR / f"{ticker.replace('.', '_')}.parquet"
        if not path.exists():
            print(f"[technical] {ticker}: no price data, skip")
            continue
        df = pd.read_parquet(path)
        out = add_technical_features(df)
        print(f"[technical] {ticker}: {len(out)} rows after indicators (from {len(df)})")
