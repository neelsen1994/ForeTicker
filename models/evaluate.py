"""
Signal-based evaluation metrics — more meaningful than RMSE for trading.
Shared by models/train.py (single train/val split) and backtest/walkforward.py (Phase 5).
"""

import numpy as np
import pandas as pd


def compute_metrics(results_df: pd.DataFrame, threshold: float = 0.0) -> dict:
    """
    results_df must have columns: return_1d (actual), predicted.
    """
    df = results_df.dropna(subset=["return_1d", "predicted"]).copy()
    if df.empty:
        return {}

    df["pred_direction"] = (df["predicted"] > threshold).astype(int)
    df["actual_direction"] = (df["return_1d"] > 0).astype(int)
    accuracy = (df["pred_direction"] == df["actual_direction"]).mean()

    # Long when predicted positive, flat otherwise
    df["strategy_return"] = df["return_1d"] * df["pred_direction"]

    std = df["strategy_return"].std()
    sharpe = (df["strategy_return"].mean() / std) * np.sqrt(252) if std > 0 else 0.0

    cumulative = (1 + df["strategy_return"]).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        "direction_accuracy": round(float(accuracy), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(float(max_drawdown), 4),
        "total_trades": int(df["pred_direction"].sum()),
        "annualized_return": round(float(df["strategy_return"].mean() * 252), 4),
    }


def naive_baseline_metrics(results_df: pd.DataFrame) -> dict:
    """'Market always goes up' baseline — what the model needs to beat."""
    df = results_df.dropna(subset=["return_1d"]).copy()
    df["predicted"] = 1.0  # always predict positive
    return compute_metrics(df)
