"""
Signal-based evaluation metrics — more meaningful than RMSE for trading.
Shared by models/train.py (single train/val split) and backtest/walkforward.py (Phase 5).
"""

import numpy as np
import pandas as pd
import torch


def predict_window(
    tft,
    training_dataset,
    context_df: pd.DataFrame,
    min_prediction_idx: int,
    dates: list,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Predicts 1-day-ahead returns for every day in context_df whose time_idx
    is >= min_prediction_idx (i.e. every day in a held-out window), not just
    the dataframe's final max_prediction_length rows. Returns a DataFrame
    indexed by date with columns [return_1d, predicted].
    """
    from pytorch_forecasting import TimeSeriesDataSet

    pred_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, context_df, predict=False, stop_randomization=True,
        min_prediction_idx=min_prediction_idx,
    )
    loader = pred_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    use_gpu = torch.cuda.is_available()
    result = tft.predict(
        loader, mode="prediction", return_index=True,
        trainer_kwargs={"accelerator": "gpu" if use_gpu else "cpu", "devices": 1},
    )

    preds_1step = result.output[:, 0].cpu().numpy()
    time_idxs = result.index["time_idx"].values
    window_dates = [dates[i] for i in time_idxs]
    actuals = context_df.set_index("time_idx").loc[time_idxs, "return_1d"].values

    return pd.DataFrame(
        {"return_1d": actuals, "predicted": preds_1step}, index=pd.Index(window_dates, name="date")
    ).sort_index()


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
