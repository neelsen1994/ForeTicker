"""
Walk-forward training entry point for a single ticker (Phase 4 scope).

Splits the feature matrix into a train window and a held-out validation
window, trains the TFT, evaluates on validation, and logs everything to
MLflow. Full rolling walk-forward evaluation across many windows is
Phase 5 (backtest/walkforward.py) — this is the single-split version used
to confirm the model trains end-to-end and produces sane metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import mlflow
import torch

from config import PROCESSED_FEATURES_DIR, MLFLOW_TRACKING_URI
from models.tft_model import create_tft_dataset, train_tft, TIME_VARYING_UNKNOWN_REALS
from models.evaluate import compute_metrics, naive_baseline_metrics

CHECKPOINT_DIR = Path("models/checkpoints")


def load_features(ticker: str) -> pd.DataFrame:
    path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No feature matrix for {ticker} — run features/builder.py first")
    return pd.read_parquet(path)


def train_ticker(
    ticker: str,
    val_months: int = 3,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    max_epochs: int = 30,
):
    df = load_features(ticker)
    df = df.sort_index()

    val_start = df.index.max() - pd.DateOffset(months=val_months)
    train_df = df[df.index < val_start].reset_index(drop=True)
    full_df = df.reset_index(drop=True)

    if len(train_df) < max_encoder_length + max_prediction_length + 30:
        raise ValueError(f"{ticker}: not enough training rows ({len(train_df)}) for the requested window")

    train_df["time_idx"] = range(len(train_df))
    train_df["ticker"] = ticker
    train_df["is_earnings_day"] = train_df["is_earnings_day"].astype(str)

    full_df["time_idx"] = range(len(full_df))
    full_df["ticker"] = ticker
    full_df["is_earnings_day"] = full_df["is_earnings_day"].astype(str)

    training_dataset = create_tft_dataset(
        train_df, ticker, max_encoder_length, max_prediction_length
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("foreticker_tft")

    with mlflow.start_run(run_name=f"{ticker}_tft"):
        mlflow.log_params({
            "ticker": ticker,
            "val_months": val_months,
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "max_epochs": max_epochs,
            "train_rows": len(train_df),
        })

        tft = train_tft(training_dataset, max_epochs=max_epochs)

        # Build validation dataset that reuses training's normalization/encoders
        from pytorch_forecasting import TimeSeriesDataSet
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, full_df, predict=True, stop_randomization=True
        )
        val_loader = validation_dataset.to_dataloader(train=False, batch_size=64, num_workers=4)

        raw_predictions = tft.predict(
            val_loader,
            mode="prediction",
            trainer_kwargs={
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1,
            },
        )
        preds = raw_predictions.cpu().numpy().flatten()

        val_actuals = full_df[full_df.index.isin(
            full_df.index[-len(preds):]
        )].tail(len(preds))

        results = pd.DataFrame({
            "return_1d": val_actuals["return_1d"].values,
            "predicted": preds,
        })

        metrics = compute_metrics(results)
        baseline = naive_baseline_metrics(results)

        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})

        print(f"[train] {ticker} model metrics:   {metrics}")
        print(f"[train] {ticker} baseline metrics: {baseline}")

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = CHECKPOINT_DIR / f"{ticker.replace('.', '_')}_tft.ckpt"
        torch.save(tft.state_dict(), ckpt_path)
        mlflow.log_artifact(str(ckpt_path))
        print(f"[train] Saved checkpoint → {ckpt_path}")

    return tft, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    train_ticker(args.ticker, max_epochs=args.epochs)
