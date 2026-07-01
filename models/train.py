"""
Single train/validation split entry point for a ticker.

Splits the feature matrix into a train window and a held-out external
validation window (val_months), trains the TFT — using a further internal
split of the training window for early stopping/checkpoint selection — then
evaluates 1-day-ahead predictions across every day in the external validation
window and logs everything to MLflow. Full rolling walk-forward evaluation
across many windows is Phase 5 (backtest/walkforward.py).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import shutil

import pandas as pd
import mlflow

from config import PROCESSED_FEATURES_DIR, MLFLOW_TRACKING_URI
from models.tft_model import create_train_val_datasets, train_tft
from models.evaluate import compute_metrics, naive_baseline_metrics, predict_window

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
    max_epochs: int = 50,
    internal_val_days: int = 60,
):
    df = load_features(ticker)
    df = df.sort_index()
    dates = df.index.to_list()

    val_start = df.index.max() - pd.DateOffset(months=val_months)
    train_idx_end = sum(1 for d in dates if d < val_start)
    train_df = df.iloc[:train_idx_end].reset_index(drop=True)

    if len(train_df) < max_encoder_length + max_prediction_length + 30:
        raise ValueError(f"{ticker}: not enough training rows ({len(train_df)}) for the requested window")

    full_df = df.reset_index(drop=True)
    full_df["time_idx"] = range(len(full_df))
    full_df["ticker"] = ticker
    full_df["is_earnings_day"] = full_df["is_earnings_day"].astype(str)

    training_dataset, validation_dataset = create_train_val_datasets(
        train_df, ticker, max_encoder_length, max_prediction_length, val_days=internal_val_days,
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
            "internal_val_days": internal_val_days,
            "train_rows": len(train_df),
        })

        tft, trainer, best_ckpt_path = train_tft(
            training_dataset, validation_dataset=validation_dataset,
            max_epochs=max_epochs, return_trainer=True,
        )

        # Evaluate across every day in the external validation window, not just the last 5
        results = predict_window(tft, training_dataset, full_df, train_idx_end, dates)

        metrics = compute_metrics(results)
        baseline = naive_baseline_metrics(results)

        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})

        print(f"[train] {ticker} model metrics:   {metrics}")
        print(f"[train] {ticker} baseline metrics: {baseline}")

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        safe_ticker = ticker.replace(".", "_")
        model_path = CHECKPOINT_DIR / f"{safe_ticker}_tft.ckpt"

        if best_ckpt_path:
            # Best-validation checkpoint already on disk (in a tempdir) — copy it into place
            shutil.copy(best_ckpt_path, model_path)
        else:
            # No internal validation was possible (too little data) — fall back to last epoch
            trainer.save_checkpoint(model_path)

        # Dataset parameters (encoders/scalers) — required to preprocess new inputs at inference time
        params_path = CHECKPOINT_DIR / f"{safe_ticker}_dataset_params.pkl"
        with open(params_path, "wb") as f:
            pickle.dump(training_dataset.get_parameters(), f)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(params_path))
        print(f"[train] Saved model → {model_path}")
        print(f"[train] Saved dataset params → {params_path}")

    return tft, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    train_ticker(args.ticker, max_epochs=args.epochs)
