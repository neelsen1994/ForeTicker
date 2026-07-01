"""
Temporal Fusion Transformer definition and training loop.

Inputs: past OHLCV + technical indicators + sentiment scores + event flags
Output: predicted next-day return (with quantile intervals via QuantileLoss)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile

import torch
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

# Features known only up to "now" — model must infer their future values
TIME_VARYING_UNKNOWN_REALS = [
    "return_1d",
    "rsi_14",
    "macd",
    "macd_diff",
    "stoch_k",
    "atr_14",
    "bb_pct",
    "sentiment_mean",
    "article_count",
    "bullish_ratio",
    "bearish_ratio",
]

# Features whose future values ARE known in advance (scheduled events)
TIME_VARYING_KNOWN_CATEGORICALS = ["is_earnings_day"]


def create_tft_dataset(
    df: pd.DataFrame,
    ticker: str,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
) -> TimeSeriesDataSet:
    df = df.copy().reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["ticker"] = ticker
    df["is_earnings_day"] = df["is_earnings_day"].astype(str)  # categorical needs str/category dtype

    dataset = TimeSeriesDataSet(
        df[: -max_prediction_length],
        time_idx="time_idx",
        target="return_1d",
        group_ids=["ticker"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return dataset


def create_train_val_datasets(
    df: pd.DataFrame,
    ticker: str,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    val_days: int = 60,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet | None]:
    """
    Splits a training window into a fit slice and a held-out internal
    validation slice (the most recent `val_days`), so EarlyStopping and the
    TFT's built-in ReduceLROnPlateau can monitor real generalization (val_loss)
    instead of train_loss. Falls back to no validation if there isn't enough
    data to spare.
    """
    df = df.copy().reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["ticker"] = ticker
    df["is_earnings_day"] = df["is_earnings_day"].astype(str)

    n = len(df)
    min_fit_rows = max_encoder_length + max_prediction_length + 30
    val_days = max(0, min(val_days, n - min_fit_rows - max_prediction_length))
    training_cutoff = n - val_days - max_prediction_length

    training_dataset = TimeSeriesDataSet(
        df[df["time_idx"] < training_cutoff],
        time_idx="time_idx",
        target="return_1d",
        group_ids=["ticker"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=TIME_VARYING_UNKNOWN_REALS,
        time_varying_known_categoricals=TIME_VARYING_KNOWN_CATEGORICALS,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation_dataset = None
    if val_days > 0:
        validation_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, df, predict=False, stop_randomization=True,
            min_prediction_idx=training_cutoff,
        )

    return training_dataset, validation_dataset


def train_tft(
    dataset: TimeSeriesDataSet,
    validation_dataset: TimeSeriesDataSet | None = None,
    max_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1.3e-4,  # found via Lightning's lr_find — 1e-3 converged too fast to a mediocre minimum
    patience: int = 8,
    return_trainer: bool = False,
):
    """
    Trains a TFT on `dataset`. If `validation_dataset` is given, EarlyStopping
    and the TFT's built-in ReduceLROnPlateau monitor real val_loss, and the
    best-validation checkpoint (not just the last epoch) is loaded and returned
    — without a validation set, both quietly degrade to monitoring train_loss,
    which rarely plateaus early and never reflects generalization.
    """
    train_loader = dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_loader = (
        validation_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
        if validation_dataset is not None else None
    )

    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=learning_rate,
        hidden_size=64,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(),
        log_interval=10,
    )

    monitor_metric = "val_loss" if val_loader is not None else "train_loss"
    callbacks = [EarlyStopping(monitor=monitor_metric, patience=patience, mode="min")]

    checkpoint_callback = None
    if val_loader is not None:
        # mkdtemp (not TemporaryDirectory) — must outlive this function so the
        # caller can read/copy best_ckpt_path; OS reclaims /tmp on its own schedule.
        ckpt_tmpdir = tempfile.mkdtemp(prefix="foreticker_tft_")
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_tmpdir, filename="best", monitor="val_loss", mode="min", save_top_k=1,
        )
        callbacks.append(checkpoint_callback)

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=callbacks,
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_ckpt_path = None
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        best_ckpt_path = checkpoint_callback.best_model_path
        tft = TemporalFusionTransformer.load_from_checkpoint(best_ckpt_path)

    if return_trainer:
        return tft, trainer, best_ckpt_path
    return tft


if __name__ == "__main__":
    from config import PROCESSED_FEATURES_DIR

    ticker = "AAPL"
    path = PROCESSED_FEATURES_DIR / f"{ticker}_features.parquet"
    df = pd.read_parquet(path)

    dataset = create_tft_dataset(df, ticker)
    print(f"[tft_model] Dataset built: {len(dataset)} samples")

    tft = train_tft(dataset, max_epochs=3)  # smoke test — short run
    print("[tft_model] Training smoke test complete")
