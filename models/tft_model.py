"""
Temporal Fusion Transformer definition and training loop.

Inputs: past OHLCV + technical indicators + sentiment scores + event flags
Output: predicted next-day return (with quantile intervals via QuantileLoss)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
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


def train_tft(
    dataset: TimeSeriesDataSet,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
) -> TemporalFusionTransformer:
    train_loader = dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=4)

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

    use_gpu = torch.cuda.is_available()
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[EarlyStopping(monitor="train_loss", patience=5, mode="min")],
        enable_progress_bar=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader)
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
