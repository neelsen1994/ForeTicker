"""
Forward-looking inference: predict the next N trading days' return for a
ticker using its saved TFT checkpoint, given all data available today.

Unlike backtest/walkforward.py (which evaluates on already-known days),
this appends synthetic future rows beyond the last known date so the model
can actually forecast forward. Time-varying *unknown* reals (technical
indicators, sentiment) are never fed into the decoder by TFT — only
time-varying *known* categoricals (is_earnings_day) and the encoder history
matter for future steps, so the placeholder values just need to exist
(non-NaN) without being meaningful.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle

import pandas as pd
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

from config import PROCESSED_FEATURES_DIR

CHECKPOINT_DIR = Path("models/checkpoints")
USE_GPU = torch.cuda.is_available()


def _load_checkpoint(ticker: str):
    safe_ticker = ticker.replace(".", "_")
    model_path = CHECKPOINT_DIR / f"{safe_ticker}_tft.ckpt"
    params_path = CHECKPOINT_DIR / f"{safe_ticker}_dataset_params.pkl"

    if not model_path.exists() or not params_path.exists():
        raise FileNotFoundError(
            f"No trained checkpoint for {ticker} — run models/train.py --ticker {ticker} first"
        )

    tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
    with open(params_path, "rb") as f:
        dataset_params = pickle.load(f)
    return tft, dataset_params


def _build_future_rows(df: pd.DataFrame, ticker: str, n_days: int) -> pd.DataFrame:
    """Synthetic placeholder rows for the next n_days business days."""
    future_dates = pd.bdate_range(start=df.index.max() + pd.Timedelta(days=1), periods=n_days)

    last_row = df.iloc[-1]
    future = pd.DataFrame([last_row.to_dict()] * n_days, index=future_dates)

    # Earnings flag, if a cached calendar exists — known in advance, safe to use
    try:
        from features.events import fetch_earnings_calendar
        earnings = fetch_earnings_calendar(ticker)
        earnings_dates = set(pd.to_datetime(earnings["date"]).dt.normalize()) if not earnings.empty else set()
        future["is_earnings_day"] = [1 if d.normalize() in earnings_dates else 0 for d in future_dates]
    except Exception:
        future["is_earnings_day"] = 0

    future["return_1d"] = 0.0  # unknown — placeholder, not used as decoder input
    return future


def predict_next(ticker: str, n_days: int | None = None) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by future business date with column
    `predicted_return_1d` — the model's forecast for that day's return.
    """
    tft, dataset_params = _load_checkpoint(ticker)
    max_encoder_length = dataset_params["max_encoder_length"]
    max_prediction_length = dataset_params["max_prediction_length"]
    n_days = min(n_days or max_prediction_length, max_prediction_length)

    feat_path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    if not feat_path.exists():
        raise FileNotFoundError(f"No feature matrix for {ticker} — run features/builder.py first")

    df = pd.read_parquet(feat_path).sort_index()
    df.index = pd.to_datetime(df.index)

    history = df.tail(max_encoder_length).copy()
    future = _build_future_rows(df, ticker, max_prediction_length)

    context = pd.concat([history, future])
    context = context.reset_index(drop=True)
    context["time_idx"] = range(len(context))
    context["ticker"] = ticker
    context["is_earnings_day"] = context["is_earnings_day"].astype(str)

    pred_dataset = TimeSeriesDataSet.from_parameters(
        dataset_params, context, predict=True, stop_randomization=True
    )
    loader = pred_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    result = tft.predict(
        loader, mode="prediction", return_index=True,
        trainer_kwargs={"accelerator": "gpu" if USE_GPU else "cpu", "devices": 1},
    )

    preds = result.output[0].cpu().numpy()[:n_days]
    future_dates = future.index[:n_days]

    return pd.DataFrame({"predicted_return_1d": preds}, index=future_dates)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    forecast = predict_next(args.ticker, args.days)
    print(forecast)
