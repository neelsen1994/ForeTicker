"""
Walk-forward validation for the TFT model.

Rolls a training window forward in time, retraining a fresh TFT each step
and evaluating 1-day-ahead predictions on the following test window.
NO data from a test window ever touches that window's training — this is
the key invariant that prevents look-ahead bias.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch
import mlflow
from pytorch_forecasting import TimeSeriesDataSet

from config import PROCESSED_FEATURES_DIR, MLFLOW_TRACKING_URI, DEFAULT_TICKERS
from models.tft_model import create_tft_dataset, train_tft
from models.evaluate import compute_metrics, naive_baseline_metrics

USE_GPU = torch.cuda.is_available()


def walk_forward_evaluate(
    df: pd.DataFrame,
    ticker: str,
    train_months: int = 18,
    test_months: int = 3,
    max_encoder_length: int = 60,
    max_prediction_length: int = 5,
    max_epochs: int = 15,
) -> pd.DataFrame:
    """
    Rolls a training window forward in time. Returns a DataFrame of
    1-day-ahead predictions vs actuals across all test windows, indexed by date.
    """
    df = df.sort_index().copy()
    dates = df.index.to_list()

    df = df.reset_index(drop=True)
    df["time_idx"] = range(len(df))
    df["ticker"] = ticker
    df["is_earnings_day"] = df["is_earnings_day"].astype(str)

    start = dates[0]
    end = dates[-1]
    current = start + pd.DateOffset(months=train_months)

    all_results = []
    window_num = 0

    while current + pd.DateOffset(months=test_months) <= end:
        test_end = current + pd.DateOffset(months=test_months)

        train_idx_end = sum(1 for d in dates if d < current)
        test_idx_end = sum(1 for d in dates if d < test_end)

        train_df = df.iloc[:train_idx_end]
        context_df = df.iloc[:test_idx_end]  # train history + test window (encoder needs lookback)

        min_train_rows = max_encoder_length + max_prediction_length + 30
        if len(train_df) < min_train_rows:
            current += pd.DateOffset(months=test_months)
            continue

        window_num += 1
        print(f"[walkforward] {ticker} window {window_num}: "
              f"train<{current.date()} test=[{current.date()},{test_end.date()})")

        training_dataset = create_tft_dataset(
            train_df.drop(columns=["time_idx", "ticker"]),
            ticker, max_encoder_length, max_prediction_length,
        )
        tft = train_tft(training_dataset, max_epochs=max_epochs)

        pred_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset, context_df, predict=False, stop_randomization=True,
            min_prediction_idx=train_idx_end,
        )
        loader = pred_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

        result = tft.predict(
            loader, mode="prediction", return_index=True,
            trainer_kwargs={"accelerator": "gpu" if USE_GPU else "cpu", "devices": 1},
        )

        preds_1step = result.output[:, 0].cpu().numpy()
        time_idxs = result.index["time_idx"].values
        window_dates = [dates[i] for i in time_idxs]
        actuals = df.set_index("time_idx").loc[time_idxs, "return_1d"].values

        window_df = pd.DataFrame({
            "date": window_dates,
            "return_1d": actuals,
            "predicted": preds_1step,
        })
        all_results.append(window_df)

        current += pd.DateOffset(months=test_months)

    if not all_results:
        raise ValueError(f"{ticker}: no walk-forward windows produced — check date range / window sizes")

    results = pd.concat(all_results, ignore_index=True).set_index("date").sort_index()
    return results


def run_ticker(
    ticker: str,
    train_months: int = 18,
    test_months: int = 3,
    max_epochs: int = 15,
) -> dict:
    path = PROCESSED_FEATURES_DIR / f"{ticker.replace('.', '_')}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No feature matrix for {ticker} — run features/builder.py first")

    df = pd.read_parquet(path)
    results = walk_forward_evaluate(df, ticker, train_months, test_months, max_epochs=max_epochs)

    metrics = compute_metrics(results)
    baseline = naive_baseline_metrics(results)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("foreticker_walkforward")
    with mlflow.start_run(run_name=f"{ticker}_walkforward"):
        mlflow.log_params({
            "ticker": ticker,
            "train_months": train_months,
            "test_months": test_months,
            "max_epochs": max_epochs,
            "test_rows": len(results),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline.items()})

    print(f"[walkforward] {ticker} model:    {metrics}")
    print(f"[walkforward] {ticker} baseline: {baseline}")
    return {"ticker": ticker, "metrics": metrics, "baseline": baseline}


def run_all(tickers: list[str], max_epochs: int = 15) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            results.append(run_ticker(ticker, max_epochs=max_epochs))
        except Exception as e:
            print(f"[walkforward] ERROR {ticker}: {e}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default=None, help="single ticker, or omit for all DEFAULT_TICKERS")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    if args.ticker:
        run_ticker(args.ticker, max_epochs=args.epochs)
    else:
        run_all(DEFAULT_TICKERS, max_epochs=args.epochs)
