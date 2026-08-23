from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import autogluon_backtest as base
import autogluon_backtest_safe as safe

NATIVE_ROWS_PATH = Path(os.environ.get("NATIVE_CHRONOS_ROWS_PATH", "native_chronos_backtest_rows.csv"))
NATIVE_SUMMARY_PATH = Path(os.environ.get("NATIVE_CHRONOS_SUMMARY_PATH", "native_chronos_backtest_summary.csv"))
NATIVE_HORIZONS_PATH = Path(os.environ.get("NATIVE_CHRONOS_HORIZONS_PATH", "native_chronos_backtest_horizons.csv"))
NATIVE_RUNTIME_PATH = Path(os.environ.get("NATIVE_CHRONOS_RUNTIME_PATH", "native_chronos_backtest_runtime.csv"))

BATCH_SIZE = int(os.environ.get("NATIVE_CHRONOS_BATCH_SIZE", "32"))
CONTEXT_LENGTH = int(os.environ.get("NATIVE_CHRONOS_CONTEXT_LENGTH", "0")) or None
CROSS_LEARNING = os.environ.get("NATIVE_CHRONOS_CROSS_LEARNING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def load_pipeline() -> Chronos2Pipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading native Chronos-2 on {device}: {base.MODEL_PATH}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        base.MODEL_PATH,
        device_map=device,
    )
    return pipeline


def native_prediction_frame(
    pipeline: Chronos2Pipeline,
    context_frame: pd.DataFrame,
    future_frame: pd.DataFrame,
    targets: list[str],
    covariates: list[str],
) -> pd.DataFrame:
    """Run the direct production-style Chronos-2 dataframe API.

    This intentionally keeps the six flow outcomes as simultaneous target columns
    for one JGH item, matching the native production architecture rather than the
    AutoGluon representation where each target is a separate item_id.
    """
    context = context_frame[["ds", *targets, *covariates]].copy()
    context.insert(0, "id", "jgh")

    future = future_frame[["ds", *covariates]].copy()
    future.insert(0, "id", "jgh")

    prediction = pipeline.predict_df(
        context,
        future_df=future,
        id_column="id",
        timestamp_column="ds",
        target=targets,
        prediction_length=base.PREDICTION_LENGTH,
        quantile_levels=list(base.QUANTILES),
        batch_size=BATCH_SIZE,
        context_length=CONTEXT_LENGTH,
        cross_learning=CROSS_LEARNING,
        freq="h",
    )

    required = {"ds", "target_name", "predictions", "0.1", "0.5", "0.9"}
    missing = sorted(required.difference(prediction.columns))
    if missing:
        raise ValueError(
            "Native Chronos-2 predict_df output is missing required columns: "
            + ", ".join(missing)
        )

    return prediction.rename(
        columns={
            "predictions": "prediction",
            "0.1": "q10",
            "0.5": "q50",
            "0.9": "q90",
        }
    )[["ds", "target_name", "prediction", "q10", "q50", "q90"]]


def rolling_native_predictions(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    train_history: pd.DataFrame,
    targets: list[str],
    covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_holdout = len(train_history)
    rows: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []

    for window in range(base.HOLDOUT_WINDOWS):
        forecast_start_idx = first_holdout + window * base.PREDICTION_LENGTH
        forecast_end_idx = forecast_start_idx + base.PREDICTION_LENGTH
        if forecast_end_idx > len(history):
            break

        context_frame = history.iloc[:forecast_start_idx].copy()
        future_frame = history.iloc[forecast_start_idx:forecast_end_idx].copy()
        cutoff = context_frame["ds"].iloc[-1]

        actual_parts = []
        for target in targets:
            actual_parts.append(
                pd.DataFrame(
                    {
                        "ds": future_frame["ds"].to_numpy(),
                        "target_name": target,
                        "actual": future_frame[target].to_numpy(),
                        "observed": future_frame[f"__observed_{target}"].to_numpy(dtype=bool),
                    }
                )
            )
        actual = pd.concat(actual_parts, ignore_index=True)

        print(
            f"Native Chronos-2 window {window + 1}/{base.HOLDOUT_WINDOWS} "
            f"from cutoff {cutoff}"
        )
        started = time.perf_counter()
        forecast = native_prediction_frame(
            pipeline=pipeline,
            context_frame=context_frame,
            future_frame=future_frame,
            targets=targets,
            covariates=covariates,
        )
        elapsed = time.perf_counter() - started
        runtime_rows.append(
            {
                "window": window + 1,
                "cutoff": cutoff,
                "prediction_seconds": elapsed,
                "forecast_rows": len(forecast),
            }
        )
        print(f"  prediction runtime: {elapsed:.2f}s")

        merged = forecast.merge(actual, on=["ds", "target_name"], how="inner")
        merged = merged[merged["observed"]].copy()
        if merged.empty:
            print(f"Skipping native window {window + 1}: no observed outcomes")
            continue

        merged["model"] = "Chronos2Native"
        merged["window"] = window + 1
        merged["cutoff"] = cutoff
        merged["horizon_hours"] = (
            (merged["ds"] - pd.Timestamp(cutoff)).dt.total_seconds() / 3600
        ).astype(int)
        merged["error"] = merged["prediction"] - merged["actual"]
        merged["absolute_error"] = merged["error"].abs()
        merged["squared_error"] = merged["error"] ** 2
        rows.append(merged.drop(columns="observed"))

    if not rows:
        raise RuntimeError("No native Chronos-2 holdout predictions were produced")

    return pd.concat(rows, ignore_index=True), pd.DataFrame(runtime_rows)


def summarize(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["model", "target_name"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            n=("error", "size"),
        )
    )

    wql_rows = []
    for (model, target_name), group in rows.groupby(["model", "target_name"], sort=False):
        wql_rows.append(
            {
                "model": model,
                "target_name": target_name,
                "WQL": base.weighted_quantile_loss(group),
            }
        )
    summary = summary.merge(
        pd.DataFrame(wql_rows),
        on=["model", "target_name"],
        how="left",
    )

    horizons = rows[rows["horizon_hours"].isin(base.HORIZON_CHECKPOINTS)]
    horizon_summary = (
        horizons.groupby(["model", "target_name", "horizon_hours"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            n=("error", "size"),
        )
    )
    return summary, horizon_summary


def main() -> None:
    if safe.INCLUDE_WEATHER:
        raise ValueError(
            "Native production comparison must use AUTOGLUON_INCLUDE_WEATHER=0; "
            "weather.csv is not an as-of historical forecast archive."
        )

    print("Benchmark: native Chronos-2 production API")
    print(f"Flow targets: {', '.join(base.FLOW_TARGETS)}")
    print(
        "Configuration: "
        f"{base.HOLDOUT_WINDOWS} external {base.PREDICTION_LENGTH}-hour windows, "
        f"batch_size={BATCH_SIZE}, cross_learning={CROSS_LEARNING}"
    )

    history, targets, covariates, _ = safe.prepare_history()
    train_history, holdout_history = base.split_train_holdout(history)
    print(f"Known covariates ({len(covariates)}): {', '.join(covariates)}")
    print(
        f"Context ends {train_history['ds'].iloc[-1]}; "
        f"external holdout begins {holdout_history['ds'].iloc[0]}"
    )

    pipeline = load_pipeline()
    rows, runtime = rolling_native_predictions(
        pipeline=pipeline,
        history=history,
        train_history=train_history,
        targets=targets,
        covariates=covariates,
    )
    summary, horizons = summarize(rows)

    rows.to_csv(NATIVE_ROWS_PATH, index=False)
    summary.to_csv(NATIVE_SUMMARY_PATH, index=False)
    horizons.to_csv(NATIVE_HORIZONS_PATH, index=False)
    runtime.to_csv(NATIVE_RUNTIME_PATH, index=False)

    print(f"Saved native rows: {NATIVE_ROWS_PATH}")
    print(f"Saved native summary: {NATIVE_SUMMARY_PATH}")
    print(f"Saved native horizon summary: {NATIVE_HORIZONS_PATH}")
    print(f"Saved native runtimes: {NATIVE_RUNTIME_PATH}")
    print("\nNative Chronos-2 external holdout results:")
    print(summary.sort_values("target_name").to_string(index=False))
    print("\nNative runtime summary:")
    print(runtime["prediction_seconds"].describe().to_string())


if __name__ == "__main__":
    main()
