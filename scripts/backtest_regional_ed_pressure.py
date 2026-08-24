#!/usr/bin/env python3
"""Leakage-safe native Chronos-2 ablation of Montréal peer-ED pressure.

Regional pressure is an observed state, not a known-future input.  Every historical
forecast therefore sees regional observations only through its cutoff and persists the
cutoff state/trend vector through the 24-hour horizon.  Realized future peer pressure is
never supplied to Chronos.

This experiment intentionally starts as a family-specific ablation.  Once sufficient
prospective history exists and a representation wins, it can be brought onto the common
hourly feature-routing base for target × horizon comparison against calendar/staffing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_staffing_features as staffing_bt
from regional_ed_pressure import (
    REGIONAL_FEATURE_COLUMNS,
    REGIONAL_STATE_COLUMNS,
    persistence_future,
)

FLOW_TARGETS = staffing_bt.FLOW_TARGETS
MODEL_ID = staffing_bt.MODEL_ID
SERIES_ID = "jgh"
SCENARIOS = ["baseline", "regional_state", "regional_state_trends"]


def load_regional_history(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "ds" not in frame:
        raise ValueError("Regional pressure history requires ds")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame = frame.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")
    for column in REGIONAL_FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame[["ds", *REGIONAL_FEATURE_COLUMNS]].reset_index(drop=True)


def select_cutoffs(
    flow: pd.DataFrame,
    regional: pd.DataFrame,
    *,
    horizon: int,
    num_cutoffs: int,
    spacing_hours: int,
    min_history_hours: int,
) -> list[pd.Timestamp]:
    common_start = max(flow["ds"].min(), regional["ds"].min()) + pd.Timedelta(hours=min_history_hours)
    common_end = min(flow["ds"].max(), regional["ds"].max()) - pd.Timedelta(hours=horizon)
    if common_end < common_start:
        observed = (regional["ds"].max() - regional["ds"].min()) / pd.Timedelta(hours=1)
        raise ValueError(
            f"Insufficient regional-pressure archive for backtest: ~{observed:.0f}h observed; "
            f"need at least {min_history_hours + horizon}h before scoring"
        )

    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        future_hours = pd.date_range(current + pd.Timedelta(hours=1), periods=horizon, freq="h")
        actual = flow.set_index("ds").reindex(future_hours)[FLOW_TARGETS]
        cutoff_state = regional.loc[regional["ds"].le(current)].tail(1)
        if len(actual) == horizon and not actual.isna().any().any() and not cutoff_state.empty:
            cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible regional-pressure backtest cutoffs")
    return sorted(cutoffs)


def scenario_frames(
    flow: pd.DataFrame,
    regional: pd.DataFrame,
    *,
    scenario: str,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    start = max(
        flow["ds"].min(),
        regional["ds"].min(),
        cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1),
    )
    history = flow.loc[flow["ds"].between(start, cutoff), ["ds", *FLOW_TARGETS]].copy()
    if history.empty:
        raise ValueError(f"No target history through cutoff {cutoff}")
    history["id"] = SERIES_ID
    if scenario == "baseline":
        return history[["id", "ds", *FLOW_TARGETS]], None

    columns = REGIONAL_STATE_COLUMNS if scenario == "regional_state" else REGIONAL_FEATURE_COLUMNS
    regional_history = regional.loc[regional["ds"].between(start, cutoff), ["ds", *columns]].copy()
    history = history.merge(regional_history, on="ds", how="left")
    for column in columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").ffill()
        fallback = history[column].median(skipna=True)
        history[column] = history[column].fillna(0.0 if not np.isfinite(fallback) else float(fallback))

    future_all = persistence_future(regional, cutoff=cutoff, horizon=horizon)
    future = future_all[["ds", *columns]].copy()
    for column in columns:
        future[column] = pd.to_numeric(future[column], errors="coerce")
        if future[column].isna().any():
            fallback = history[column].iloc[-1] if not history[column].empty else 0.0
            future[column] = future[column].fillna(float(fallback))
    future["id"] = SERIES_ID
    return history[["id", "ds", *FLOW_TARGETS, *columns]], future[["id", "ds", *columns]]


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    horizon: int,
) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "prediction_length": horizon,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": FLOW_TARGETS,
        "quantile_levels": [0.5],
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    return result[["ds", "target_name", "predictions"]].rename(columns={"predictions": "prediction"})


def metric_table(detail: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = detail.groupby(keys, as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)
    baseline = table.loc[table["scenario"].eq("baseline"), [*keys[:-1], "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    table = table.merge(baseline, on=keys[:-1], how="left")
    table["mae_improvement_pct"] = (
        (table["baseline_mae"] - table["mae"]) / table["baseline_mae"].replace(0, np.nan) * 100
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-csv", required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=72)
    parser.add_argument("--min-history-hours", type=int, default=24 * 28)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-regional-pressure"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow = staffing_bt.load_flow()
    regional = load_regional_history(args.regional_csv)
    cutoffs = select_cutoffs(
        flow,
        regional,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "cutoffs.csv", index=False)
    print(f"Regional archive: {regional['ds'].min()}..{regional['ds'].max()} ({len(regional)} rows)")
    print(f"Cutoffs: {cutoffs}")
    print("Future regional pressure policy: persist cutoff state/trends; realized future values are never used.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)
    frames: list[pd.DataFrame] = []
    flow_index = flow.set_index("ds")
    for cutoff in cutoffs:
        actual = flow_index.reindex(
            pd.date_range(cutoff + pd.Timedelta(hours=1), periods=args.horizon, freq="h")
        )[FLOW_TARGETS].reset_index().rename(columns={"index": "ds"})
        actual = actual.melt(id_vars="ds", var_name="target_name", value_name="actual")
        for scenario in SCENARIOS:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future = scenario_frames(
                flow,
                regional,
                scenario=scenario,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
            )
            forecast = run_forecast(pipeline, history, future, horizon=args.horizon)
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
            joined["horizon_band"] = pd.cut(
                joined["horizon_hour"], [0, 4, 8, 12, 24],
                labels=["h01_04", "h05_08", "h09_12", "h13_24"], include_lowest=True,
            ).astype(str)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary = metric_table(detail, ["target_name", "scenario"]).sort_values(["target_name", "mae"])
    by_band = metric_table(detail, ["target_name", "horizon_band", "scenario"]).sort_values(
        ["target_name", "horizon_band", "mae"]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    by_band.to_csv(args.output_dir / "by_horizon_band.csv", index=False)
    print("\n=== Regional pressure summary ===")
    print(summary.to_string(index=False))
    print("\n=== By horizon band ===")
    print(by_band.to_string(index=False))


if __name__ == "__main__":
    main()
