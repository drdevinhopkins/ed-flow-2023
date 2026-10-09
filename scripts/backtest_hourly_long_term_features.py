#!/usr/bin/env python3
"""Backtest explicit annual memory and secular growth with short Chronos-2 context.

The experiment asks whether 60-90 days of recent raw context performs better when
Chronos also receives leakage-safe information from the multi-year archive. The
8,192-hour model maximum remains a reference arm.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_history_windows as history_bt
from hourly_long_term_features import build_long_term_feature_frame, scenario_columns

FLOW_TARGETS = base.FLOW_TARGETS
MODEL_ID = base.MODEL_ID
DEFAULT_WINDOWS = [1440, 2160, history_bt.MODEL_MAX_CONTEXT]
SCENARIOS = [
    "baseline",
    "annual_fourier",
    "annual_memory",
    "secular_growth",
    "annual_plus_growth",
]


def window_label(hours: int) -> str:
    return history_bt.window_label(hours)


def scenario_frames(
    flow: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    history_hours: int,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    start = cutoff - pd.Timedelta(hours=history_hours - 1)
    history = flow.loc[
        flow["ds"].between(start, cutoff), ["ds", *FLOW_TARGETS]
    ].copy()
    if len(history) != history_hours:
        raise ValueError(
            f"Expected {history_hours} history rows at {cutoff}, got {len(history)}"
        )
    if history[FLOW_TARGETS].isna().any().any():
        bad = history[FLOW_TARGETS].columns[history[FLOW_TARGETS].isna().any()].tolist()
        raise ValueError(f"Missing target history at {cutoff}: {bad}")

    history["id"] = "jgh"
    if scenario == "baseline":
        return history[["id", "ds", *FLOW_TARGETS]], None

    future_ds = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    all_ds = pd.Series(pd.DatetimeIndex(history["ds"]).append(future_ds))
    features = build_long_term_feature_frame(
        flow,
        all_ds,
        FLOW_TARGETS,
        cutoff=cutoff,
    )
    columns = scenario_columns(scenario, FLOW_TARGETS)
    selected = features[["ds", *columns]].copy()

    hist_features = selected.loc[selected["ds"] <= cutoff]
    future_features = selected.loc[selected["ds"] > cutoff]
    history = history.merge(hist_features, on="ds", how="left")
    future = pd.DataFrame({"id": "jgh", "ds": future_ds}).merge(
        future_features, on="ds", how="left"
    )

    if history[columns].isna().any().any() or future[columns].isna().any().any():
        bad = sorted(
            set(history[columns].columns[history[columns].isna().any()].tolist())
            | set(future[columns].columns[future[columns].isna().any()].tolist())
        )
        raise ValueError(
            f"Missing long-term feature values at cutoff={cutoff}, "
            f"history={window_label(history_hours)}, scenario={scenario}: {bad}"
        )

    for column in columns:
        history[column] = pd.to_numeric(history[column], errors="raise").astype("float64")
        future[column] = pd.to_numeric(future[column], errors="raise").astype("float64")
    return history[["id", "ds", *FLOW_TARGETS, *columns]], future[["id", "ds", *columns]]


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    horizon: int,
    history_hours: int,
) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "prediction_length": horizon,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": FLOW_TARGETS,
        "quantile_levels": [0.5],
        "context_length": history_hours,
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].rename(
        columns={"predictions": "prediction"}
    )


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["abs_actual"] = out["actual"].abs()
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=1008)
    parser.add_argument("--cutoff-index", type=int)
    parser.add_argument("--windows", nargs="+", type=int, default=DEFAULT_WINDOWS)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-hourly-long-term"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = sorted(set(args.windows))
    if min(args.horizon, args.num_cutoffs, args.spacing_hours, *windows) < 1:
        raise ValueError("Backtest sizes and history windows must be positive")
    if max(windows) > history_bt.MODEL_MAX_CONTEXT:
        raise ValueError("History exceeds Chronos-2 maximum context")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow = base.load_flow()
    cutoffs = history_bt.select_cutoffs(
        flow,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        max_history_hours=max(windows),
    )
    if args.cutoff_index is not None:
        if not 0 <= args.cutoff_index < len(cutoffs):
            raise IndexError(
                f"cutoff-index {args.cutoff_index} outside 0..{len(cutoffs)-1}"
            )
        cutoffs = [cutoffs[args.cutoff_index]]

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Windows: {', '.join(window_label(w) for w in windows)}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Cutoffs: {cutoffs}")
    print(
        "Long-term features are leakage-safe: annual lags use prior calendar years; "
        "secular trend is computed as-of each row and frozen after the cutoff."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = base.actuals_long(flow, cutoff, args.horizon)
        for history_hours in windows:
            for scenario in args.scenarios:
                print(
                    f"Forecasting cutoff={cutoff} history={window_label(history_hours)} "
                    f"scenario={scenario}"
                )
                history, future = scenario_frames(
                    flow,
                    cutoff=cutoff,
                    horizon=args.horizon,
                    history_hours=history_hours,
                    scenario=scenario,
                )
                forecast = run_forecast(
                    pipeline,
                    history,
                    future,
                    horizon=args.horizon,
                    history_hours=history_hours,
                )
                joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
                joined["cutoff"] = cutoff
                joined["history_label"] = window_label(history_hours)
                joined["history_hours"] = history_hours
                joined["scenario"] = scenario
                joined["horizon_hour"] = (
                    (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
                ).astype(int)
                joined["horizon_band"] = pd.cut(
                    joined["horizon_hour"],
                    bins=[0, 4, 8, 12, 24],
                    labels=["h01_04", "h05_08", "h09_12", "h13_24"],
                    include_lowest=True,
                ).astype(str)
                frames.append(add_errors(joined))

    detail = pd.concat(frames, ignore_index=True).dropna(subset=["prediction", "actual"])
    detail.to_csv(args.output_dir / "detail.csv", index=False)

    quick = (
        detail.groupby(["target_name", "history_label", "scenario"], as_index=False)
        .agg(mae=("abs_error", "mean"))
        .sort_values(["target_name", "history_label", "mae"])
    )
    print("\n=== Per-job MAE ===")
    print(quick.to_string(index=False))


if __name__ == "__main__":
    main()
