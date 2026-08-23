#!/usr/bin/env python3
"""Rolling Chronos-2 ablation of engineered ED staffing covariates.

Scenarios isolate the incremental value of:

* the current representation (one categorical role column per physician + role counts),
* schedule structure (coverage, handoffs, shift phase, composition, continuity),
* leakage-safe physician flow fingerprints learned at each historical cutoff, and
* physician identity combined with the engineered features.

No named physician profiles are written to disk. Validation artifacts contain only
aggregate forecast accuracy by target/scenario/horizon.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from forecast_oncall_impact import derive_flow_metrics
from staffing_features import (
    StaffingFeatureFrames,
    build_effect_score_features,
    build_schedule_feature_frames,
    fit_physician_effect_profiles,
    parse_hour,
    sanitize_identity_for_cutoff,
)

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
SHIFT_URL = (
    "https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/"
    "all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1"
)
MODEL_ID = "amazon/chronos-2"
FLOW_TARGETS = [
    "Total_TBS",
    "POD_TBS",
    "Vertical_TBS",
    "TTStr",
    "Overflow",
    "WAITINGADM",
]
SCENARIOS = [
    "baseline",
    "current_staffing",
    "structure",
    "structure_effects",
    "structure_identity",
    "full",
]


def load_flow(url: str = FLOW_URL) -> pd.DataFrame:
    """Load and regularize the six operational flow targets, retaining inflow for profiling."""
    raw = pd.read_csv(url)
    raw["ds"] = parse_hour(raw["ds"])
    raw = raw.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    derived, _ = derive_flow_metrics(raw)
    source_map = {
        "Total_TBS": "total_tbs",
        "POD_TBS": "pod_tbs",
        "Vertical_TBS": "vertical_tbs",
        "Overflow": "overflow",
    }
    for target, source in source_map.items():
        if target not in derived.columns:
            if source not in derived.columns:
                raise ValueError(f"Could not derive required target {target} from {source}")
            derived[target] = pd.to_numeric(derived[source], errors="coerce")

    missing = [target for target in FLOW_TARGETS if target not in derived.columns]
    if missing:
        raise ValueError(f"Missing required flow target(s): {', '.join(missing)}")

    keep = ["ds", *FLOW_TARGETS]
    if "Inflow_Total" in derived.columns:
        keep.append("Inflow_Total")
    flow = derived[keep].copy()
    for column in keep[1:]:
        flow[column] = pd.to_numeric(flow[column], errors="coerce")

    index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    flow = flow.set_index("ds").reindex(index).reset_index()
    for column in keep[1:]:
        flow[column] = flow[column].ffill()
    return flow


def load_shifts(url: str = SHIFT_URL) -> pd.DataFrame:
    return pd.read_csv(url)


def select_cutoffs(
    flow: pd.DataFrame,
    frames: StaffingFeatureFrames,
    *,
    horizon: int,
    num_cutoffs: int,
    spacing_hours: int,
    min_history_hours: int,
) -> list[pd.Timestamp]:
    schedule_end = min(frames.current["ds"].max(), frames.structure["ds"].max())
    common_end = min(flow["ds"].max(), schedule_end) - pd.Timedelta(hours=horizon)
    common_start = max(flow["ds"].min(), frames.current["ds"].min()) + pd.Timedelta(
        hours=min_history_hours
    )
    if common_end < common_start:
        raise ValueError(f"No common staffing backtest period: {common_start} to {common_end}")

    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible staffing backtest cutoffs")
    return sorted(cutoffs)


def _window(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return frame.loc[frame["ds"].between(start, end)].copy()


def _merge_feature_frame(base: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in features.columns if c != "ds"]
    out = base.merge(features, on="ds", how="left")
    for column in columns:
        if column.startswith("physician__"):
            out[column] = out[column].fillna("NotWorking").astype(str)
        elif column == "oncall_physician_id":
            out[column] = out[column].fillna("None").astype(str)
        else:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out


def _sanitize_other_categories(
    history: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.copy()
    future = future.copy()
    for column in future.columns:
        if column in {"id", "ds"} or column not in history.columns:
            continue
        if pd.api.types.is_numeric_dtype(history[column]) or pd.api.types.is_numeric_dtype(
            future[column]
        ):
            continue
        history[column] = history[column].fillna("None").astype(str)
        future[column] = future[column].fillna("None").astype(str)
        seen = set(history[column])
        if not seen:
            continue
        fallback = "None" if "None" in seen else history[column].mode().iloc[0]
        future.loc[~future[column].isin(seen), column] = fallback
    return history, future


def _normalize_numeric(
    history: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.copy()
    future = future.copy()
    for column in future.columns:
        if column in {"id", "ds"} or column not in history.columns:
            continue
        if pd.api.types.is_numeric_dtype(history[column]) or pd.api.types.is_numeric_dtype(
            future[column]
        ):
            history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
            future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")
    return history, future


def scenario_frames(
    scenario: str,
    flow: pd.DataFrame,
    shifts: pd.DataFrame,
    schedule_frames: StaffingFeatureFrames,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
    effect_min_hours: int,
    effect_shrinkage_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    history_start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    history = _window(flow, history_start, cutoff)
    if history.empty:
        raise ValueError(f"No history at cutoff {cutoff}")
    if history[FLOW_TARGETS].isna().any().any():
        bad = history[FLOW_TARGETS].columns[history[FLOW_TARGETS].isna().any()].tolist()
        raise ValueError(f"Missing target history at cutoff {cutoff}: {bad}")

    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})

    if scenario == "baseline":
        history = history[["ds", *FLOW_TARGETS]].copy()
        history["id"] = "jgh"
        return history[["id", "ds", *FLOW_TARGETS]], None

    if scenario == "current_staffing":
        selected = schedule_frames.current
        history = _merge_feature_frame(history, selected)
        future = _merge_feature_frame(future, selected)

    elif scenario == "structure":
        history = _merge_feature_frame(history, schedule_frames.structure)
        future = _merge_feature_frame(future, schedule_frames.structure)

    elif scenario == "structure_identity":
        structure_identity = schedule_frames.structure.merge(
            schedule_frames.identity, on="ds", how="outer"
        )
        history = _merge_feature_frame(history, structure_identity)
        future = _merge_feature_frame(future, structure_identity)

    elif scenario in {"structure_effects", "full"}:
        profiles = fit_physician_effect_profiles(
            flow,
            shifts,
            FLOW_TARGETS,
            profile_end=cutoff,
            min_active_hours=effect_min_hours,
            shrinkage_hours=effect_shrinkage_hours,
        )
        effects = build_effect_score_features(shifts, profiles, FLOW_TARGETS)
        engineered = schedule_frames.structure.merge(effects, on="ds", how="outer")
        if scenario == "full":
            engineered = engineered.merge(schedule_frames.identity, on="ds", how="outer")
            oncall = schedule_frames.current[["ds", "oncall_physician_id"]]
            engineered = engineered.merge(oncall, on="ds", how="left")
        history = _merge_feature_frame(history, engineered)
        future = _merge_feature_frame(future, engineered)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    history = history.drop(columns=["Inflow_Total"], errors="ignore")
    history["id"] = "jgh"
    future["id"] = "jgh"

    history, future = sanitize_identity_for_cutoff(history, future)
    history, future = _sanitize_other_categories(history, future)
    history, future = _normalize_numeric(history, future)

    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    return (
        history[["id", "ds", *FLOW_TARGETS, *covariates]],
        future[["id", "ds", *covariates]],
    )


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
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].copy()


def actuals_long(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.loc[flow["ds"].isin(hours), ["ds", *FLOW_TARGETS]].copy()
    return actual.melt(id_vars="ds", var_name="target_name", value_name="actual")


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    summary = detail.groupby(["target_name", "scenario"], as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary["wape"] = summary["abs_error_sum"] / summary["abs_actual_sum"].replace(0, np.nan)
    summary = summary.drop(columns=["abs_error_sum", "abs_actual_sum"])
    baseline = summary.loc[summary["scenario"].eq("baseline"), ["target_name", "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    summary = summary.merge(baseline, on="target_name", how="left")
    summary["mae_improvement"] = summary["baseline_mae"] - summary["mae"]
    summary["mae_improvement_pct"] = (
        summary["mae_improvement"] / summary["baseline_mae"].replace(0, np.nan) * 100
    )
    return summary.sort_values(["target_name", "mae"]).reset_index(drop=True)


def summarize_horizon(detail: pd.DataFrame) -> pd.DataFrame:
    return (
        detail.groupby(["target_name", "scenario", "horizon_hour"], as_index=False)
        .agg(n=("abs_error", "size"), mae=("abs_error", "mean"), mean_error=("error", "mean"))
        .sort_values(["target_name", "scenario", "horizon_hour"])
        .reset_index(drop=True)
    )


def winners(summary: pd.DataFrame) -> pd.DataFrame:
    nonbaseline = summary.loc[~summary["scenario"].eq("baseline")].copy()
    idx = nonbaseline.groupby("target_name")["mae"].idxmin()
    result = nonbaseline.loc[idx].copy().sort_values("target_name")
    result["beats_baseline"] = result["mae_improvement"].gt(0)
    return result.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=6)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--min-history-hours", type=int, default=24 * 28)
    parser.add_argument("--effect-min-hours", type=int, default=24)
    parser.add_argument("--effect-shrinkage-hours", type=float, default=72.0)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--flow-url", default=FLOW_URL)
    parser.add_argument("--shift-url", default=SHIFT_URL)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = [
        args.horizon,
        args.num_cutoffs,
        args.spacing_hours,
        args.max_history_days,
        args.min_history_hours,
        args.effect_min_hours,
    ]
    if min(positive) < 1 or args.effect_shrinkage_hours < 0:
        raise ValueError("Backtest sizes must be positive and shrinkage must be non-negative")

    flow = load_flow(args.flow_url)
    shifts = load_shifts(args.shift_url)
    schedule_frames = build_schedule_feature_frames(shifts)
    cutoffs = select_cutoffs(
        flow,
        schedule_frames,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print(
        "Physician flow fingerprints are leakage-safe, cutoff-fitted predictive associations; "
        "they are not causal performance scores."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    records: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = actuals_long(flow, cutoff, args.horizon)
        if len(actual) != args.horizon * len(FLOW_TARGETS):
            raise ValueError(f"Incomplete actuals after cutoff {cutoff}")
        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future = scenario_frames(
                scenario,
                flow,
                shifts,
                schedule_frames,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
                effect_min_hours=args.effect_min_hours,
                effect_shrinkage_hours=args.effect_shrinkage_hours,
            )
            forecast = run_forecast(pipeline, history, future, horizon=args.horizon).rename(
                columns={"predictions": "prediction"}
            )
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            if len(joined) != len(actual):
                raise ValueError(
                    f"Forecast/actual mismatch cutoff={cutoff} scenario={scenario}: "
                    f"{len(joined)} vs {len(actual)}"
                )
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_hour"] = (
                (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
            ).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            records.append(joined)

    detail = pd.concat(records, ignore_index=True)
    summary = summarize(detail)
    by_horizon = summarize_horizon(detail)
    win = winners(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "staffing_feature_detail.csv", index=False)
    summary.to_csv(args.output_dir / "staffing_feature_summary.csv", index=False)
    by_horizon.to_csv(args.output_dir / "staffing_feature_by_horizon.csv", index=False)
    win.to_csv(args.output_dir / "staffing_feature_winners.csv", index=False)

    print("\n=== Staffing feature summary ===")
    print(
        summary[
            ["target_name", "scenario", "mae", "mae_improvement", "mae_improvement_pct", "wape"]
        ].to_string(index=False)
    )
    print("\n=== Best non-baseline scenario per target ===")
    print(
        win[
            ["target_name", "scenario", "mae", "mae_improvement_pct", "beats_baseline"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
