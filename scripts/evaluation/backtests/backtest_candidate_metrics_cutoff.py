#!/usr/bin/env python3
"""Run one common-cutoff Chronos-2 feature ablation for candidate ED flow targets.

The experiment forecasts the existing eight operational targets plus five candidate
metrics in the same 13-target Chronos-2 bundle, then retains only candidate-target rows
for scoring. Production routing is not modified by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_calendar_features as calendar_bt
import backtest_hourly_final_features as final_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from candidate_flow_metrics import (
    ALL_EXPERIMENT_TARGETS,
    CANDIDATE_TARGETS,
    build_experiment_flow,
    candidate_quality_summary,
)
from staffing_features import (
    build_effect_score_features,
    build_schedule_feature_frames,
    fit_physician_effect_profiles,
    sanitize_identity_for_cutoff,
)

TARGETS = list(ALL_EXPERIMENT_TARGETS)
SCENARIOS = list(final_bt.SCENARIOS)
FAMILY = dict(final_bt.FAMILY)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-index", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--min-history-hours", type=int, default=672)
    parser.add_argument("--effect-min-hours", type=int, default=24)
    parser.add_argument("--effect-shrinkage-hours", type=float, default=72.0)
    parser.add_argument("--model-id", default=final_bt.MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_flow() -> pd.DataFrame:
    raw = pd.read_csv(staffing_bt.FLOW_URL)
    return build_experiment_flow(raw)


def select_common_cutoffs(
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
    schedule_frames,
    *,
    horizon: int,
    num_cutoffs: int,
    spacing_hours: int,
    min_history_hours: int,
) -> list[pd.Timestamp]:
    schedule_start = max(schedule_frames.current["ds"].min(), schedule_frames.structure["ds"].min())
    schedule_end = min(schedule_frames.current["ds"].max(), schedule_frames.structure["ds"].max())
    common_start = max(
        flow["ds"].min(), staffing["ds"].min(), weather["ds"].min(), schedule_start
    ) + pd.Timedelta(hours=min_history_hours)
    common_end = min(
        flow["ds"].max(), staffing["ds"].max(), weather["ds"].max(), schedule_end
    ) - pd.Timedelta(hours=horizon)
    if common_end < common_start:
        raise ValueError(f"No common comparison window: {common_start} to {common_end}")

    indexed = flow.set_index("ds")
    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        future_hours = pd.date_range(current + pd.Timedelta(hours=1), periods=horizon, freq="h")
        future_actual = indexed.reindex(future_hours)[TARGETS]
        history_start = current - pd.Timedelta(days=365) + pd.Timedelta(hours=1)
        history = flow.loc[flow["ds"].between(history_start, current), TARGETS]
        if (
            len(future_actual) == horizon
            and not future_actual.isna().any().any()
            and not history.empty
            and not history.isna().any().any()
        ):
            cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible common cutoffs with complete 13-target history")
    return sorted(cutoffs)


def _staffing_frames(
    scenario: str,
    flow: pd.DataFrame,
    shifts: pd.DataFrame,
    schedule_frames,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
    effect_min_hours: int,
    effect_shrinkage_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    history_start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    history = flow.loc[flow["ds"].between(history_start, cutoff)].copy()
    if history.empty or history[TARGETS].isna().any().any():
        raise ValueError(f"Incomplete candidate target history at cutoff {cutoff}")

    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})
    if scenario == "baseline":
        history["id"] = "jgh"
        return history[["id", "ds", *TARGETS]], None

    if scenario == "staffing_current":
        selected = schedule_frames.current
        history = staffing_bt._merge_feature_frame(history, selected)
        future = staffing_bt._merge_feature_frame(future, selected)
    elif scenario == "staffing_structure_effects":
        profiles = fit_physician_effect_profiles(
            flow,
            shifts,
            TARGETS,
            profile_end=cutoff,
            min_active_hours=effect_min_hours,
            shrinkage_hours=effect_shrinkage_hours,
        )
        effects = build_effect_score_features(shifts, profiles, TARGETS)
        engineered = schedule_frames.structure.merge(effects, on="ds", how="outer")
        history = staffing_bt._merge_feature_frame(history, engineered)
        future = staffing_bt._merge_feature_frame(future, engineered)
    else:
        raise ValueError(f"Unsupported staffing scenario: {scenario}")

    history["id"] = "jgh"
    future["id"] = "jgh"
    history, future = sanitize_identity_for_cutoff(history, future)
    history, future = staffing_bt._sanitize_other_categories(history, future)
    history, future = staffing_bt._normalize_numeric(history, future)
    covariates = [
        column for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    return history[["id", "ds", *TARGETS, *covariates]], future[["id", "ds", *covariates]]


def scenario_frames(
    scenario: str,
    *,
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
    shifts: pd.DataFrame,
    schedule_frames,
    calendar: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
    effect_min_hours: int,
    effect_shrinkage_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    # Calendar and weather helpers reference module-level target lists. Patch only their
    # in-memory experiment globals; repository production routing remains untouched.
    base.FLOW_TARGETS = TARGETS
    calendar_bt.FLOW_TARGETS = TARGETS

    if scenario == "baseline":
        return _staffing_frames(
            scenario, flow, shifts, schedule_frames,
            cutoff=cutoff, horizon=horizon, max_history_days=max_history_days,
            effect_min_hours=effect_min_hours,
            effect_shrinkage_hours=effect_shrinkage_hours,
        )
    if scenario == "calendar_demand":
        return calendar_bt.scenario_frames(
            flow,
            calendar,
            scenario="demand_calendar",
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
        )
    if scenario == "weather_raw":
        history, future, _ = weather_bt.scenario_frames(
            "raw_weather", flow, staffing, weather,
            cutoff=cutoff, horizon=horizon, max_history_days=max_history_days,
        )
        return history, future
    if scenario == "weather_raw_plus_snow":
        history, future, _ = weather_bt.scenario_frames(
            "raw_plus_snow", flow, staffing, weather,
            cutoff=cutoff, horizon=horizon, max_history_days=max_history_days,
        )
        return history, future
    if scenario in {"staffing_current", "staffing_structure_effects"}:
        return _staffing_frames(
            scenario, flow, shifts, schedule_frames,
            cutoff=cutoff, horizon=horizon, max_history_days=max_history_days,
            effect_min_hours=effect_min_hours,
            effect_shrinkage_hours=effect_shrinkage_hours,
        )
    raise ValueError(f"Unknown scenario: {scenario}")


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
        "target": TARGETS,
        "quantile_levels": [0.5],
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].rename(columns={"predictions": "prediction"})


def actuals_long(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.set_index("ds").reindex(hours)[TARGETS]
    actual.index.name = "ds"
    return actual.reset_index().melt(id_vars="ds", var_name="target_name", value_name="actual")


def main() -> None:
    args = parse_args()
    if not 0 <= args.cutoff_index < args.num_cutoffs:
        raise ValueError(f"cutoff-index must be in 0..{args.num_cutoffs - 1}")

    flow = load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)

    cutoffs = select_common_cutoffs(
        flow, staffing, weather, schedule_frames,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )
    if len(cutoffs) != args.num_cutoffs:
        raise ValueError(f"Expected {args.num_cutoffs} cutoffs, got {len(cutoffs)}")
    cutoff = cutoffs[args.cutoff_index]
    calendar = final_bt.build_calendar_frame(flow, cutoffs, args.horizon)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cutoff_index == 0:
        candidate_quality_summary(flow).to_csv(args.output_dir / "candidate-quality.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cutoff index={args.cutoff_index}; cutoff={cutoff}; device={device}")
    print(f"Full experiment bundle: {', '.join(TARGETS)}")
    print(f"Candidate targets retained: {', '.join(CANDIDATE_TARGETS)}")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)

    actual = actuals_long(flow, cutoff, args.horizon)
    frames: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        print(f"Forecasting cutoff={cutoff} scenario={scenario}")
        history, future = scenario_frames(
            scenario,
            flow=flow,
            staffing=staffing,
            weather=weather,
            shifts=shifts,
            schedule_frames=schedule_frames,
            calendar=calendar,
            cutoff=cutoff,
            horizon=args.horizon,
            max_history_days=args.max_history_days,
            effect_min_hours=args.effect_min_hours,
            effect_shrinkage_hours=args.effect_shrinkage_hours,
        )
        forecast = run_forecast(pipeline, history, future, horizon=args.horizon)
        joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
        joined["cutoff"] = cutoff
        joined["scenario"] = scenario
        joined["family"] = FAMILY[scenario]
        joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
        frames.append(final_bt.add_errors(joined))

    detail = pd.concat(frames, ignore_index=True)
    detail = detail.loc[detail["target_name"].isin(CANDIDATE_TARGETS)].copy()
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["h01_04", "h05_08", "h09_12", "h13_24"],
        include_lowest=True,
    ).astype(str)

    expected = len(CANDIDATE_TARGETS) * len(SCENARIOS) * args.horizon
    if len(detail) != expected:
        raise RuntimeError(f"Expected {expected} retained rows, got {len(detail)}")

    detail.to_csv(args.output_dir / f"detail-{args.cutoff_index}.csv", index=False)
    pd.DataFrame({"cutoff_index": [args.cutoff_index], "cutoff": [cutoff]}).to_csv(
        args.output_dir / f"cutoff-{args.cutoff_index}.csv", index=False
    )
    print(f"Saved {len(detail)} candidate rows for cutoff index {args.cutoff_index}")


if __name__ == "__main__":
    main()
