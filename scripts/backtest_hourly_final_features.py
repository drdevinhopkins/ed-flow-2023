#!/usr/bin/env python3
"""Apples-to-apples native Chronos-2 comparison of hourly feature-family finalists.

All scenarios use the same rolling cutoffs, target history, 24-hour forecast horizon,
and Chronos-2 model. Earlier family-specific ablations narrowed the candidates to:

* calendar: demand/system closure calendar (not the JGH mismatch interactions),
* weather: raw hourly weather and raw weather + snow/recovery state,
* staffing: the current representation and engineered structure + leakage-safe
  physician/role flow fingerprints.

The goal is to choose features per operational target and forecast horizon rather than
forcing one global covariate set across all six targets.

Historical weather remains a stitched/revised series rather than archived forecast-time
snapshots, so weather results are signal-potential estimates and should retain that caveat.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_calendar_features as calendar_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from hourly_calendar_features import add_hourly_calendar_features
from staffing_features import build_schedule_feature_frames

FLOW_TARGETS = base.FLOW_TARGETS
MODEL_ID = base.MODEL_ID
SERIES_ID = "jgh"

SCENARIOS = [
    "baseline",
    "calendar_demand",
    "weather_raw",
    "weather_raw_plus_snow",
    "staffing_current",
    "staffing_structure_effects",
]

FAMILY = {
    "baseline": "baseline",
    "calendar_demand": "calendar",
    "weather_raw": "weather",
    "weather_raw_plus_snow": "weather",
    "staffing_current": "staffing",
    "staffing_structure_effects": "staffing",
}


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
    schedule_start = max(
        schedule_frames.current["ds"].min(),
        schedule_frames.structure["ds"].min(),
    )
    schedule_end = min(
        schedule_frames.current["ds"].max(),
        schedule_frames.structure["ds"].max(),
    )
    common_start = max(
        flow["ds"].min(), staffing["ds"].min(), weather["ds"].min(), schedule_start
    ) + pd.Timedelta(hours=min_history_hours)
    common_end = min(
        flow["ds"].max(), staffing["ds"].max(), weather["ds"].max(), schedule_end
    ) - pd.Timedelta(hours=horizon)
    if common_end < common_start:
        raise ValueError(f"No common comparison window: {common_start} to {common_end}")

    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        hours = pd.date_range(current + pd.Timedelta(hours=1), periods=horizon, freq="h")
        actual = flow.set_index("ds").reindex(hours)[FLOW_TARGETS]
        if len(actual) == horizon and not actual.isna().any().any():
            cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible common cutoffs")
    return sorted(cutoffs)


def build_calendar_frame(flow: pd.DataFrame, cutoffs: list[pd.Timestamp], horizon: int) -> pd.DataFrame:
    start = flow["ds"].min().floor("h")
    end = max(cutoffs) + pd.Timedelta(hours=horizon)
    hours = pd.date_range(start, end, freq="h")
    return add_hourly_calendar_features(pd.DataFrame({"ds": hours}))


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
    if scenario == "baseline":
        return staffing_bt.scenario_frames(
            "baseline",
            flow,
            shifts,
            schedule_frames,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
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
            "raw_weather",
            flow,
            staffing,
            weather,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
        )
        return history, future
    if scenario == "weather_raw_plus_snow":
        history, future, _ = weather_bt.scenario_frames(
            "raw_plus_snow",
            flow,
            staffing,
            weather,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
        )
        return history, future
    if scenario == "staffing_current":
        return staffing_bt.scenario_frames(
            "current_staffing",
            flow,
            shifts,
            schedule_frames,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
            effect_min_hours=effect_min_hours,
            effect_shrinkage_hours=effect_shrinkage_hours,
        )
    if scenario == "staffing_structure_effects":
        return staffing_bt.scenario_frames(
            "structure_effects",
            flow,
            shifts,
            schedule_frames,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
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


def metrics(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = detail.groupby(group_cols, as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)
    return table


def add_baseline_comparison(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    baseline = table.loc[table["scenario"].eq("baseline"), [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    out = table.merge(baseline, on=keys, how="left")
    out["mae_improvement"] = out["baseline_mae"] - out["mae"]
    out["mae_improvement_pct"] = (
        out["mae_improvement"] / out["baseline_mae"].replace(0, np.nan) * 100
    )
    out["beats_baseline"] = out["mae_improvement"].gt(0)
    out["family"] = out["scenario"].map(FAMILY)
    return out


def winners(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = table.groupby(keys, observed=True)["mae"].idxmin()
    return table.loc[idx].sort_values(keys).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--min-history-hours", type=int, default=24 * 28)
    parser.add_argument("--effect-min-hours", type=int, default=24)
    parser.add_argument("--effect-shrinkage-hours", type=float, default=72.0)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation-output-hourly-final")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if min(
        args.horizon,
        args.num_cutoffs,
        args.spacing_hours,
        args.max_history_days,
        args.min_history_hours,
        args.effect_min_hours,
    ) < 1:
        raise ValueError("Backtest sizes must be positive")

    flow = staffing_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)

    cutoffs = select_common_cutoffs(
        flow,
        staffing,
        weather,
        schedule_frames,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )
    calendar = build_calendar_frame(flow, cutoffs, args.horizon)
    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "cutoffs.csv", index=False)

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    print(f"Common cutoffs ({len(cutoffs)}): {cutoffs}")
    print("Weather caveat: realized/revised weather, not archived forecast-snapshot replay.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = base.actuals_long(flow, cutoff, args.horizon)
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
            joined["horizon_hour"] = (
                (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
            ).astype(int)
            frames.append(add_errors(joined))

    detail = pd.concat(frames, ignore_index=True)
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["h01_04", "h05_08", "h09_12", "h13_24"],
        include_lowest=True,
    ).astype(str)
    detail.to_csv(args.output_dir / "detail.csv", index=False)

    overall = add_baseline_comparison(
        metrics(detail, ["target_name", "scenario"]), ["target_name"]
    ).sort_values(["target_name", "mae"])
    overall.to_csv(args.output_dir / "summary.csv", index=False)

    by_band = add_baseline_comparison(
        metrics(detail, ["target_name", "horizon_band", "scenario"]),
        ["target_name", "horizon_band"],
    ).sort_values(["target_name", "horizon_band", "mae"])
    by_band.to_csv(args.output_dir / "by_horizon_band.csv", index=False)

    by_hour = add_baseline_comparison(
        metrics(detail, ["target_name", "horizon_hour", "scenario"]),
        ["target_name", "horizon_hour"],
    ).sort_values(["target_name", "horizon_hour", "mae"])
    by_hour.to_csv(args.output_dir / "by_horizon_hour.csv", index=False)

    overall_winners = winners(overall, ["target_name"])
    overall_winners.to_csv(args.output_dir / "winners_by_target.csv", index=False)
    band_winners = winners(by_band, ["target_name", "horizon_band"])
    band_winners.to_csv(args.output_dir / "winners_by_target_horizon_band.csv", index=False)

    nonbaseline = overall.loc[~overall["scenario"].eq("baseline")].copy()
    best_nonbaseline = winners(nonbaseline, ["target_name"])
    best_nonbaseline.to_csv(args.output_dir / "best_feature_family_by_target.csv", index=False)

    print("\n=== Overall winners (baseline allowed) ===")
    print(
        overall_winners[
            ["target_name", "family", "scenario", "mae", "mae_improvement_pct", "beats_baseline"]
        ].to_string(index=False)
    )
    print("\n=== Best feature-family candidate per target ===")
    print(
        best_nonbaseline[
            ["target_name", "family", "scenario", "mae", "mae_improvement_pct", "beats_baseline"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
