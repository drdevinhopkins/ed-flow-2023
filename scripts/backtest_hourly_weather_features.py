#!/usr/bin/env python3
"""Native Chronos-2 ablation of engineered hourly weather covariates.

This is deliberately separate from production ``chronos_forecast.py``.  It tests whether
hourly translations of the validated daily weather concepts improve the six canonical
ED flow targets before anything is promoted to the operational model.

Historical Forecast weather is a stitched/revised weather series, not the exact forecast
snapshot available at each historical cutoff. Results therefore measure weather-signal
potential, not a fully leakage-free real-time replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
from hourly_weather_features import (
    add_hourly_weather_features,
    fit_hourly_climatology,
    weather_columns_for_scenario,
)

WEATHER_ONLY_SCENARIOS = {
    "raw_weather",
    "raw_plus_snow",
    "raw_plus_thermal",
    "raw_plus_storm",
    "engineered_weather",
}
ALL_COVARIATE_SCENARIOS = {
    "all_raw",
    "all_raw_plus_snow",
    "all_engineered",
}
SCENARIOS = [
    "baseline",
    "raw_weather",
    "raw_plus_snow",
    "raw_plus_thermal",
    "raw_plus_storm",
    "engineered_weather",
    "all_raw",
    "all_raw_plus_snow",
    "all_engineered",
]


def load_weather(source: str) -> pd.DataFrame:
    weather = pd.read_csv(source)
    weather["ds"] = base.parse_ds(weather["ds"])
    weather = weather.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")
    return weather.reset_index(drop=True)


def _weather_scenario_name(scenario: str) -> str:
    if scenario == "all_raw":
        return "raw_weather"
    if scenario == "all_raw_plus_snow":
        return "raw_plus_snow"
    if scenario == "all_engineered":
        return "engineered_weather"
    return scenario


def engineered_weather_at_cutoff(
    weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> pd.DataFrame:
    history_start = cutoff - pd.Timedelta(days=max_history_days + 7)
    future_end = cutoff + pd.Timedelta(hours=horizon)
    window = weather.loc[(weather["ds"] >= history_start) & (weather["ds"] <= future_end)].copy()
    history_weather = window.loc[window["ds"] <= cutoff].copy()
    climatology = fit_hourly_climatology(history_weather)
    featured = add_hourly_weather_features(window, climatology)

    expected_future = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    available = set(featured["ds"])
    missing = [stamp for stamp in expected_future if stamp not in available]
    if missing:
        raise ValueError(f"Weather missing future hours at cutoff {cutoff}: {missing[:6]}")
    return featured


def _add_staffing(history: pd.DataFrame, future: pd.DataFrame, staffing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    staff_columns = [column for column in staffing.columns if column != "ds"]
    history = history.merge(staffing, on="ds", how="left")
    future = future.merge(staffing, on="ds", how="left")
    for column in staff_columns:
        if column.startswith("physician__"):
            history[column] = history[column].fillna("NotWorking")
            future[column] = future[column].fillna("NotWorking")
        elif column == "oncall_physician_id":
            history[column] = history[column].fillna("None")
            future[column] = future[column].fillna("None")
        else:
            history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0)
            future[column] = pd.to_numeric(future[column], errors="coerce").fillna(0)
    return history, future


def scenario_frames(
    scenario: str,
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    history_start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    history = flow.loc[(flow["ds"] >= history_start) & (flow["ds"] <= cutoff)].copy()
    if history.empty or history[base.FLOW_TARGETS].isna().any().any():
        raise ValueError(f"Incomplete target history at cutoff {cutoff}")

    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})
    featured_weather = None

    use_all = scenario in ALL_COVARIATE_SCENARIOS
    use_weather = scenario in WEATHER_ONLY_SCENARIOS or use_all

    if use_all:
        history, future = _add_staffing(history, future, staffing)
        history = base.add_holiday_flags(history)
        future = base.add_holiday_flags(future)

    if use_weather:
        featured_weather = engineered_weather_at_cutoff(
            weather,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=max_history_days,
        )
        weather_scenario = _weather_scenario_name(scenario)
        weather_columns = weather_columns_for_scenario(weather_scenario)
        history = history.merge(featured_weather[["ds", *weather_columns]], on="ds", how="left")
        future = future.merge(featured_weather[["ds", *weather_columns]], on="ds", how="left")
        history[weather_columns] = history[weather_columns].ffill().bfill()
        if future[weather_columns].isna().any().any():
            missing_columns = future[weather_columns].columns[future[weather_columns].isna().any()].tolist()
            raise ValueError(f"Missing future weather covariates at {cutoff}: {missing_columns}")

    history["id"] = "jgh"
    future["id"] = "jgh"
    if scenario == "baseline":
        return history[["id", "ds", *base.FLOW_TARGETS]], None, None

    covariates = [
        column for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    history, future = base.normalize_numeric_covariates(history, future)
    return (
        history[["id", "ds", *base.FLOW_TARGETS, *covariates]],
        future[["id", "ds", *covariates]],
        featured_weather,
    )


def add_event_labels(
    detail: pd.DataFrame,
    featured_weather: pd.DataFrame,
) -> pd.DataFrame:
    label_columns = [
        "ds",
        "major_snow_24h_event",
        "post_major_snow_6_24h",
        "post_major_snow_24_48h",
        "post_major_snow_48_72h",
        "freeze_thaw_transition",
        "cold_windy_event",
        "storm_severity_index",
    ]
    labels = featured_weather[label_columns].copy()
    labels["storm_event"] = (labels["storm_severity_index"] >= 2.0).astype(float)
    return detail.merge(labels, on="ds", how="left")


def summarize(detail: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    summary = detail.groupby([*grouping, "target_name", "scenario"], as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary["wape"] = summary.pop("abs_error_sum") / summary.pop("abs_actual_sum").replace(0, np.nan)

    keys = [*grouping, "target_name"]
    for reference in ["baseline", "raw_weather", "all_raw"]:
        ref = summary.loc[summary["scenario"] == reference, [*keys, "mae"]].rename(
            columns={"mae": f"{reference}_mae"}
        )
        summary = summary.merge(ref, on=keys, how="left")
        summary[f"mae_improvement_vs_{reference}"] = summary[f"{reference}_mae"] - summary["mae"]
        summary[f"mae_improvement_vs_{reference}_pct"] = (
            summary[f"mae_improvement_vs_{reference}"]
            / summary[f"{reference}_mae"].replace(0, np.nan)
            * 100
        )
    summary["weather_validation_mode"] = "realized/revised weather; not archived forecast snapshots"
    return summary.sort_values([*keys, "mae"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-url", default=base.WEATHER_URL)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=12)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=180)
    parser.add_argument("--min-history-hours", type=int, default=24 * 28)
    parser.add_argument("--model-id", default=base.MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("validation/hourly-weather-backtest"))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flow = base.load_flow()
    staffing = base.load_staffing()
    weather = load_weather(args.weather_url)
    cutoffs = base.select_cutoffs(
        flow,
        staffing,
        weather,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=max(args.min_history_hours, 24 * 28),
    )
    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "hourly_weather_cutoffs.csv", index=False)

    print(f"Targets: {', '.join(base.FLOW_TARGETS)}")
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print(f"Scenarios: {args.scenarios}")
    print("WARNING: historical weather is signal-potential data, not forecast-snapshot replay.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)

    frames: list[pd.DataFrame] = []
    label_cache: dict[pd.Timestamp, pd.DataFrame] = {}
    for cutoff in cutoffs:
        actual = base.actuals_long(flow, cutoff, args.horizon)
        if cutoff not in label_cache:
            label_cache[cutoff] = engineered_weather_at_cutoff(
                weather,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
            )
        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future, _ = scenario_frames(
                scenario,
                flow,
                staffing,
                weather,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
            )
            forecast = base.run_forecast(pipeline, history, future, horizon=args.horizon).rename(
                columns={"predictions": "prediction"}
            )
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            joined = add_event_labels(joined, label_cache[cutoff])
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["1-4h", "5-8h", "9-12h", "13-24h"],
        include_lowest=True,
    )
    detail.to_csv(args.output_dir / "hourly_weather_detail.csv", index=False)

    summary = summarize(detail, [])
    summary.to_csv(args.output_dir / "hourly_weather_summary.csv", index=False)
    by_horizon = summarize(detail, ["horizon_band"])
    by_horizon.to_csv(args.output_dir / "hourly_weather_by_horizon.csv", index=False)

    event_frames: list[pd.DataFrame] = []
    event_masks = {
        "major_snow_24h": detail["major_snow_24h_event"] > 0,
        "post_major_snow": detail[["post_major_snow_6_24h", "post_major_snow_24_48h", "post_major_snow_48_72h"]].max(axis=1) > 0,
        "freeze_thaw": detail["freeze_thaw_transition"] > 0,
        "cold_windy": detail["cold_windy_event"] > 0,
        "storm": detail["storm_event"] > 0,
    }
    for event, mask in event_masks.items():
        if mask.any():
            part = detail.loc[mask].copy()
            part["event"] = event
            event_frames.append(part)
    if event_frames:
        by_event = summarize(pd.concat(event_frames, ignore_index=True), ["event"])
        by_event.to_csv(args.output_dir / "hourly_weather_by_event.csv", index=False)

    winners = (
        summary.sort_values(["target_name", "mae"])
        .groupby("target_name", as_index=False)
        .first()
    )
    winners.to_csv(args.output_dir / "hourly_weather_winners.csv", index=False)
    print("\nOverall winners:")
    print(winners[["target_name", "scenario", "n", "mae", "mae_improvement_vs_baseline_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
