#!/usr/bin/env python3
"""Targeted Chronos-2 weather/lag backtest for hourly ED inflow.

Targets
-------
* INFLOW_AMBULATORY
* INFLOW_STRETCHER
* INFLOW_AMBULANCES
* Inflow_Total

The experiment separates contemporaneous rain/snow from recent precipitation and
post-snow recovery state. It also writes a descriptive lag-association table after
adjusting each inflow target for month x day-of-week x hour-of-day seasonality.

Historical Forecast weather is a stitched/revised weather series rather than the exact
forecast snapshot available at each historical cutoff. Forecast results therefore
measure weather-signal potential, not leakage-free prospective performance. The lag
association table is descriptive and should not be interpreted causally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
from backtest_hourly_weather_features import engineered_weather_at_cutoff, load_weather
from hourly_weather_features import RAW_HOURLY_WEATHER_COLUMNS, RAW_PLUS_SNOW_HOURLY_COLUMNS

TARGETS = [
    "INFLOW_AMBULATORY",
    "INFLOW_STRETCHER",
    "INFLOW_AMBULANCES",
    "Inflow_Total",
]
LAGS = [0, 1, 3, 6, 12, 24, 48, 72]

RAIN_LAG_COLUMNS = [
    "rain",
    "rain_lag_1h",
    "rain_lag_3h",
    "rain_lag_6h",
    "rain_lag_12h",
    "rain_lag_24h",
    "rain_6h",
    "rain_12h",
    "rain_24h",
]
SNOW_LAG_COLUMNS = [
    "snowfall",
    "snowfall_lag_1h",
    "snowfall_lag_3h",
    "snowfall_lag_6h",
    "snowfall_lag_12h",
    "snowfall_lag_24h",
    "snowfall_6h",
    "snowfall_12h",
    "snowfall_24h",
    "snowfall_48h",
    "snowfall_72h",
]
SNOW_RECOVERY_COLUMNS = [
    "snow_depth",
    "hours_since_snow_capped",
    "hours_since_major_snow_capped",
    "major_snow_24h_event",
    "post_major_snow_6_24h",
    "post_major_snow_24_48h",
    "post_major_snow_48_72h",
    "freeze_thaw_transition",
    "post_snow_thaw",
    "refreeze_after_thaw",
]

SCENARIO_COLUMNS = {
    "baseline": [],
    "rain_now": ["rain"],
    "rain_lags": RAIN_LAG_COLUMNS,
    "snow_now": ["snowfall"],
    "snow_lags": SNOW_LAG_COLUMNS,
    "snow_recovery": SNOW_RECOVERY_COLUMNS,
    "rain_snow_lags": list(dict.fromkeys([*RAIN_LAG_COLUMNS, *SNOW_LAG_COLUMNS, *SNOW_RECOVERY_COLUMNS])),
    "raw_weather": RAW_HOURLY_WEATHER_COLUMNS,
    "raw_plus_snow": RAW_PLUS_SNOW_HOURLY_COLUMNS,
}
SCENARIOS = list(SCENARIO_COLUMNS)


def load_inflow() -> pd.DataFrame:
    # The shared loader and Chronos helper use this module-level target list dynamically.
    base.FLOW_TARGETS = TARGETS
    return base.load_flow()


def add_targeted_lag_features(featured: pd.DataFrame) -> pd.DataFrame:
    out = featured.sort_values("ds").copy()
    rain = pd.to_numeric(out["rain"], errors="coerce").fillna(0.0)
    snow = pd.to_numeric(out["snowfall"], errors="coerce").fillna(0.0)

    for lag in [1, 3, 6, 12, 24]:
        out[f"rain_lag_{lag}h"] = rain.shift(lag).fillna(0.0)
        out[f"snowfall_lag_{lag}h"] = snow.shift(lag).fillna(0.0)
    for hours in [6, 12, 24]:
        out[f"rain_{hours}h"] = rain.rolling(hours, min_periods=1).sum()
    return out


def featured_weather_at_cutoff(
    weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> pd.DataFrame:
    featured = engineered_weather_at_cutoff(
        weather,
        cutoff=cutoff,
        horizon=horizon,
        max_history_days=max_history_days,
    )
    return add_targeted_lag_features(featured)


def scenario_frames(
    scenario: str,
    flow: pd.DataFrame,
    featured_weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    history_start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    history = flow.loc[(flow["ds"] >= history_start) & (flow["ds"] <= cutoff)].copy()
    if history.empty or history[TARGETS].isna().any().any():
        raise ValueError(f"Incomplete inflow history at cutoff {cutoff}")

    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})
    history["id"] = "jgh"
    future["id"] = "jgh"

    if scenario == "baseline":
        return history[["id", "ds", *TARGETS]], None

    columns = SCENARIO_COLUMNS[scenario]
    missing = [column for column in columns if column not in featured_weather.columns]
    if missing:
        raise ValueError(f"Scenario {scenario} missing weather columns: {missing}")

    history = history.merge(featured_weather[["ds", *columns]], on="ds", how="left")
    future = future.merge(featured_weather[["ds", *columns]], on="ds", how="left")
    history[columns] = history[columns].ffill().bfill().fillna(0.0)
    if future[columns].isna().any().any():
        bad = future.loc[future[columns].isna().any(axis=1), "ds"].head(6).tolist()
        raise ValueError(f"Missing future weather covariates for {scenario} at {bad}")

    history, future = base.normalize_numeric_covariates(history, future)
    return (
        history[["id", "ds", *TARGETS, *columns]],
        future[["id", "ds", *columns]],
    )


def add_event_labels(detail: pd.DataFrame, featured: pd.DataFrame) -> pd.DataFrame:
    labels = featured[[
        "ds",
        "rain",
        "snowfall",
        "rain_6h",
        "snowfall_6h",
        "snowfall_24h",
        "major_snow_24h_event",
        "post_major_snow_6_24h",
        "post_major_snow_24_48h",
        "post_major_snow_48_72h",
    ]].copy()
    labels["rain_hour"] = (labels["rain"] >= 0.2).astype(float)
    labels["rain_recent_6h"] = (labels["rain_6h"] >= 1.0).astype(float)
    labels["snow_hour"] = (labels["snowfall"] >= 0.2).astype(float)
    labels["snow_recent_6h"] = (labels["snowfall_6h"] >= 0.5).astype(float)
    return detail.merge(labels, on="ds", how="left")


def summarize(detail: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    summary = detail.groupby([*grouping, "target_name", "scenario"], observed=True, as_index=False).agg(
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
    baseline = summary.loc[summary["scenario"] == "baseline", [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    summary = summary.merge(baseline, on=keys, how="left")
    summary["mae_improvement_vs_baseline"] = summary["baseline_mae"] - summary["mae"]
    summary["mae_improvement_vs_baseline_pct"] = (
        summary["mae_improvement_vs_baseline"] / summary["baseline_mae"].replace(0, np.nan) * 100
    )
    summary["weather_validation_mode"] = "realized/revised weather; not archived forecast snapshots"
    return summary.sort_values([*keys, "mae"])


def lag_associations(flow: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Season/hour-adjusted descriptive association between prior weather and inflow."""
    raw = weather[["ds", "rain", "snowfall"]].copy().sort_values("ds")
    raw["rain"] = pd.to_numeric(raw["rain"], errors="coerce").fillna(0.0)
    raw["snowfall"] = pd.to_numeric(raw["snowfall"], errors="coerce").fillna(0.0)
    raw["snowfall_24h"] = raw["snowfall"].rolling(24, min_periods=1).sum()
    merged = flow.merge(raw, on="ds", how="inner").sort_values("ds").reset_index(drop=True)

    merged["month"] = merged["ds"].dt.month
    merged["dow"] = merged["ds"].dt.dayofweek
    merged["hour"] = merged["ds"].dt.hour

    exposure_defs = {
        "rain_hour": (merged["rain"] >= 0.2),
        "snow_hour": (merged["snowfall"] >= 0.2),
        "major_snow_24h": (merged["snowfall_24h"] >= 5.0),
    }

    records: list[dict[str, object]] = []
    for target in TARGETS:
        target_values = pd.to_numeric(merged[target], errors="coerce")
        expected = target_values.groupby([merged["month"], merged["dow"], merged["hour"]]).transform("median")
        residual = target_values - expected
        residual_sd = float(residual.std(skipna=True))

        for exposure_name, exposure in exposure_defs.items():
            exposure = exposure.astype(bool)
            for lag in LAGS:
                prior = exposure.shift(lag, fill_value=False)
                valid = residual.notna()
                event_values = residual.loc[valid & prior]
                none_values = residual.loc[valid & ~prior]
                if event_values.empty or none_values.empty:
                    continue
                difference = float(event_values.mean() - none_values.mean())
                records.append({
                    "target_name": target,
                    "exposure": exposure_name,
                    "lag_hours": lag,
                    "n_event": int(event_values.size),
                    "n_nonevent": int(none_values.size),
                    "adjusted_mean_inflow_difference": difference,
                    "standardized_difference": difference / residual_sd if residual_sd > 0 else np.nan,
                    "event_residual_mean": float(event_values.mean()),
                    "nonevent_residual_mean": float(none_values.mean()),
                    "interpretation": "descriptive month/day-of-week/hour adjusted association; not causal",
                })
    return pd.DataFrame(records).sort_values(["target_name", "exposure", "lag_hours"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather-url", default=base.WEATHER_URL)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=12)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=180)
    parser.add_argument("--min-history-hours", type=int, default=24 * 28)
    parser.add_argument("--model-id", default=base.MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("validation/inflow-weather-lag-backtest"))
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flow = load_inflow()
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
    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "inflow_weather_cutoffs.csv", index=False)

    associations = lag_associations(flow, weather)
    associations.to_csv(args.output_dir / "inflow_weather_lag_associations.csv", index=False)

    print(f"Targets: {', '.join(TARGETS)}")
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print(f"Scenarios: {args.scenarios}")
    print("WARNING: historical weather is signal-potential data, not forecast-snapshot replay.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = base.actuals_long(flow, cutoff, args.horizon)
        featured = featured_weather_at_cutoff(
            weather,
            cutoff=cutoff,
            horizon=args.horizon,
            max_history_days=args.max_history_days,
        )
        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future = scenario_frames(
                scenario,
                flow,
                featured,
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
            joined = add_event_labels(joined, featured)
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True).dropna(subset=["prediction", "actual"])
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["1-4h", "5-8h", "9-12h", "13-24h"],
        include_lowest=True,
    )
    detail.to_csv(args.output_dir / "inflow_weather_detail.csv", index=False)

    summary = summarize(detail, [])
    summary.to_csv(args.output_dir / "inflow_weather_summary.csv", index=False)
    by_horizon = summarize(detail, ["horizon_band"])
    by_horizon.to_csv(args.output_dir / "inflow_weather_by_horizon.csv", index=False)

    event_masks = {
        "rain_hour": detail["rain_hour"] > 0,
        "rain_recent_6h": detail["rain_recent_6h"] > 0,
        "snow_hour": detail["snow_hour"] > 0,
        "snow_recent_6h": detail["snow_recent_6h"] > 0,
        "major_snow_24h": detail["major_snow_24h_event"] > 0,
        "post_major_snow_6_24h": detail["post_major_snow_6_24h"] > 0,
        "post_major_snow_24_48h": detail["post_major_snow_24_48h"] > 0,
        "post_major_snow_48_72h": detail["post_major_snow_48_72h"] > 0,
    }
    event_frames: list[pd.DataFrame] = []
    for event, mask in event_masks.items():
        if mask.any():
            part = detail.loc[mask].copy()
            part["event"] = event
            event_frames.append(part)
    if event_frames:
        summarize(pd.concat(event_frames, ignore_index=True), ["event"]).to_csv(
            args.output_dir / "inflow_weather_by_event.csv", index=False
        )

    winners = summary.sort_values(["target_name", "mae"]).groupby("target_name", as_index=False).first()
    winners.to_csv(args.output_dir / "inflow_weather_winners.csv", index=False)
    horizon_winners = by_horizon.sort_values(["target_name", "horizon_band", "mae"]).groupby(
        ["target_name", "horizon_band"], observed=True, as_index=False
    ).first()
    horizon_winners.to_csv(args.output_dir / "inflow_weather_horizon_winners.csv", index=False)

    print("\nOverall winners:")
    print(winners[["target_name", "scenario", "n", "mae", "mae_improvement_vs_baseline_pct"]].to_string(index=False))
    print("\nHorizon winners:")
    print(horizon_winners[["target_name", "horizon_band", "scenario", "mae", "mae_improvement_vs_baseline_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
