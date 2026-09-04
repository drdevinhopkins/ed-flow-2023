#!/usr/bin/env python3
"""Explain the operational D+1..D+7 Chronos-2 daily ED arrival forecast.

This companion keeps ``daily_visits_forecast.csv`` unchanged. It produces:
- ``daily_visits_forecast_explained.csv``: one row per forecast day, enriched with an
  eight-week same-weekday baseline and the three largest model perturbation drivers.
- ``daily_visits_explainability.csv``: long-form driver-group effects for Power BI and
  validation.

Explainability is model perturbation, not SHAP and not causal attribution. For each
feature group, the future covariates are replaced with seasonally typical/neutral values
while the fitted Chronos-2 context is held fixed. The effect is:

    full forecast - counterfactual forecast

A positive effect therefore means the current value of that group raises the model
forecast relative to the neutral counterfactual.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline

import forecast_daily_visits as forecast
from daily_weather_feature_set import (
    SNOW_RECOVERY_COLUMNS,
    SURFACE_CONDITION_COLUMNS,
)
from forecast_daily_visits_from_daily import load_daily_visits_from_dropbox

DEFAULT_FORECAST = Path("daily_visits_forecast.csv")
DEFAULT_WEATHER_SNAPSHOT = Path("daily_visits_weather_snapshot.csv")
DEFAULT_EXPLAINED_OUTPUT = Path("daily_visits_forecast_explained.csv")
DEFAULT_LONG_OUTPUT = Path("daily_visits_explainability.csv")
EXPLAINABILITY_METHOD = "chronos2_group_counterfactual_v1"
SEASONAL_WINDOW_DAYS = 28
WEEKDAY_BASELINE_WEEKS = 8

TEMPERATURE_COLUMNS = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "apparent_temp_mean",
    "apparent_temp_min",
    "apparent_temp_max",
]
PRECIPITATION_COLUMNS = ["precip_sum", "rain_sum"]
CURRENT_SNOW_COLUMNS = [
    "snowfall_sum",
    "snow_depth_max",
    "snowfall_anomaly",
    "snowfall_anomaly_z",
    "major_snow_event",
    "snow_wind_event",
]
SNOW_RECOVERY_ONLY_COLUMNS = [
    column
    for column in SNOW_RECOVERY_COLUMNS
    if column not in {"snowfall_anomaly", "snowfall_anomaly_z", "major_snow_event", "snow_wind_event"}
]
WIND_COLUMNS = ["wind_mean", "wind_max", "gust_max"]
ATMOSPHERE_COLUMNS = [
    "humidity_mean",
    "pressure_mean",
    "pressure_min",
    "pressure_max",
    "pressure_range",
    "cloud_mean",
]

FEATURE_GROUPS = {
    "calendar_closure": list(forecast.CALENDAR_CLOSURE_COLUMNS),
    "temperature": TEMPERATURE_COLUMNS,
    "precipitation": PRECIPITATION_COLUMNS,
    "current_snow": CURRENT_SNOW_COLUMNS,
    "snow_recovery": SNOW_RECOVERY_ONLY_COLUMNS,
    "wind": WIND_COLUMNS,
    "atmosphere": ATMOSPHERE_COLUMNS,
    "surface_conditions": list(SURFACE_CONDITION_COLUMNS),
}

ZERO_NEUTRAL_COLUMNS = set(forecast.CALENDAR_CLOSURE_COLUMNS)
ZERO_NEUTRAL_COLUMNS.update(SNOW_RECOVERY_COLUMNS)
# Zero means "just had major snow" for this recency counter, so neutralize it
# to its seasonally typical historical value instead of zero.
ZERO_NEUTRAL_COLUMNS.discard("days_since_major_snow_capped")
ZERO_NEUTRAL_COLUMNS.update(SURFACE_CONDITION_COLUMNS)


def _prediction_values(frame: pd.DataFrame) -> pd.Series:
    """Return the point prediction regardless of raw/formatted Chronos column naming."""
    for column in ("daily_visits_prediction", "predictions", "0.5"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="raise").astype(float)
    raise ValueError("Forecast frame has no point-prediction column")


def _circular_day_distance(left: pd.Series, target_dayofyear: int) -> pd.Series:
    delta = (left.astype(int) - int(target_dayofyear)).abs()
    return np.minimum(delta, 366 - delta)


def seasonal_reference(
    history: pd.DataFrame,
    future: pd.DataFrame,
    columns: list[str],
    *,
    window_days: int = SEASONAL_WINDOW_DAYS,
) -> pd.DataFrame:
    """Build neutral future covariates from leakage-free historical seasonal medians."""
    reference = future[["id", "ds"]].copy()
    history_dates = pd.to_datetime(history["ds"]).dt.normalize()
    dayofyear = history_dates.dt.dayofyear
    fallback_history = history.tail(min(len(history), 365))

    for column in columns:
        if column in ZERO_NEUTRAL_COLUMNS:
            reference[column] = 0.0
            continue

        values = pd.to_numeric(history[column], errors="coerce")
        fallback = pd.to_numeric(fallback_history[column], errors="coerce").median(skipna=True)
        if not np.isfinite(fallback):
            fallback = values.median(skipna=True)
        if not np.isfinite(fallback):
            fallback = 0.0

        refs: list[float] = []
        for ds in pd.to_datetime(future["ds"]).dt.normalize():
            distance = _circular_day_distance(dayofyear, ds.dayofyear)
            local = values.loc[distance <= window_days].dropna()
            value = local.median() if len(local) >= 7 else fallback
            refs.append(float(value) if np.isfinite(value) else float(fallback))
        reference[column] = refs
    return reference


def build_neutral_future(
    history: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds", forecast.TARGET}
    ]
    return seasonal_reference(history, future, covariates)


def _robust_scale(history: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(history[column], errors="coerce").dropna()
    if values.empty:
        return 1.0
    q25 = float(values.quantile(0.25))
    q75 = float(values.quantile(0.75))
    scale = q75 - q25
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = float(values.std(ddof=0))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return scale


def strongest_feature_detail(
    history: pd.DataFrame,
    actual_row: pd.Series,
    neutral_row: pd.Series,
    columns: list[str],
) -> tuple[str | None, float | None, float | None]:
    candidates: list[tuple[float, str, float, float]] = []
    for column in columns:
        if column not in actual_row.index or column not in neutral_row.index:
            continue
        actual = pd.to_numeric(pd.Series([actual_row[column]]), errors="coerce").iloc[0]
        neutral = pd.to_numeric(pd.Series([neutral_row[column]]), errors="coerce").iloc[0]
        if not np.isfinite(actual) or not np.isfinite(neutral):
            continue
        score = abs(float(actual) - float(neutral)) / _robust_scale(history, column)
        candidates.append((score, column, float(actual), float(neutral)))
    if not candidates:
        return None, None, None
    _score, column, actual, neutral = max(candidates, key=lambda item: item[0])
    return column, actual, neutral


def weekday_baseline(
    history: pd.DataFrame,
    future_dates: pd.Series,
    *,
    weeks: int = WEEKDAY_BASELINE_WEEKS,
) -> pd.Series:
    dates = pd.to_datetime(history["ds"]).dt.normalize()
    target = pd.to_numeric(history[forecast.TARGET], errors="coerce")
    baseline: list[float] = []
    for future_date in pd.to_datetime(future_dates).dt.normalize():
        mask = dates.dt.weekday.eq(future_date.weekday()) & target.notna()
        values = target.loc[mask].tail(weeks)
        if values.empty:
            values = target.dropna().tail(28)
        baseline.append(float(values.mean()))
    return pd.Series(baseline, index=future_dates.index, dtype=float)


def build_explanation_rows(
    pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame,
    full_forecast: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
) -> pd.DataFrame:
    """Run one neutralized counterfactual per operational feature group."""
    neutral = build_neutral_future(history, future)
    full_dates = pd.to_datetime(full_forecast["ds"]).dt.normalize()
    full_values = _prediction_values(full_forecast).to_numpy()

    rows: list[dict[str, object]] = []
    for group, requested_columns in FEATURE_GROUPS.items():
        columns = [column for column in requested_columns if column in future.columns]
        if not columns:
            continue

        counterfactual_future = future.copy()
        for column in columns:
            counterfactual_future[column] = neutral[column].to_numpy()

        counterfactual = forecast.run_daily_forecast(
            pipeline,
            history,
            counterfactual_future,
            horizon_days=horizon_days,
            context_days=context_days,
        )
        counter_values = _prediction_values(counterfactual).to_numpy()
        counter_dates = pd.to_datetime(counterfactual["ds"]).dt.normalize()
        if counter_dates.tolist() != full_dates.tolist():
            raise ValueError(f"Counterfactual date mismatch for group {group}")

        for idx, ds in enumerate(full_dates):
            actual_row = future.loc[pd.to_datetime(future["ds"]).dt.normalize().eq(ds)].iloc[0]
            neutral_row = neutral.loc[pd.to_datetime(neutral["ds"]).dt.normalize().eq(ds)].iloc[0]
            feature_name, feature_value, neutral_value = strongest_feature_detail(
                history, actual_row, neutral_row, columns
            )
            rows.append(
                {
                    "ds": ds,
                    "horizon_day": idx + 1,
                    "driver_group": group,
                    "effect_visits": float(full_values[idx] - counter_values[idx]),
                    "full_prediction": float(full_values[idx]),
                    "counterfactual_prediction": float(counter_values[idx]),
                    "representative_feature": feature_name,
                    "feature_value": feature_value,
                    "neutral_value": neutral_value,
                    "explainability_method": EXPLAINABILITY_METHOD,
                }
            )
    return pd.DataFrame(rows)


def enrich_forecast(
    formatted_forecast: pd.DataFrame,
    history: pd.DataFrame,
    explanations: pd.DataFrame,
) -> pd.DataFrame:
    out = formatted_forecast.copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.normalize()
    out["seasonal_weekday_baseline"] = weekday_baseline(history, out["ds"])
    out["delta_vs_weekday_baseline"] = (
        _prediction_values(out).to_numpy() - out["seasonal_weekday_baseline"].to_numpy()
    )
    out["explainability_method"] = EXPLAINABILITY_METHOD

    explanation_text: list[str] = []
    top_rows: dict[int, list[pd.Series]] = {}
    for idx, ds in enumerate(out["ds"]):
        day_rows = explanations.loc[explanations["ds"].eq(ds)].copy()
        day_rows["abs_effect"] = day_rows["effect_visits"].abs()
        ranked = [row for _, row in day_rows.sort_values("abs_effect", ascending=False).head(3).iterrows()]
        top_rows[idx] = ranked

        prediction = float(_prediction_values(out.iloc[[idx]]).iloc[0])
        baseline = float(out.iloc[idx]["seasonal_weekday_baseline"])
        pieces = []
        for row in ranked:
            pieces.append(f"{row['driver_group']} {float(row['effect_visits']):+.1f}")
        explanation_text.append(
            f"Forecast {prediction:.0f}; 8-week same-weekday baseline {baseline:.0f} "
            f"({prediction - baseline:+.0f}). Largest model perturbations: "
            + (", ".join(pieces) if pieces else "none available")
            + "."
        )

    for rank in range(1, 4):
        groups: list[str | None] = []
        effects: list[float | None] = []
        features: list[str | None] = []
        feature_values: list[float | None] = []
        neutral_values: list[float | None] = []
        for idx in range(len(out)):
            ranked = top_rows[idx]
            if len(ranked) >= rank:
                row = ranked[rank - 1]
                groups.append(str(row["driver_group"]))
                effects.append(float(row["effect_visits"]))
                features.append(row["representative_feature"])
                feature_values.append(row["feature_value"])
                neutral_values.append(row["neutral_value"])
            else:
                groups.append(None)
                effects.append(None)
                features.append(None)
                feature_values.append(None)
                neutral_values.append(None)
        out[f"top_driver_{rank}"] = groups
        out[f"top_driver_{rank}_effect"] = effects
        out[f"top_driver_{rank}_feature"] = features
        out[f"top_driver_{rank}_feature_value"] = feature_values
        out[f"top_driver_{rank}_neutral_value"] = neutral_values

    out["explanation_text"] = explanation_text
    return out


def future_from_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    required = [*forecast.CALENDAR_CLOSURE_COLUMNS, *forecast.RAW_PLUS_SNOW_COLUMNS]
    missing = [column for column in required if column not in snapshot.columns]
    if missing:
        raise ValueError(f"Weather snapshot missing model covariates: {missing}")
    out = snapshot[["ds", *required]].copy()
    out["ds"] = pd.to_datetime(out["ds"]).dt.normalize()
    out["id"] = forecast.SERIES_ID
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="raise").astype(float)
    return out[["id", "ds", *required]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST)
    parser.add_argument("--weather-snapshot", type=Path, default=DEFAULT_WEATHER_SNAPSHOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPLAINED_OUTPUT)
    parser.add_argument("--long-output", type=Path, default=DEFAULT_LONG_OUTPUT)
    parser.add_argument("--context-days", type=int, default=forecast.DEFAULT_CONTEXT_DAYS)
    parser.add_argument("--min-history-days", type=int, default=forecast.DEFAULT_MIN_HISTORY_DAYS)
    parser.add_argument("--model-id", default=forecast.MODEL_ID)
    parser.add_argument("--no-dropbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formatted = pd.read_csv(args.forecast)
    snapshot = pd.read_csv(args.weather_snapshot)
    formatted["ds"] = pd.to_datetime(formatted["ds"]).dt.normalize()
    snapshot["ds"] = pd.to_datetime(snapshot["ds"]).dt.normalize()

    if formatted.empty:
        raise ValueError("Forecast CSV is empty")
    horizon_days = len(formatted)
    if horizon_days != formatted["ds"].nunique():
        raise ValueError("Forecast must have one row per date")

    cutoff = pd.Timestamp(formatted["data_cutoff"].iloc[0]).normalize()
    if not formatted["data_cutoff"].astype(str).eq(str(formatted["data_cutoff"].iloc[0])).all():
        raise ValueError("Forecast contains multiple data cutoffs")

    daily = load_daily_visits_from_dropbox()
    current_complete = daily.loc[daily[forecast.TARGET].notna(), "ds"]
    current_cutoff = pd.Timestamp(current_complete.max()).normalize()
    if current_cutoff != cutoff:
        raise ValueError(
            f"Daily target advanced while explaining forecast: forecast cutoff={cutoff.date()} "
            f"current cutoff={current_cutoff.date()}"
        )

    weather_start = cutoff - pd.Timedelta(days=args.context_days + 14)
    hourly_weather = forecast.build_operational_weather(
        start=weather_start,
        end=cutoff,
        forecast_days=max(horizon_days + 1, 8),
    )
    rebuilt_cutoff, history, _rebuilt_future = forecast.build_forecast_frames(
        daily,
        hourly_weather,
        horizon_days=horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
    )
    if rebuilt_cutoff != cutoff:
        raise ValueError("Rebuilt forecast context cutoff does not match persisted forecast")

    future = future_from_snapshot(snapshot)
    if future["ds"].tolist() != formatted["ds"].tolist():
        raise ValueError("Persisted weather snapshot dates do not match forecast dates")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device} for grouped perturbation explainability", flush=True)
    pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)

    explanations = build_explanation_rows(
        pipeline,
        history,
        future,
        formatted,
        horizon_days=horizon_days,
        context_days=args.context_days,
    )
    explained = enrich_forecast(formatted, history, explanations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.long_output.parent.mkdir(parents=True, exist_ok=True)
    explained.to_csv(args.output, index=False)
    explanations.to_csv(args.long_output, index=False)
    print(f"Wrote {len(explained)} explained forecast rows to {args.output}", flush=True)
    print(f"Wrote {len(explanations)} driver rows to {args.long_output}", flush=True)
    print(explained[[
        "ds",
        "daily_visits_prediction",
        "seasonal_weekday_baseline",
        "top_driver_1",
        "top_driver_1_effect",
        "top_driver_2",
        "top_driver_2_effect",
        "explanation_text",
    ]].to_string(index=False), flush=True)

    if not args.no_dropbox:
        dbx = forecast._dropbox_client()
        if dbx is None:
            print("Dropbox credentials not present; leaving explainability as local CSVs", flush=True)
        else:
            forecast.upload_to_dropbox(
                dbx, args.output, name="daily_visits_forecast_explained.csv", overwrite=True
            )
            forecast.upload_to_dropbox(
                dbx, args.long_output, name="daily_visits_explainability.csv", overwrite=True
            )
            generated = pd.Timestamp.now(tz="UTC")
            stamp = generated.strftime("%Y%m%dT%H%M%SZ")
            forecast.upload_to_dropbox(
                dbx,
                args.output,
                folder="daily_visits_explainability_snapshots",
                name=f"daily_visits_forecast_explained_{stamp}.csv",
                overwrite=False,
            )
            forecast.upload_to_dropbox(
                dbx,
                args.long_output,
                folder="daily_visits_explainability_snapshots",
                name=f"daily_visits_explainability_{stamp}.csv",
                overwrite=False,
            )
            print("Uploaded current and archived daily explainability outputs to Dropbox", flush=True)


if __name__ == "__main__":
    main()
