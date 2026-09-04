#!/usr/bin/env python3
"""Generate the latest total-arrivals-by-midnight forecast.

This is an additive companion to ``hourly_forecast_v2_1.py``.  It never changes
``forecast-v2.1.csv`` and suppresses output when the live hourly feed is stale,
incomplete, or invalid.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from intraday_day_completion_model import (
    FLOW_URL,
    LOCAL_TZ,
    TOTAL_TBS_COMPONENTS,
    _fit_quantile_models,
    _predict_remaining_quantiles,
    add_curve_features,
    apply_quantile_corrections,
    build_snapshots,
    build_weather_features,
    expected_local_hours,
    feature_sets,
    fit_completion_curve,
    fit_prior_update,
    fit_quantile_corrections,
    load_hourly_flow,
)

MODEL_VERSION = "intraday-ensemble-v1-2026-08-28"
MODEL_HOURS = range(6, 23)
PROSPECTIVE_HOURS = range(11, 19)
DEFAULT_OUTPUT = Path("intraday-daily-inflow-forecast.csv")
DEFAULT_STATUS = Path("intraday-daily-inflow-status.json")


class DataQualityError(RuntimeError):
    """An expected input problem that must suppress the forecast."""


def _local_naive(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(LOCAL_TZ)
    return stamp.tz_convert(LOCAL_TZ).tz_localize(None)


def validate_live_flow(
    flow: pd.DataFrame,
    *,
    now: pd.Timestamp,
    max_age_minutes: int = 90,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """Return a safe current-day prefix or raise a suppressing error."""

    latest_ds = _local_naive(pd.Timestamp(flow["ds"].max()))
    local_now = _local_naive(now)
    age_minutes = (local_now - latest_ds).total_seconds() / 60.0
    if age_minutes < -5:
        raise DataQualityError(
            f"latest flow timestamp is {abs(age_minutes):.0f} minutes in the future"
        )
    if age_minutes > max_age_minutes:
        raise DataQualityError(
            f"latest flow timestamp is stale ({age_minutes:.0f} minutes; "
            f"limit {max_age_minutes})"
        )

    latest_day = latest_ds.normalize()
    current = flow.loc[flow["day"].eq(latest_day)].sort_values(
        ["ds", "_source_order"]
    ).copy()
    actual_hours = current["ds"].dt.hour.astype(int).tolist()
    expected_hours = expected_local_hours(latest_day)
    latest_positions = [
        position
        for position, hour in enumerate(expected_hours)
        if hour == int(latest_ds.hour)
    ]
    latest_position = min(
        latest_positions,
        key=lambda position: abs((position + 1) - len(actual_hours)),
    )
    expected_prefix = expected_hours[: latest_position + 1]
    if actual_hours != expected_prefix:
        raise DataQualityError(
            "current day has missing or out-of-order hours: "
            f"expected {expected_prefix}, got {actual_hours}"
        )

    cutoff_hour = int(latest_ds.hour)
    if cutoff_hour not in MODEL_HOURS:
        raise DataQualityError(
            f"cutoff hour {cutoff_hour:02d}:00 is outside the 06:00-22:00 model window"
        )

    current_inflow = pd.to_numeric(current["Inflow_Total"], errors="coerce")
    if (
        current_inflow.isna().any()
        or not np.isfinite(current_inflow.to_numpy(dtype=float)).all()
        or current_inflow.lt(0).any()
    ):
        raise DataQualityError(
            "current-day inflow contains missing, non-finite, or negative values"
        )

    missing_state = [
        column for column in TOTAL_TBS_COMPONENTS if column not in current.columns
    ]
    if missing_state:
        raise DataQualityError(
            f"current flow is missing Total_TBS components: {missing_state}"
        )
    latest_state = current.iloc[-1][list(TOTAL_TBS_COMPONENTS)].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        latest_state.isna().any()
        or not np.isfinite(latest_state.to_numpy(dtype=float)).all()
        or latest_state.lt(0).any()
    ):
        raise DataQualityError(
            "latest Total_TBS components contain missing, non-finite, or negative values"
        )
    return latest_ds, current


def build_intraday_forecast(
    *,
    flow_source: str | Path,
    weather_source: str | Path,
    generated_at: pd.Timestamp,
    max_age_minutes: int = 90,
    min_train_days: int = 365,
    calibration_days: int = 56,
    calibration_shrinkage_days: float = 28.0,
    max_iter: int = 200,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit the locked ensemble and return one guarded live forecast row."""

    flow = load_hourly_flow(flow_source)
    latest_ds, _ = validate_live_flow(
        flow, now=generated_at, max_age_minutes=max_age_minutes
    )
    weather = build_weather_features(weather_source)
    live_day = latest_ds.normalize()

    training_days = int(
        flow.loc[
            flow["is_complete_day"] & flow["day"].lt(live_day), "day"
        ].nunique()
    )
    if training_days < min_train_days:
        raise DataQualityError(
            f"only {training_days} complete training days; require {min_train_days}"
        )

    prepared_flow = flow.copy()
    prepared_flow.loc[prepared_flow["day"].eq(live_day), "is_complete_day"] = True
    snapshots = build_snapshots(prepared_flow, calendar_mode="rich", weather=weather)
    train = snapshots.loc[snapshots["day"].lt(live_day)].copy()
    live = snapshots.loc[
        snapshots["day"].eq(live_day) & snapshots["ds"].eq(latest_ds)
    ].copy()
    if len(live) != 1:
        raise DataQualityError(
            f"expected one live snapshot at {latest_ds}; found {len(live)}"
        )
    weather_columns = [
        column for column in live if column.startswith("weather_current_")
    ]
    if not weather_columns or live[weather_columns].isna().all(axis=None):
        raise DataQualityError(
            "no cutoff-safe weather observation is available within two hours"
        )

    curve = fit_completion_curve(train)
    train = add_curve_features(train, curve)
    live = add_curve_features(live, curve)

    train_days = np.array(sorted(pd.to_datetime(train["day"].unique())))
    calibration_size = min(calibration_days, len(train_days) - 28)
    if calibration_size < 7:
        raise DataQualityError("insufficient history for nested calibration")
    calibration_start = train_days[-calibration_size]
    core = train.loc[train["day"].lt(calibration_start)].copy()
    calibration = train.loc[train["day"].ge(calibration_start)].copy()
    inner_curve = fit_completion_curve(core)
    core = add_curve_features(core, inner_curve)
    calibration = add_curve_features(calibration, inner_curve)

    state_features = feature_sets(core)["boosted_state"]
    calibration_models = _fit_quantile_models(
        core,
        features=state_features,
        max_iter=max_iter,
        random_state=random_state,
    )
    calibration_prediction = _predict_remaining_quantiles(
        calibration_models, calibration, features=state_features
    )
    corrections = fit_quantile_corrections(
        calibration["remaining_arrivals"].to_numpy(dtype=float),
        calibration_prediction,
        calibration["cutoff_hour"].to_numpy(dtype=int),
        shrinkage_days=calibration_shrinkage_days,
    )

    sets = feature_sets(train)
    calendar_weather_features = sets.get("boosted_calendar_weather")
    if not calendar_weather_features:
        raise DataQualityError("calendar/weather feature route is unavailable")
    state_models = _fit_quantile_models(
        train,
        features=sets["boosted_state"],
        max_iter=max_iter,
        random_state=random_state,
    )
    weather_models = _fit_quantile_models(
        train,
        features=calendar_weather_features,
        max_iter=max_iter,
        random_state=random_state,
    )
    state_raw = _predict_remaining_quantiles(
        state_models, live, features=sets["boosted_state"]
    )
    state_calibrated = apply_quantile_corrections(
        state_raw,
        live["cutoff_hour"].to_numpy(dtype=int),
        corrections,
    )[0]
    weather_raw = _predict_remaining_quantiles(
        weather_models, live, features=calendar_weather_features
    )[0]

    observed = float(live["cumulative_arrivals"].iat[0])
    predicted_total = max(
        observed,
        observed + 0.5 * (state_calibrated[1] + weather_raw[1]),
    )
    p10_total = max(
        observed,
        min(observed + state_calibrated[0], predicted_total),
    )
    p90_total = max(
        predicted_total,
        observed + state_calibrated[2],
    )

    prior_params = fit_prior_update(train)
    cutoff_hour = int(live["cutoff_hour"].iat[0])
    baseline_total = max(
        observed,
        float(
            live["prior_total"].iat[0]
            + prior_params.loc[cutoff_hour, "beta"]
            * live["pace_residual"].iat[0]
        ),
    )
    if not observed <= p10_total <= predicted_total <= p90_total:
        raise RuntimeError("intraday forecast interval invariant failed")

    generated_at = pd.Timestamp(generated_at)
    if generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("UTC")
    expected_additional = float(predicted_total - observed)
    forecast_text = (
        f"{observed:.0f} arrivals through {cutoff_hour:02d}:00. "
        f"Forecast: {predicted_total:.0f} total by midnight "
        f"(P80 {p10_total:.0f}-{p90_total:.0f}); "
        f"{expected_additional:.0f} additional arrivals expected."
    )
    return {
        "generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
        "generated_at_local": generated_at.tz_convert(LOCAL_TZ).isoformat(),
        "forecast_day": live_day.date().isoformat(),
        "cutoff_ds_local": latest_ds.isoformat(),
        "cutoff_hour": cutoff_hour,
        "observed_arrivals": observed,
        "predicted_total": float(predicted_total),
        "p10_total": float(p10_total),
        "p90_total": float(p90_total),
        "expected_additional_arrivals": expected_additional,
        "prior_update_baseline": float(baseline_total),
        "model_version": MODEL_VERSION,
        "method": "50pct_calendar_weather_plus_50pct_calibrated_ed_state",
        "within_prospective_window": cutoff_hour in PROSPECTIVE_HOURS,
        "prospective_window": "11:00-18:00 America/Montreal",
        "forecast_text": forecast_text,
        "status": "experimental_forecast",
    }


def write_forecast(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def write_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2) + "\n")


def upload_to_dropbox(paths: list[Path]) -> None:
    import dropbox
    import requests

    from utils import upload

    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([app_key, app_secret, refresh_token]):
        raise RuntimeError("Dropbox credentials are required when --upload-dropbox is used")
    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    client = dropbox.Dropbox(response.json()["access_token"])
    for path in paths:
        upload(client, str(path), "", "", path.name, overwrite=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-csv", default=FLOW_URL)
    parser.add_argument("--weather-csv", required=True)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--status-json", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--now")
    parser.add_argument("--max-age-minutes", type=int, default=90)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--calibration-days", type=int, default=56)
    parser.add_argument("--calibration-shrinkage-days", type=float, default=28.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--upload-dropbox", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generated_at = (
        pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    )
    if generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("UTC")

    exit_code = 0
    forecast: dict[str, object] | None = None
    try:
        forecast = build_intraday_forecast(
            flow_source=args.flow_csv,
            weather_source=args.weather_csv,
            generated_at=generated_at,
            max_age_minutes=args.max_age_minutes,
            min_train_days=args.min_train_days,
            calibration_days=args.calibration_days,
            calibration_shrinkage_days=args.calibration_shrinkage_days,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        write_forecast(args.output_csv, forecast)
        status = {
            "status": "forecast_written",
            "generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
            "forecast": forecast,
        }
    except DataQualityError as exc:
        status = {
            "status": "suppressed_data_quality",
            "reason": str(exc),
            "generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
            "model_version": MODEL_VERSION,
        }
    except Exception as exc:  # unexpected errors are recorded and surfaced to Actions
        status = {
            "status": "suppressed_model_error",
            "reason": f"{type(exc).__name__}: {exc}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
        }
        exit_code = 2

    write_status(args.status_json, status)
    print(json.dumps(status, indent=2), flush=True)
    if args.upload_dropbox:
        upload_paths = [args.status_json]
        if forecast is not None:
            upload_paths.insert(0, args.output_csv)
        upload_to_dropbox(upload_paths)
        print("Uploaded " + ", ".join(path.name for path in upload_paths))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
