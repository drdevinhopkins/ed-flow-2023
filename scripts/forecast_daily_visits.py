#!/usr/bin/env python3
"""Operational 7-day native Chronos-2 forecast of JGH daily ED arrivals.

The model uses the validated calendar/closure covariates plus the promoted
``raw_plus_snow`` weather feature set. The weather feature set is shared with the
backtest through ``daily_weather_feature_set.py``.

Historical weather is backfilled only for the model context window. Future rows are
fetched directly from the current Open-Meteo forecast in Montreal local time. Seasonal
weather climatology is fit strictly through the last complete ED day.

The research backtests required at least 120 contiguous complete target days. Production
uses a 28-day hard floor so a single older incomplete source day does not suppress the
forecast entirely; output explicitly records whether the current context is shorter than
the 120-day validated-context threshold.

Each operational run also writes the exact D+1..D+7 covariate snapshot passed to
Chronos. Once scheduled on main, date-stamped forecast and weather snapshots are archived
to Dropbox so future accuracy validation can use genuinely forecast-time-available inputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dropbox
import numpy as np
import pandas as pd
import requests
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from backtest_daily_holiday_targeted import (
    CALENDAR_CLOSURE_COLUMNS,
    engineer_targeted_features,
)
from backtest_holiday_features import (
    FLOW_URL,
    MAX_CONTEXT_DAYS,
    MODEL_ID,
    SERIES_ID,
    TARGET,
    contiguous_history,
    load_daily_visits,
)
from daily_weather_feature_set import (
    PROMOTED_WEATHER_FEATURE_SET,
    RAW_PLUS_SNOW_COLUMNS,
)
from live_weather_forecast import build_operational_weather
from utils import upload
from weather_features import add_weather_features, aggregate_hourly_weather, fit_climatology

DEFAULT_HORIZON_DAYS = 7
DEFAULT_CONTEXT_DAYS = 1095
DEFAULT_MIN_HISTORY_DAYS = 28
VALIDATED_MIN_HISTORY_DAYS = 120
DEFAULT_OUTPUT = Path("daily_visits_forecast.csv")
DEFAULT_WEATHER_SNAPSHOT_OUTPUT = Path("daily_visits_weather_snapshot.csv")


def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = engineer_targeted_features(frame)
    keep = list(frame.columns)
    for column in CALENDAR_CLOSURE_COLUMNS:
        if column not in keep:
            keep.append(column)
    return featured[keep]


def _fill_covariates(
    history: pd.DataFrame,
    future: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match the validated weather-backtest missing-value policy."""
    history = history.copy()
    future = future.copy()

    zero_history = {
        "snowfall_lag1",
        "snowfall_lag2",
        "snowfall_lag3",
        "snowfall_3d_total",
        "snowfall_7d_total",
        "snow_days_3d",
        "day_after_major_snow",
        "two_days_after_major_snow",
        "three_days_after_major_snow",
        "days_since_major_snow_capped",
        "major_snow_event",
        "snow_wind_event",
        "freeze_thaw_day",
        "post_snow_thaw",
        "refreeze_after_thaw",
    }

    for column in columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
        future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")

        if column in zero_history:
            history[column] = history[column].fillna(0.0)
        else:
            history[column] = history[column].ffill()
            fallback = history[column].median(skipna=True)
            if not np.isfinite(fallback):
                fallback = 0.0
            history[column] = history[column].fillna(float(fallback))

        if future[column].isna().any():
            missing = future.loc[future[column].isna(), "ds"].dt.date.tolist()[:7]
            raise ValueError(f"Missing future covariate {column} at {missing}")

    return history, future


def build_forecast_frames(
    daily: pd.DataFrame,
    hourly_weather: pd.DataFrame,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    context_days: int = DEFAULT_CONTEXT_DAYS,
    min_history_days: int = DEFAULT_MIN_HISTORY_DAYS,
) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    """Build the exact history/future frames passed to Chronos-2."""
    complete = daily.loc[daily[TARGET].notna(), "ds"]
    if complete.empty:
        raise ValueError("No complete daily ED arrivals available")
    cutoff = pd.Timestamp(complete.max()).normalize()

    history = contiguous_history(
        daily,
        cutoff,
        context_days=context_days,
        min_history_days=min_history_days,
    )
    future_dates = pd.date_range(
        cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D"
    )
    future = pd.DataFrame({"ds": future_dates})

    daily_weather = aggregate_hourly_weather(hourly_weather)
    daily_weather = (
        daily_weather.loc[daily_weather["temp_mean"].notna()]
        .sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )
    if daily_weather.empty:
        raise ValueError("Weather source produced no daily rows")

    history_weather = daily_weather.loc[daily_weather["ds"] <= cutoff].copy()
    climatology = fit_climatology(history_weather)
    future_end = future_dates.max()
    weather_through_horizon = daily_weather.loc[daily_weather["ds"] <= future_end].copy()
    featured_weather = add_weather_features(weather_through_horizon, climatology)

    featured_dates = set(featured_weather["ds"])
    missing_dates = [date.date() for date in future_dates if date not in featured_dates]
    if missing_dates:
        raise ValueError(f"Live weather does not cover forecast dates: {missing_dates}")

    history = _calendar_features(history)
    future = _calendar_features(future)

    weather_columns = RAW_PLUS_SNOW_COLUMNS
    history = history.merge(
        featured_weather[["ds", *weather_columns]], on="ds", how="left"
    )
    future = future.merge(
        featured_weather[["ds", *weather_columns]], on="ds", how="left"
    )
    history, future = _fill_covariates(history, future, weather_columns)

    covariates = [*CALENDAR_CLOSURE_COLUMNS, *weather_columns]
    covariates = list(dict.fromkeys(covariates))
    for column in CALENDAR_CLOSURE_COLUMNS:
        history[column] = (
            pd.to_numeric(history[column], errors="coerce").fillna(0).astype("float64")
        )
        future[column] = (
            pd.to_numeric(future[column], errors="coerce").fillna(0).astype("float64")
        )

    history["id"] = SERIES_ID
    future["id"] = SERIES_ID
    return (
        cutoff,
        history[["id", "ds", TARGET, *covariates]],
        future[["id", "ds", *covariates]],
    )


def run_daily_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
) -> pd.DataFrame:
    result = pipeline.predict_df(
        history,
        prediction_length=horizon_days,
        future_df=future,
        id_column="id",
        timestamp_column="ds",
        target=[TARGET],
        quantile_levels=[0.1, 0.5, 0.9],
        context_length=min(context_days, MAX_CONTEXT_DAYS, len(history)),
    )
    if len(result) != horizon_days:
        raise ValueError(f"Expected {horizon_days} forecast rows, got {len(result)}")
    return result.copy()


def format_output(
    forecast: pd.DataFrame,
    future: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    history_days: int,
    generated_at: pd.Timestamp,
) -> pd.DataFrame:
    out = forecast.copy()
    if "predictions" in out.columns:
        out = out.rename(columns={"predictions": "daily_visits_prediction"})
    out["ds"] = pd.to_datetime(out["ds"]).dt.normalize()
    out["data_cutoff"] = cutoff
    out["horizon_day"] = ((out["ds"] - cutoff) / pd.Timedelta(days=1)).astype(int)
    out["weather_feature_set"] = PROMOTED_WEATHER_FEATURE_SET
    out["history_days"] = history_days
    out["validated_context_threshold_days"] = VALIDATED_MIN_HISTORY_DAYS
    out["short_context_warning"] = history_days < VALIDATED_MIN_HISTORY_DAYS
    out["forecast_generated_at_utc"] = generated_at.isoformat()

    context_columns = [
        "ds",
        "temp_mean",
        "temp_min",
        "temp_max",
        "apparent_temp_mean",
        "precip_sum",
        "snowfall_sum",
        "snow_depth_max",
        "wind_max",
        "gust_max",
        "major_snow_event",
        "day_after_major_snow",
        "two_days_after_major_snow",
        "three_days_after_major_snow",
        "freeze_thaw_day",
        "post_snow_thaw",
        "refreeze_after_thaw",
    ]
    context_columns = [column for column in context_columns if column in future.columns]
    out = out.merge(future[context_columns], on="ds", how="left")

    preferred = [
        "ds",
        "target_name",
        "daily_visits_prediction",
        "0.1",
        "0.5",
        "0.9",
        "data_cutoff",
        "horizon_day",
        "weather_feature_set",
        "history_days",
        "validated_context_threshold_days",
        "short_context_warning",
        "forecast_generated_at_utc",
    ]
    preferred += [column for column in context_columns if column != "ds"]
    ordered = [column for column in preferred if column in out.columns]
    ordered += [column for column in out.columns if column not in ordered]
    return out[ordered]


def build_weather_snapshot(
    future: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    generated_at: pd.Timestamp,
) -> pd.DataFrame:
    """Persist the exact known-future covariates supplied to Chronos at this run."""
    columns = [
        column
        for column in [*CALENDAR_CLOSURE_COLUMNS, *RAW_PLUS_SNOW_COLUMNS]
        if column in future.columns
    ]
    out = future[["ds", *columns]].copy()
    out["data_cutoff"] = cutoff
    out["horizon_day"] = ((out["ds"] - cutoff) / pd.Timedelta(days=1)).astype(int)
    out["weather_feature_set"] = PROMOTED_WEATHER_FEATURE_SET
    out["forecast_generated_at_utc"] = generated_at.isoformat()
    return out


def _dropbox_client() -> dropbox.Dropbox | None:
    key = os.environ.get("DROPBOX_APP_KEY")
    secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([key, secret, refresh]):
        return None
    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": key,
            "client_secret": secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return dropbox.Dropbox(response.json()["access_token"])


def upload_to_dropbox(
    dbx: dropbox.Dropbox,
    path: Path,
    *,
    name: str,
    folder: str = "",
    overwrite: bool = True,
) -> None:
    result = upload(dbx, str(path), folder, "", name, overwrite=overwrite)
    if result is None:
        raise RuntimeError(f"Dropbox upload failed for {folder}/{name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--context-days", type=int, default=DEFAULT_CONTEXT_DAYS)
    parser.add_argument("--min-history-days", type=int, default=DEFAULT_MIN_HISTORY_DAYS)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--flow-url", default=FLOW_URL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--weather-snapshot-output",
        type=Path,
        default=DEFAULT_WEATHER_SNAPSHOT_OUTPUT,
    )
    parser.add_argument("--dropbox-name", default="daily_visits_forecast.csv")
    parser.add_argument("--no-dropbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon_days < 1:
        raise ValueError("horizon-days must be >= 1")

    daily = load_daily_visits(args.flow_url)
    complete = daily.loc[daily[TARGET].notna(), "ds"]
    if complete.empty:
        raise ValueError("No complete daily ED arrival observations")
    cutoff = pd.Timestamp(complete.max()).normalize()

    # Include enough pre-context weather to calculate 7-day rolling/lagged snow features.
    weather_start = cutoff - pd.Timedelta(days=args.context_days + 14)
    print(
        f"Building weather context {weather_start.date()}..{cutoff.date()} plus current forecast",
        flush=True,
    )
    hourly_weather = build_operational_weather(
        start=weather_start,
        end=cutoff,
        forecast_days=max(args.horizon_days + 1, 8),
    )

    cutoff, history, future = build_forecast_frames(
        daily,
        hourly_weather,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
    )
    history_days = len(history)
    if history_days < VALIDATED_MIN_HISTORY_DAYS:
        print(
            f"WARNING: current contiguous target history is {history_days} days, below the "
            f"{VALIDATED_MIN_HISTORY_DAYS}-day research-validation threshold",
            flush=True,
        )
    print(
        f"Daily target cutoff={cutoff.date()} history_days={history_days} "
        f"future={future['ds'].min().date()}..{future['ds'].max().date()} "
        f"feature_set={PROMOTED_WEATHER_FEATURE_SET}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}", flush=True)
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )
    forecast = run_daily_forecast(
        pipeline,
        history,
        future,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
    )
    generated_at = pd.Timestamp.now(tz="UTC")
    output = format_output(
        forecast,
        future,
        cutoff=cutoff,
        history_days=history_days,
        generated_at=generated_at,
    )
    weather_snapshot = build_weather_snapshot(
        future,
        cutoff=cutoff,
        generated_at=generated_at,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.weather_snapshot_output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    weather_snapshot.to_csv(args.weather_snapshot_output, index=False)
    print(f"Wrote {len(output)} forecast rows to {args.output}", flush=True)
    print(
        f"Wrote {len(weather_snapshot)} exact future-covariate rows to "
        f"{args.weather_snapshot_output}",
        flush=True,
    )
    print(output.to_string(index=False), flush=True)

    if not args.no_dropbox:
        dbx = _dropbox_client()
        if dbx is None:
            print("Dropbox credentials not present; leaving forecast as local CSV", flush=True)
        else:
            upload_to_dropbox(dbx, args.output, name=args.dropbox_name, overwrite=True)
            stamp = generated_at.strftime("%Y%m%dT%H%M%SZ")
            upload_to_dropbox(
                dbx,
                args.output,
                folder="daily_visits_forecast_snapshots",
                name=f"daily_visits_forecast_{stamp}.csv",
                overwrite=False,
            )
            upload_to_dropbox(
                dbx,
                args.weather_snapshot_output,
                folder="weather_forecast_snapshots",
                name=f"daily_visits_weather_{stamp}.csv",
                overwrite=False,
            )
            print("Archived forecast and weather snapshots to Dropbox", flush=True)


if __name__ == "__main__":
    main()
