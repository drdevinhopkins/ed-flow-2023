#!/usr/bin/env python3
"""Generate the additive target/horizon-routed hourly Chronos-2 forecast.

This script is intentionally independent from ``chronos_forecast.py``. It does not read,
modify, replace, or upload any existing production forecast CSV. Its only output is
``forecast-v2.csv`` containing the six validated hourly flow targets for the next 24h.

Weather-winning routes remain disabled by default because the retrospective hourly
weather validation used revised/realized weather rather than archived forecast-time
snapshots. Set ``CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=1`` to opt in later.
"""

from __future__ import annotations

import os
from pathlib import Path

import dropbox
import pandas as pd
import requests
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_final_features as final_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from hourly_feature_routing import (
    FLOW_TARGETS,
    ROUTING_VERSION,
    horizon_band,
    scenario_for,
    scenarios_needed,
)
from staffing_features import build_schedule_feature_frames
from utils import upload

HORIZON = 24
MAX_HISTORY_DAYS = 365
EFFECT_MIN_HOURS = 24
EFFECT_SHRINKAGE_HOURS = 72.0
OUTPUT_PATH = Path("forecast-v2.csv")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _predict_scenario(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "prediction_length": HORIZON,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": list(FLOW_TARGETS),
        "quantile_levels": [0.2, 0.5, 0.8],
    }
    if future is not None:
        kwargs["future_df"] = future

    result = pipeline.predict_df(history, **kwargs).copy()
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")

    for column in ["0.2", "0.8"]:
        if column not in result.columns:
            result[column] = result["predictions"]

    result["ds"] = pd.to_datetime(result["ds"], format="mixed", errors="coerce")
    return result[["ds", "target_name", "predictions", "0.2", "0.8"]]


def build_forecast_v2(
    pipeline: Chronos2Pipeline,
    *,
    allow_weather: bool,
) -> pd.DataFrame:
    flow = staffing_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)

    cutoff = pd.Timestamp(flow["ds"].max()).floor("h")
    calendar = final_bt.build_calendar_frame(flow, [cutoff], HORIZON)

    scenario_forecasts: dict[str, pd.DataFrame] = {}
    for scenario in sorted(scenarios_needed(allow_weather=allow_weather)):
        print(f"Forecast v2: scenario={scenario}")
        history, future = final_bt.scenario_frames(
            scenario,
            flow=flow,
            staffing=staffing,
            weather=weather,
            shifts=shifts,
            schedule_frames=schedule_frames,
            calendar=calendar,
            cutoff=cutoff,
            horizon=HORIZON,
            max_history_days=MAX_HISTORY_DAYS,
            effect_min_hours=EFFECT_MIN_HOURS,
            effect_shrinkage_hours=EFFECT_SHRINKAGE_HOURS,
        )
        scenario_forecasts[scenario] = _predict_scenario(pipeline, history, future)

    generated_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    expected_hours = pd.date_range(
        cutoff + pd.Timedelta(hours=1), periods=HORIZON, freq="h"
    )
    rows: list[dict[str, object]] = []

    for target in FLOW_TARGETS:
        for horizon_hour, stamp in enumerate(expected_hours, start=1):
            scenario = scenario_for(
                target, horizon_hour, allow_weather=allow_weather
            )
            scenario_frame = scenario_forecasts[scenario]
            match = scenario_frame.loc[
                scenario_frame["target_name"].eq(target)
                & scenario_frame["ds"].eq(stamp)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one v2 row for {target} h={horizon_hour} "
                    f"scenario={scenario}; got {len(match)}"
                )

            row = match.iloc[0]
            rows.append(
                {
                    "ds": stamp,
                    "target_name": target,
                    "forecast": float(row["predictions"]),
                    "forecast_lower": float(row["0.2"]),
                    "forecast_upper": float(row["0.8"]),
                    "horizon_hour": horizon_hour,
                    "horizon_band": horizon_band(horizon_hour),
                    "scenario": scenario,
                    "feature_family": final_bt.FAMILY[scenario],
                    "forecast_origin": cutoff,
                    "generated_at_utc": generated_at_utc,
                    "routing_version": ROUTING_VERSION,
                    "weather_routing_enabled": allow_weather,
                }
            )

    result = pd.DataFrame(rows).sort_values(
        ["ds", "target_name"], ignore_index=True
    )
    expected_rows = len(FLOW_TARGETS) * HORIZON
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Incomplete forecast-v2.csv: expected {expected_rows} rows, got {len(result)}"
        )
    if result[["ds", "target_name"]].duplicated().any():
        raise RuntimeError("Duplicate ds/target_name rows in forecast-v2.csv")
    if result[["forecast", "forecast_lower", "forecast_upper"]].isna().any().any():
        raise RuntimeError("Missing routed forecast values in forecast-v2.csv")
    return result


def _upload_to_dropbox(path: Path) -> None:
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([app_key, app_secret, refresh_token]):
        raise RuntimeError("Dropbox credentials are required to publish forecast-v2.csv")

    token_response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        },
        timeout=30,
    )
    token_response.raise_for_status()
    access_token = token_response.json()["access_token"]
    dbx = dropbox.Dropbox(access_token)
    upload(dbx, str(path), "", "", path.name, overwrite=True)


def main() -> None:
    allow_weather = _env_flag(
        "CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING", default=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {base.MODEL_ID} on {device}")
    print(
        f"Routing version={ROUTING_VERSION}; "
        f"weather={'enabled' if allow_weather else 'disabled'}"
    )

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        base.MODEL_ID, device_map=device
    )
    forecast = build_forecast_v2(pipeline, allow_weather=allow_weather)
    forecast.to_csv(OUTPUT_PATH, index=False)
    _upload_to_dropbox(OUTPUT_PATH)

    print(f"Wrote and uploaded {OUTPUT_PATH} ({len(forecast)} rows)")
    print(
        forecast.groupby(["target_name", "scenario"])
        .size()
        .rename("hours")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
