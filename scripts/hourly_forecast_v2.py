#!/usr/bin/env python3
"""Generate the additive target/horizon-routed hourly Chronos-2 forecast.

This script is intentionally independent from ``chronos_forecast.py``. It does not read,
modify, replace, or upload any existing production forecast CSV. Its only output is
``forecast-v2.csv``.

For each configured flow target, the output contains the latest 24 observed hourly values
ending at the forecast origin plus the next 24 routed Chronos-2 forecasts. Observed values
and forecasts live in separate ``actual`` and ``forecast`` columns so Power BI can plot
the two series independently without reshaping the file.

Observed and forecast rows both carry the same anomaly reference intervals used by
``ED_Hourly_Forecasts_Anomalies_v1.0.csv``. Observed colours are directional for ED flow:
anomalously low actuals remain green, while only actuals above the upper anomaly bound
are red. Forecast colours retain the existing two-sided anomaly convention.

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
HISTORY_HOURS = 24
MAX_HISTORY_DAYS = 365
EFFECT_MIN_HOURS = 24
EFFECT_SHRINKAGE_HOURS = 72.0
OUTPUT_PATH = Path("forecast-v2.csv")
ANOMALY_RANGES_URL = (
    "https://www.dropbox.com/scl/fi/fjz0am427gw35sz7l994m/"
    "anomaly_detection_ranges.csv?rlkey=lib9w0jz2zei5n566jv76o7ol&raw=1"
)

ANOMALY_TARGET_ALIASES = {
    "Total_TBS": "total_tbs",
    "POD_TBS": "pod_tbs",
    "Vertical_TBS": "vert_tbs",
    "TTStr": "TTStr",
    "Overflow": "overflow",
    "WAITINGADM": "WAITINGADM",
    "TRG_HALLWAY1": "TRG_HALLWAY1",
    "TRG_HALLWAY_TBS": "TRG_HALLWAY_TBS",
}

RED = "#D13438"
AMBER = "#FFB900"
GREEN = "#107C10"


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


def _load_anomaly_ranges() -> pd.DataFrame:
    frame = pd.read_csv(ANOMALY_RANGES_URL)
    if "ds" not in frame.columns:
        raise ValueError("Anomaly range file is missing ds")
    frame["ds"] = pd.to_datetime(
        frame["ds"], format="mixed", errors="coerce"
    ).dt.floor("h")
    frame = frame.dropna(subset=["ds"]).drop_duplicates("ds", keep="last")
    return frame


def _anomaly_columns(frame: pd.DataFrame, target: str) -> tuple[str, str, str]:
    prefixes = [ANOMALY_TARGET_ALIASES[target], target]
    for prefix in prefixes:
        columns = (
            f"{prefix}_yhat",
            f"{prefix}_yhat_lower",
            f"{prefix}_yhat_upper",
        )
        if all(column in frame.columns for column in columns):
            return columns
    raise ValueError(
        f"Anomaly range file has no expected yhat columns for {target}; "
        f"tried prefixes {prefixes}"
    )


def _anomaly_values(
    anomaly_ranges: pd.DataFrame,
    *,
    target: str,
    stamp: pd.Timestamp,
    value: float,
    historical: bool = False,
) -> dict[str, object]:
    yhat_col, lower_col, upper_col = _anomaly_columns(anomaly_ranges, target)
    match = anomaly_ranges.loc[
        anomaly_ranges["ds"].eq(stamp), [yhat_col, lower_col, upper_col]
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one anomaly-range row for {target} at {stamp}; got {len(match)}"
        )

    yhat = pd.to_numeric(match.iloc[0][yhat_col], errors="coerce")
    lower = pd.to_numeric(match.iloc[0][lower_col], errors="coerce")
    upper = pd.to_numeric(match.iloc[0][upper_col], errors="coerce")
    if pd.isna(yhat) or pd.isna(lower) or pd.isna(upper):
        raise ValueError(f"Missing anomaly interval for {target} at {stamp}")

    yhat = float(yhat)
    lower = float(lower)
    upper = float(upper)
    is_anomaly = value < lower or value > upper

    if historical:
        # Directional operational colour: unusually low congestion is favorable.
        if value > upper:
            colour = RED
        elif value > yhat:
            colour = AMBER
        else:
            colour = GREEN
    elif is_anomaly:
        colour = RED
    elif value > yhat:
        colour = AMBER
    else:
        colour = GREEN

    return {
        "anomaly_yhat": yhat,
        "anomaly_yhat_lower": lower,
        "anomaly_yhat_upper": upper,
        "is_anomaly": "yes" if is_anomaly else "no",
        "colour": colour,
    }


def _observed_rows(
    flow: pd.DataFrame,
    anomaly_ranges: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    generated_at_utc: str,
    allow_weather: bool,
) -> list[dict[str, object]]:
    history = flow.loc[
        flow["ds"].le(cutoff), ["ds", *FLOW_TARGETS]
    ].tail(HISTORY_HOURS)
    if len(history) != HISTORY_HOURS:
        raise ValueError(
            f"Expected {HISTORY_HOURS} recent observed hours; got {len(history)}"
        )

    rows: list[dict[str, object]] = []
    for _, source in history.iterrows():
        stamp = pd.Timestamp(source["ds"]).floor("h")
        for target in FLOW_TARGETS:
            actual = pd.to_numeric(source[target], errors="coerce")
            if pd.isna(actual):
                raise ValueError(f"Missing observed {target} at {stamp}")
            actual = float(actual)
            anomaly = _anomaly_values(
                anomaly_ranges,
                target=target,
                stamp=stamp,
                value=actual,
                historical=True,
            )
            rows.append(
                {
                    "ds": stamp,
                    "target_name": target,
                    "actual": actual,
                    "forecast": None,
                    "forecast_lower": None,
                    "forecast_upper": None,
                    "anomaly_yhat": anomaly["anomaly_yhat"],
                    "anomaly_yhat_lower": anomaly["anomaly_yhat_lower"],
                    "anomaly_yhat_upper": anomaly["anomaly_yhat_upper"],
                    "actual_anomaly": anomaly["is_anomaly"],
                    "actual_colour": anomaly["colour"],
                    "forecast_anomaly": None,
                    "forecast_colour": None,
                    "row_type": "observed",
                    "horizon_hour": None,
                    "horizon_band": None,
                    "scenario": None,
                    "feature_family": None,
                    "forecast_origin": cutoff,
                    "generated_at_utc": generated_at_utc,
                    "routing_version": ROUTING_VERSION,
                    "weather_routing_enabled": allow_weather,
                }
            )
    return rows


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
    anomaly_ranges = _load_anomaly_ranges()

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
    rows = _observed_rows(
        flow,
        anomaly_ranges,
        cutoff=cutoff,
        generated_at_utc=generated_at_utc,
        allow_weather=allow_weather,
    )

    expected_hours = pd.date_range(
        cutoff + pd.Timedelta(hours=1), periods=HORIZON, freq="h"
    )
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

            source = match.iloc[0]
            forecast_value = float(source["predictions"])
            anomaly = _anomaly_values(
                anomaly_ranges,
                target=target,
                stamp=stamp,
                value=forecast_value,
            )
            rows.append(
                {
                    "ds": stamp,
                    "target_name": target,
                    "actual": None,
                    "forecast": forecast_value,
                    "forecast_lower": float(source["0.2"]),
                    "forecast_upper": float(source["0.8"]),
                    "anomaly_yhat": anomaly["anomaly_yhat"],
                    "anomaly_yhat_lower": anomaly["anomaly_yhat_lower"],
                    "anomaly_yhat_upper": anomaly["anomaly_yhat_upper"],
                    "actual_anomaly": None,
                    "actual_colour": None,
                    "forecast_anomaly": anomaly["is_anomaly"],
                    "forecast_colour": anomaly["colour"],
                    "row_type": "forecast",
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
    expected_rows = len(FLOW_TARGETS) * (HISTORY_HOURS + HORIZON)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Incomplete forecast-v2.csv: expected {expected_rows} rows, got {len(result)}"
        )
    if result[["ds", "target_name"]].duplicated().any():
        raise RuntimeError("Duplicate ds/target_name rows in forecast-v2.csv")

    observed = result["row_type"].eq("observed")
    future = result["row_type"].eq("forecast")
    if (
        result.loc[observed, "actual"].isna().any()
        or result.loc[observed, "forecast"].notna().any()
    ):
        raise RuntimeError("Observed rows must populate actual only")
    if (
        result.loc[future, "forecast"].isna().any()
        or result.loc[future, "actual"].notna().any()
    ):
        raise RuntimeError("Forecast rows must populate forecast only")
    if result[
        ["anomaly_yhat", "anomaly_yhat_lower", "anomaly_yhat_upper"]
    ].isna().any().any():
        raise RuntimeError("Missing anomaly reference intervals in forecast-v2.csv")
    if result.loc[
        observed, ["actual_anomaly", "actual_colour"]
    ].isna().any().any():
        raise RuntimeError("Missing observed anomaly annotations in forecast-v2.csv")
    if result.loc[
        future, ["forecast_anomaly", "forecast_colour"]
    ].isna().any().any():
        raise RuntimeError("Missing forecast anomaly annotations in forecast-v2.csv")
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
    output = build_forecast_v2(pipeline, allow_weather=allow_weather)
    output.to_csv(OUTPUT_PATH, index=False)
    _upload_to_dropbox(OUTPUT_PATH)

    print(f"Wrote and uploaded {OUTPUT_PATH} ({len(output)} rows)")
    print(output.groupby(["row_type", "target_name"]).size().rename("rows"))


if __name__ == "__main__":
    main()
