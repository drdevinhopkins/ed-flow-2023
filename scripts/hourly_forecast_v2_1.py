#!/usr/bin/env python3
"""Generate additive hourly Chronos-2 forecast v2.1 with explainability.

This script leaves ``hourly_forecast_v2.py`` and ``forecast-v2.csv`` untouched. It uses
exactly the same validated target/horizon routing as v2, but also retains the history-only
Chronos baseline for every future target/hour and reports the routed scenario delta:

    feature_effect = routed_forecast - baseline_forecast

The delta is an associational scenario contrast, not a causal effect. Positive values mean
the routed feature context raises the forecast relative to the history-only baseline;
negative values mean it lowers the forecast.

The only published CSV from this script is ``forecast-v2.1.csv``.
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
import hourly_forecast_v2 as v2
from hourly_feature_routing import (
    FLOW_TARGETS,
    ROUTING_VERSION,
    horizon_band,
    scenario_for,
    scenarios_needed,
)
from staffing_features import build_schedule_feature_frames
from utils import upload

OUTPUT_PATH = Path("forecast-v2.1.csv")
EXPLANATION_VERSION = "routed-baseline-delta-v1"
EXPLANATION_METHOD = "routed_scenario_minus_history_only_baseline"
EXPLANATION_CAVEAT = "Associational scenario contrast; not a causal effect."

SCENARIO_LABELS = {
    "baseline": "history-only baseline",
    "calendar_demand": "calendar demand context",
    "weather_raw": "weather context",
    "weather_raw_plus_snow": "weather and snow/recovery context",
    "staffing_current": "current staffing context",
    "staffing_structure_effects": "staffing structure and physician-effect context",
}


def _scenario_row(
    frame: pd.DataFrame,
    *,
    target: str,
    stamp: pd.Timestamp,
    scenario: str,
) -> pd.Series:
    match = frame.loc[
        frame["target_name"].eq(target) & frame["ds"].eq(stamp)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Expected one row for {target} at {stamp} in scenario={scenario}; "
            f"got {len(match)}"
        )
    return match.iloc[0]


def _explain_effect(
    *,
    scenario: str,
    baseline_forecast: float,
    routed_forecast: float,
) -> dict[str, object]:
    effect = float(routed_forecast - baseline_forecast)
    if abs(baseline_forecast) < 1e-9:
        effect_pct: float | None = None
    else:
        effect_pct = effect / abs(baseline_forecast) * 100.0

    if effect > 1e-9:
        direction = "higher"
        verb = "raises"
    elif effect < -1e-9:
        direction = "lower"
        verb = "lowers"
    else:
        direction = "unchanged"
        verb = "does not change"
        effect = 0.0
        if effect_pct is not None:
            effect_pct = 0.0

    label = SCENARIO_LABELS.get(scenario, scenario.replace("_", " "))
    if scenario == "baseline":
        text = (
            "History-only Chronos baseline selected for this target/horizon; "
            "the routed feature-family contribution is 0.0."
        )
    else:
        text = (
            f"{label.capitalize()} {verb} the forecast by {abs(effect):.1f} "
            f"versus the history-only Chronos baseline "
            f"({baseline_forecast:.1f} -> {routed_forecast:.1f})."
        )

    return {
        "baseline_forecast": float(baseline_forecast),
        "feature_effect": effect,
        "feature_effect_pct": effect_pct,
        "explanation_family": final_bt.FAMILY[scenario],
        "explanation_direction": direction,
        "explanation_method": EXPLANATION_METHOD,
        "explanation_caveat": EXPLANATION_CAVEAT,
        "explanation_text": text,
        "explanation_version": EXPLANATION_VERSION,
    }


def _observed_rows_v2_1(
    flow: pd.DataFrame,
    anomaly_ranges: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    generated_at_utc: str,
    allow_weather: bool,
) -> list[dict[str, object]]:
    rows = v2._observed_rows(
        flow,
        anomaly_ranges,
        cutoff=cutoff,
        generated_at_utc=generated_at_utc,
        allow_weather=allow_weather,
    )
    for row in rows:
        row.update(
            {
                "baseline_forecast": None,
                "feature_effect": None,
                "feature_effect_pct": None,
                "explanation_family": None,
                "explanation_direction": None,
                "explanation_method": None,
                "explanation_caveat": None,
                "explanation_text": None,
                "explanation_version": EXPLANATION_VERSION,
            }
        )
    return rows


def build_forecast_v2_1(
    pipeline: Chronos2Pipeline,
    *,
    allow_weather: bool,
) -> pd.DataFrame:
    flow = staffing_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)
    anomaly_ranges = v2._load_anomaly_ranges()

    cutoff = pd.Timestamp(flow["ds"].max()).floor("h")
    calendar = final_bt.build_calendar_frame(flow, [cutoff], v2.HORIZON)

    # Baseline is explicitly included because it is the reference for every explanation.
    # With the current v2 routing it is already among scenarios_needed(), so v2.1 adds no
    # extra model inference relative to generating the routed scenario set.
    scenarios = set(scenarios_needed(allow_weather=allow_weather)) | {"baseline"}
    scenario_forecasts: dict[str, pd.DataFrame] = {}
    for scenario in sorted(scenarios):
        print(f"Forecast v2.1: scenario={scenario}")
        history, future = final_bt.scenario_frames(
            scenario,
            flow=flow,
            staffing=staffing,
            weather=weather,
            shifts=shifts,
            schedule_frames=schedule_frames,
            calendar=calendar,
            cutoff=cutoff,
            horizon=v2.HORIZON,
            max_history_days=v2.MAX_HISTORY_DAYS,
            effect_min_hours=v2.EFFECT_MIN_HOURS,
            effect_shrinkage_hours=v2.EFFECT_SHRINKAGE_HOURS,
        )
        scenario_forecasts[scenario] = v2._predict_scenario(
            pipeline, history, future
        )

    baseline_frame = scenario_forecasts["baseline"]
    generated_at_utc = pd.Timestamp.now(tz="UTC").isoformat()
    rows = _observed_rows_v2_1(
        flow,
        anomaly_ranges,
        cutoff=cutoff,
        generated_at_utc=generated_at_utc,
        allow_weather=allow_weather,
    )

    expected_hours = pd.date_range(
        cutoff + pd.Timedelta(hours=1), periods=v2.HORIZON, freq="h"
    )
    for target in FLOW_TARGETS:
        for horizon_hour, stamp in enumerate(expected_hours, start=1):
            scenario = scenario_for(
                target, horizon_hour, allow_weather=allow_weather
            )
            routed = _scenario_row(
                scenario_forecasts[scenario],
                target=target,
                stamp=stamp,
                scenario=scenario,
            )
            baseline = _scenario_row(
                baseline_frame,
                target=target,
                stamp=stamp,
                scenario="baseline",
            )

            forecast_value = float(routed["predictions"])
            baseline_value = float(baseline["predictions"])
            explanation = _explain_effect(
                scenario=scenario,
                baseline_forecast=baseline_value,
                routed_forecast=forecast_value,
            )
            anomaly = v2._anomaly_values(
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
                    "forecast_lower": float(routed["0.2"]),
                    "forecast_upper": float(routed["0.8"]),
                    "baseline_forecast": explanation["baseline_forecast"],
                    "feature_effect": explanation["feature_effect"],
                    "feature_effect_pct": explanation["feature_effect_pct"],
                    "explanation_family": explanation["explanation_family"],
                    "explanation_direction": explanation["explanation_direction"],
                    "explanation_method": explanation["explanation_method"],
                    "explanation_caveat": explanation["explanation_caveat"],
                    "explanation_text": explanation["explanation_text"],
                    "explanation_version": explanation["explanation_version"],
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
    expected_rows = len(FLOW_TARGETS) * (v2.HISTORY_HOURS + v2.HORIZON)
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Incomplete forecast-v2.1.csv: expected {expected_rows} rows, "
            f"got {len(result)}"
        )
    if result[["ds", "target_name"]].duplicated().any():
        raise RuntimeError("Duplicate ds/target_name rows in forecast-v2.1.csv")

    observed = result["row_type"].eq("observed")
    future = result["row_type"].eq("forecast")
    if result.loc[future, "baseline_forecast"].isna().any():
        raise RuntimeError("Missing baseline forecast in v2.1 future rows")
    if result.loc[future, "feature_effect"].isna().any():
        raise RuntimeError("Missing feature effect in v2.1 future rows")
    if result.loc[
        future,
        [
            "explanation_family",
            "explanation_direction",
            "explanation_method",
            "explanation_caveat",
            "explanation_text",
            "explanation_version",
        ],
    ].isna().any().any():
        raise RuntimeError("Missing explainability metadata in v2.1 future rows")

    calculated = (
        result.loc[future, "forecast"] - result.loc[future, "baseline_forecast"]
    )
    delta_error = (calculated - result.loc[future, "feature_effect"]).abs()
    if delta_error.gt(1e-8).any():
        raise RuntimeError("feature_effect does not equal forecast - baseline_forecast")

    baseline_selected = future & result["scenario"].eq("baseline")
    if result.loc[baseline_selected, "feature_effect"].abs().gt(1e-8).any():
        raise RuntimeError("Baseline-routed rows must have zero feature_effect")
    if result.loc[observed, "baseline_forecast"].notna().any():
        raise RuntimeError("Observed rows must not carry future baseline forecasts")

    return result


def _upload_to_dropbox(path: Path) -> None:
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([app_key, app_secret, refresh_token]):
        raise RuntimeError(
            "Dropbox credentials are required to publish forecast-v2.1.csv"
        )

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
    allow_weather = v2._env_flag(
        "CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING", default=False
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {base.MODEL_ID} on {device}")
    print(
        f"Forecast=v2.1; routing version={ROUTING_VERSION}; "
        f"weather={'enabled' if allow_weather else 'disabled'}; "
        f"explanation={EXPLANATION_VERSION}"
    )

    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        base.MODEL_ID, device_map=device
    )
    output = build_forecast_v2_1(pipeline, allow_weather=allow_weather)
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH} ({len(output)} rows)")
    _upload_to_dropbox(OUTPUT_PATH)
    print(f"Uploaded {OUTPUT_PATH} to Dropbox")


if __name__ == "__main__":
    main()
