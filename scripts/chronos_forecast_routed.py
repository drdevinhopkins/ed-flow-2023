#!/usr/bin/env python3
"""Run the existing hourly forecast, then promote validated target/horizon routes.

The legacy ``chronos_forecast.py`` remains the source for all existing outputs and
non-routed targets. This wrapper reuses the already-loaded Chronos-2 pipeline, runs only
the validated feature-family finalists needed by the routing table, and replaces the
six canonical flow-target forecasts in the generated CSVs.

Weather winners are disabled by default because retrospective hourly weather validation
used revised/realized weather rather than archived forecast-time snapshots. Set
``CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=1`` to opt in once prospective validation is
considered sufficient.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path

import pandas as pd

import backtest_covariate_ablation as base
import backtest_hourly_final_features as final_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from hourly_feature_routing import (
    FLOW_TARGETS,
    ROUTING_VERSION,
    scenario_for,
    scenarios_needed,
)
from staffing_features import build_schedule_feature_frames

HORIZON = 24
MAX_HISTORY_DAYS = 365
EFFECT_MIN_HOURS = 24
EFFECT_SHRINKAGE_HOURS = 72.0

# The validated feature work uses canonical target names, while the legacy production
# forecast script derives four of those targets under older lowercase aliases. Keep this
# translation explicit so routed values land in both the long and wide production files.
LEGACY_TARGET_ALIASES: dict[str, str] = {
    "Total_TBS": "total_tbs",
    "POD_TBS": "pod_tbs",
    "Vertical_TBS": "vert_tbs",
    "TTStr": "TTStr",
    "Overflow": "overflow",
    "WAITINGADM": "WAITINGADM",
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _predict_scenario(pipeline, history: pd.DataFrame, future: pd.DataFrame | None) -> pd.DataFrame:
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


def _build_routed_forecast(pipeline, *, allow_weather: bool) -> pd.DataFrame:
    flow = staffing_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)

    cutoff = pd.Timestamp(flow["ds"].max()).floor("h")
    calendar = final_bt.build_calendar_frame(flow, [cutoff], HORIZON)

    scenario_forecasts: dict[str, pd.DataFrame] = {}
    for scenario in sorted(scenarios_needed(allow_weather=allow_weather)):
        print(f"Production routed forecast: scenario={scenario}")
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

    rows: list[dict[str, object]] = []
    expected_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    for target in FLOW_TARGETS:
        for horizon_hour, stamp in enumerate(expected_hours, start=1):
            scenario = scenario_for(target, horizon_hour, allow_weather=allow_weather)
            forecast = scenario_forecasts[scenario]
            match = forecast.loc[
                forecast["target_name"].eq(target) & forecast["ds"].eq(stamp)
            ]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one routed forecast row for {target} h={horizon_hour} "
                    f"scenario={scenario}; got {len(match)}"
                )
            row = match.iloc[0]
            rows.append(
                {
                    "ds": stamp,
                    "target_name": target,
                    "horizon_hour": horizon_hour,
                    "scenario": scenario,
                    "predictions": float(row["predictions"]),
                    "0.2": float(row["0.2"]),
                    "0.8": float(row["0.8"]),
                    "routing_version": ROUTING_VERSION,
                    "weather_routing_enabled": allow_weather,
                }
            )
    routed = pd.DataFrame(rows)
    if len(routed) != len(FLOW_TARGETS) * HORIZON:
        raise RuntimeError(f"Incomplete routed forecast: {len(routed)} rows")
    return routed


def _patch_chronos_long(path: Path, routed: pd.DataFrame) -> None:
    frame = pd.read_csv(path)
    frame["ds"] = pd.to_datetime(frame["ds"], format="mixed", errors="coerce")

    route = routed.copy()
    route["routed_target_name"] = route["target_name"]
    route["target_name"] = route["target_name"].map(LEGACY_TARGET_ALIASES)
    if route["target_name"].isna().any():
        raise ValueError("Missing legacy target alias while patching chronos_forecast.csv")
    route = route.rename(
        columns={
            "predictions": "routed_forecast",
            "0.2": "routed_forecast_lower",
            "0.8": "routed_forecast_upper",
            "scenario": "routed_scenario",
        }
    )[
        [
            "ds",
            "target_name",
            "routed_target_name",
            "routed_forecast",
            "routed_forecast_lower",
            "routed_forecast_upper",
            "routed_scenario",
            "routing_version",
            "weather_routing_enabled",
        ]
    ]
    frame = frame.merge(route, on=["ds", "target_name"], how="left")
    mask = frame["routed_forecast"].notna()
    expected = len(FLOW_TARGETS) * HORIZON
    if int(mask.sum()) != expected:
        raise ValueError(
            f"Expected to patch {expected} routed rows in {path}, patched {int(mask.sum())}"
        )
    if "forecast_all_vars_with_future" not in frame.columns:
        raise ValueError("chronos_forecast.csv is missing forecast_all_vars_with_future")
    frame.loc[mask, "forecast_all_vars_with_future"] = frame.loc[mask, "routed_forecast"]
    frame.to_csv(path, index=False)


def _patch_wide_output(path: Path, routed: pd.DataFrame) -> None:
    frame = pd.read_csv(path)
    frame["ds"] = pd.to_datetime(frame["ds"], format="mixed", errors="coerce")
    route_metadata: dict[str, pd.Series] = {}

    for target in FLOW_TARGETS:
        legacy_target = LEGACY_TARGET_ALIASES[target]
        target_route = routed.loc[routed["target_name"].eq(target)].set_index("ds")
        forecast_col = f"{legacy_target}_forecast"
        lower_col = f"{legacy_target}_forecast_lower"
        upper_col = f"{legacy_target}_forecast_upper"
        required = [forecast_col, lower_col, upper_col]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(
                f"Wide output is missing routed columns for {target}: {missing}"
            )

        route_series = frame["ds"].map(target_route["scenario"])
        prediction_series = frame["ds"].map(target_route["predictions"])
        lower_series = frame["ds"].map(target_route["0.2"])
        upper_series = frame["ds"].map(target_route["0.8"])
        mask = prediction_series.notna()
        if int(mask.sum()) != HORIZON:
            raise ValueError(
                f"Expected {HORIZON} future rows for {target} in wide output; "
                f"found {int(mask.sum())}"
            )

        frame.loc[mask, forecast_col] = prediction_series.loc[mask]
        frame.loc[mask, lower_col] = lower_series.loc[mask]
        frame.loc[mask, upper_col] = upper_series.loc[mask]
        route_metadata[f"{target}_forecast_route"] = route_series

        yhat = f"{legacy_target}_yhat"
        yhat_lower = f"{legacy_target}_yhat_lower"
        yhat_upper = f"{legacy_target}_yhat_upper"
        anomaly_col = f"{legacy_target}_anomaly"
        colour_col = f"{legacy_target}_colour"
        anomaly_inputs = [yhat, yhat_lower, yhat_upper, anomaly_col, colour_col]
        missing_anomaly = [column for column in anomaly_inputs if column not in frame.columns]
        if missing_anomaly:
            raise ValueError(
                f"Wide output is missing anomaly columns for {target}: {missing_anomaly}"
            )

        valid = mask & frame[yhat_lower].notna() & frame[yhat_upper].notna()
        is_anomaly = (
            (prediction_series < pd.to_numeric(frame[yhat_lower], errors="coerce"))
            | (prediction_series > pd.to_numeric(frame[yhat_upper], errors="coerce"))
        )
        frame.loc[valid, anomaly_col] = is_anomaly.loc[valid].map(
            {True: "yes", False: "no"}
        )
        center = pd.to_numeric(frame[yhat], errors="coerce")
        normal_high = valid & ~is_anomaly & prediction_series.gt(center)
        normal_low = valid & ~is_anomaly & ~prediction_series.gt(center)
        frame.loc[valid & is_anomaly, colour_col] = "#D13438"
        frame.loc[normal_high, colour_col] = "#FFB900"
        frame.loc[normal_low, colour_col] = "#107C10"

    metadata = pd.DataFrame(route_metadata, index=frame.index)
    metadata["hourly_routing_version"] = ROUTING_VERSION
    metadata["hourly_weather_routing_enabled"] = bool(
        routed["weather_routing_enabled"].iloc[0]
    )
    frame = pd.concat([frame, metadata], axis=1)
    frame.to_csv(path, index=False)


def main() -> None:
    # Execute the proven legacy workflow in-process so its loaded Chronos-2 model can be
    # reused for routed scenarios rather than loading the model a second time.
    legacy_path = Path(__file__).with_name("chronos_forecast.py")
    legacy = runpy.run_path(str(legacy_path))
    pipeline = legacy.get("pipeline")
    if pipeline is None:
        raise RuntimeError("Legacy hourly forecast did not expose the Chronos pipeline")

    allow_weather = _env_flag("CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING", default=False)
    print(
        f"Applying {ROUTING_VERSION}; "
        f"weather routing={'enabled' if allow_weather else 'disabled'}"
    )
    routed = _build_routed_forecast(pipeline, allow_weather=allow_weather)
    routed_path = Path("hourly_feature_routing.csv")
    routed.to_csv(routed_path, index=False)

    chronos_path = Path("chronos_forecast.csv")
    wide_path = Path("ED_Hourly_Forecasts_Anomalies_v1.0.csv")
    _patch_chronos_long(chronos_path, routed)
    _patch_wide_output(wide_path, routed)

    # Reuse the Dropbox session created by the legacy script and overwrite only the
    # forecast files modified above plus the routing audit table.
    dbx = legacy.get("dbx")
    upload = legacy.get("upload")
    if dbx is None or upload is None:
        raise RuntimeError("Legacy hourly forecast did not expose Dropbox upload state")
    for local_name in [str(chronos_path), str(wide_path), str(routed_path)]:
        upload(dbx, local_name, "", "", Path(local_name).name, overwrite=True)

    print("Routed hourly forecasts written and uploaded successfully")
    print(
        routed.groupby(["target_name", "scenario"]).size().rename("hours").reset_index().to_string(index=False)
    )


if __name__ == "__main__":
    main()
