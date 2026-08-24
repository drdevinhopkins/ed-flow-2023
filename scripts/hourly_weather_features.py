#!/usr/bin/env python3
"""Leakage-aware hourly weather feature engineering for native Chronos-2.

The daily-arrivals weather work showed that raw meteorology plus snow/recovery state
was more useful than indiscriminately adding every anomaly/shock feature.  This module
translates those concepts to hourly resolution for the ED flow model.

Design rules
------------
* Raw weather stays available as known future covariates.
* Seasonal baselines are fit only from weather at/before the forecast cutoff.
* Future rolling features may use earlier forecast hours because those values are known
  at prediction time; they never use target observations after the cutoff.
* Snow/recovery and freeze/thaw state are represented explicitly at hourly resolution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_HOURLY_WEATHER_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
]

THERMAL_COLUMNS = [
    "temperature_anomaly",
    "temperature_anomaly_z",
    "apparent_temperature_anomaly",
    "apparent_temperature_anomaly_z",
    "wind_anomaly_z",
    "gust_anomaly_z",
    "cold_anomaly_event",
    "warm_anomaly_event",
    "cold_windy_event",
    "thermal_cold_stress",
    "thermal_heat_stress",
]

SNOW_RECOVERY_COLUMNS = [
    "snowfall_6h",
    "snowfall_12h",
    "snowfall_24h",
    "snowfall_48h",
    "snowfall_72h",
    "precipitation_6h",
    "precipitation_24h",
    "hours_since_snow_capped",
    "hours_since_major_snow_capped",
    "major_snow_24h_event",
    "post_major_snow_6_24h",
    "post_major_snow_24_48h",
    "post_major_snow_48_72h",
    "snow_wind_event",
    "freeze_thaw_transition",
    "post_snow_thaw",
    "refreeze_after_thaw",
]

STORM_COLUMNS = [
    "temperature_change_3h",
    "temperature_change_6h",
    "temperature_change_12h",
    "pressure_change_3h",
    "pressure_change_6h",
    "pressure_change_12h",
    "gust_change_3h",
    "rapid_cold_snap_6h",
    "rapid_warmup_6h",
    "pressure_drop_event",
    "storm_severity_index",
    "travel_disruption_index",
]

RAW_PLUS_SNOW_HOURLY_COLUMNS = list(
    dict.fromkeys([*RAW_HOURLY_WEATHER_COLUMNS, *SNOW_RECOVERY_COLUMNS])
)
ENGINEERED_HOURLY_WEATHER_COLUMNS = list(
    dict.fromkeys(
        [
            *RAW_HOURLY_WEATHER_COLUMNS,
            *SNOW_RECOVERY_COLUMNS,
            *THERMAL_COLUMNS,
            *STORM_COLUMNS,
        ]
    )
)

CLIMATOLOGY_COLUMNS = [
    "temperature_2m",
    "apparent_temperature",
    "wind_speed_10m",
    "wind_gusts_10m",
]


def _numeric_weather(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["ds"] = pd.to_datetime(out["ds"], format="mixed", errors="coerce")
    if getattr(out["ds"].dt, "tz", None) is not None:
        out["ds"] = out["ds"].dt.tz_convert("America/Montreal").dt.tz_localize(None)
    out["ds"] = out["ds"].dt.floor("h")
    out = out.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    for column in RAW_HOURLY_WEATHER_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")

    # Zero is the physically meaningful fallback for missing precip/snow amounts.
    for column in ["precipitation", "rain", "snowfall", "snow_depth"]:
        out[column] = out[column].fillna(0.0)

    continuous = [
        "temperature_2m",
        "apparent_temperature",
        "wind_speed_10m",
        "wind_gusts_10m",
        "relative_humidity_2m",
        "pressure_msl",
        "cloud_cover",
    ]
    out[continuous] = out[continuous].ffill().bfill()
    return out.reset_index(drop=True)


def fit_hourly_climatology(
    history_weather: pd.DataFrame,
    *,
    trailing_days: int = 60,
) -> pd.DataFrame:
    """Fit recent seasonal/hour-of-day baselines using only past weather.

    A trailing local-hour climatology adapts to Montreal seasonality without requiring
    multiple complete historical years.  The same cutoff-fitted baseline is applied to
    both the historical context and known future weather rows.
    """
    weather = _numeric_weather(history_weather)
    if len(weather) < 24 * 28:
        raise ValueError("At least 28 days of hourly weather are required for climatology")
    cutoff = weather["ds"].max()
    start = cutoff - pd.Timedelta(days=trailing_days)
    recent = weather.loc[weather["ds"] >= start].copy()
    recent["hour"] = recent["ds"].dt.hour

    records: list[dict[str, float]] = []
    for hour, group in recent.groupby("hour"):
        record: dict[str, float] = {"hour": int(hour)}
        for column in CLIMATOLOGY_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce")
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            record[f"{column}__mean"] = mean
            record[f"{column}__std"] = std
        records.append(record)
    climate = pd.DataFrame(records).sort_values("hour")
    if climate["hour"].nunique() != 24:
        raise ValueError("Hourly climatology does not cover all 24 local hours")
    return climate


def _hours_since(mask: pd.Series, cap: int) -> pd.Series:
    values = mask.fillna(False).astype(bool).to_numpy()
    result = np.full(len(values), float(cap), dtype="float64")
    last = None
    for idx, active in enumerate(values):
        if active:
            last = idx
            result[idx] = 0.0
        elif last is not None:
            result[idx] = float(min(idx - last, cap))
    return pd.Series(result, index=mask.index)


def add_hourly_weather_features(
    weather: pd.DataFrame,
    climatology: pd.DataFrame,
) -> pd.DataFrame:
    out = _numeric_weather(weather)
    out["hour"] = out["ds"].dt.hour
    out = out.merge(climatology, on="hour", how="left")

    anomaly_map = {
        "temperature_2m": "temperature",
        "apparent_temperature": "apparent_temperature",
        "wind_speed_10m": "wind",
        "wind_gusts_10m": "gust",
    }
    for source, prefix in anomaly_map.items():
        mean = out[f"{source}__mean"]
        std = out[f"{source}__std"].replace(0, 1.0)
        out[f"{prefix}_anomaly"] = out[source] - mean
        out[f"{prefix}_anomaly_z"] = (out[source] - mean) / std

    # Thermal state.
    out["cold_anomaly_event"] = (out["temperature_anomaly_z"] <= -1.5).astype(float)
    out["warm_anomaly_event"] = (out["temperature_anomaly_z"] >= 1.5).astype(float)
    out["cold_windy_event"] = (
        (out["temperature_anomaly_z"] <= -1.0)
        & ((out["wind_anomaly_z"] >= 1.0) | (out["wind_gusts_10m"] >= 40.0))
    ).astype(float)
    out["thermal_cold_stress"] = (
        np.maximum(-out["temperature_anomaly_z"], 0.0)
        * (1.0 + np.maximum(out["wind_anomaly_z"], 0.0))
    )
    out["thermal_heat_stress"] = np.maximum(out["apparent_temperature_anomaly_z"], 0.0)

    # Rolling exposure.  The current hour is included, which is appropriate for known
    # future weather: at a forecast cutoff the entire future weather path is available.
    for hours in [6, 12, 24, 48, 72]:
        out[f"snowfall_{hours}h"] = out["snowfall"].rolling(hours, min_periods=1).sum()
    for hours in [6, 24]:
        out[f"precipitation_{hours}h"] = out["precipitation"].rolling(hours, min_periods=1).sum()

    snow_hour = out["snowfall"] >= 0.2
    major_snow = out["snowfall_24h"] >= 5.0
    out["hours_since_snow_capped"] = _hours_since(snow_hour, 72)
    out["hours_since_major_snow_capped"] = _hours_since(major_snow, 96)
    out["major_snow_24h_event"] = major_snow.astype(float)
    since_major = out["hours_since_major_snow_capped"]
    out["post_major_snow_6_24h"] = ((since_major >= 6) & (since_major < 24)).astype(float)
    out["post_major_snow_24_48h"] = ((since_major >= 24) & (since_major < 48)).astype(float)
    out["post_major_snow_48_72h"] = ((since_major >= 48) & (since_major < 72)).astype(float)
    out["snow_wind_event"] = (
        (out["snowfall_6h"] >= 1.0) & (out["wind_gusts_10m"] >= 35.0)
    ).astype(float)

    # Surface-state transitions.
    prev_temp = out["temperature_2m"].shift(1)
    freeze_to_thaw = (prev_temp <= 0.0) & (out["temperature_2m"] > 0.0)
    thaw_to_freeze = (prev_temp > 0.0) & (out["temperature_2m"] <= 0.0)
    out["freeze_thaw_transition"] = (freeze_to_thaw | thaw_to_freeze).astype(float)
    recent_snow = out["snowfall_24h"] >= 1.0
    out["post_snow_thaw"] = (recent_snow & (out["temperature_2m"] > 0.0)).astype(float)
    recent_thaw = out["post_snow_thaw"].rolling(12, min_periods=1).max().shift(1).fillna(0) > 0
    out["refreeze_after_thaw"] = (recent_thaw & (out["temperature_2m"] <= 0.0)).astype(float)

    # Short-term shocks.
    for hours in [3, 6, 12]:
        out[f"temperature_change_{hours}h"] = out["temperature_2m"] - out["temperature_2m"].shift(hours)
        out[f"pressure_change_{hours}h"] = out["pressure_msl"] - out["pressure_msl"].shift(hours)
    out["gust_change_3h"] = out["wind_gusts_10m"] - out["wind_gusts_10m"].shift(3)
    out["rapid_cold_snap_6h"] = (out["temperature_change_6h"] <= -6.0).astype(float)
    out["rapid_warmup_6h"] = (out["temperature_change_6h"] >= 6.0).astype(float)
    out["pressure_drop_event"] = (out["pressure_change_6h"] <= -6.0).astype(float)

    precip_scale = np.clip(out["precipitation_6h"] / 10.0, 0.0, 3.0)
    gust_scale = np.clip(out["wind_gusts_10m"] / 50.0, 0.0, 3.0)
    pressure_scale = np.clip(-out["pressure_change_6h"] / 8.0, 0.0, 3.0)
    snow_scale = np.clip(out["snowfall_24h"] / 8.0, 0.0, 3.0)
    out["storm_severity_index"] = precip_scale + gust_scale + pressure_scale + snow_scale
    out["travel_disruption_index"] = (
        snow_scale
        + np.clip(out["snow_depth"] * 10.0, 0.0, 2.0)
        + np.clip(out["wind_gusts_10m"] / 45.0, 0.0, 2.0)
        + out["freeze_thaw_transition"]
    )

    # Early lag rows are historical context; use neutral values rather than dropping rows.
    for column in [*THERMAL_COLUMNS, *SNOW_RECOVERY_COLUMNS, *STORM_COLUMNS]:
        out[column] = pd.to_numeric(out[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[column] = out[column].fillna(0.0).astype("float64")

    return out.drop(
        columns=[
            "hour",
            *[f"{column}__mean" for column in CLIMATOLOGY_COLUMNS],
            *[f"{column}__std" for column in CLIMATOLOGY_COLUMNS],
        ],
        errors="ignore",
    )


def weather_columns_for_scenario(scenario: str) -> list[str]:
    mapping = {
        "raw_weather": RAW_HOURLY_WEATHER_COLUMNS,
        "raw_plus_snow": RAW_PLUS_SNOW_HOURLY_COLUMNS,
        "raw_plus_thermal": list(dict.fromkeys([*RAW_HOURLY_WEATHER_COLUMNS, *THERMAL_COLUMNS])),
        "raw_plus_storm": list(dict.fromkeys([*RAW_HOURLY_WEATHER_COLUMNS, *STORM_COLUMNS])),
        "engineered_weather": ENGINEERED_HOURLY_WEATHER_COLUMNS,
    }
    if scenario not in mapping:
        raise KeyError(f"Unknown hourly weather scenario: {scenario}")
    return mapping[scenario]
