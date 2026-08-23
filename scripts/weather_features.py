"""Weather feature engineering for daily JGH ED arrival forecasting.

The raw source is the existing hourly Open-Meteo ``weather.csv`` table.  This module
turns it into daily, interpretable known covariates: seasonal anomalies, abrupt changes,
accumulated exposure, post-storm recovery, freeze/thaw, and compound weather events.

Climatology is fit only on weather rows available on or before the forecast cutoff by the
backtest caller.  That avoids using future weather to define what is "normal", although
historical backtests still use realized/revised weather rather than archived forecast
snapshots and therefore measure weather-signal potential rather than a leakage-free
real-time replay.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_DAILY_COLUMNS = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "apparent_temp_mean",
    "apparent_temp_min",
    "apparent_temp_max",
    "precip_sum",
    "rain_sum",
    "snowfall_sum",
    "snow_depth_max",
    "wind_mean",
    "wind_max",
    "gust_max",
    "humidity_mean",
    "pressure_mean",
    "pressure_min",
    "pressure_max",
    "pressure_range",
    "cloud_mean",
]

CLIMATOLOGY_SOURCE_COLUMNS = [
    "temp_mean",
    "temp_min",
    "temp_max",
    "apparent_temp_mean",
    "precip_sum",
    "snowfall_sum",
    "wind_max",
    "gust_max",
    "pressure_mean",
]

ANOMALY_COLUMNS = [
    "temp_anomaly",
    "temp_anomaly_z",
    "temp_min_anomaly",
    "temp_max_anomaly",
    "apparent_temp_anomaly",
    "apparent_temp_anomaly_z",
    "precip_anomaly",
    "precip_anomaly_z",
    "snowfall_anomaly",
    "snowfall_anomaly_z",
    "wind_anomaly",
    "wind_anomaly_z",
    "gust_anomaly",
    "gust_anomaly_z",
    "pressure_anomaly",
    "pressure_anomaly_z",
]

SHOCK_COLUMNS = [
    "temp_change_1d",
    "temp_change_3d",
    "apparent_temp_change_1d",
    "pressure_change_1d",
    "pressure_change_3d",
    "wind_change_1d",
    "gust_change_1d",
    "rapid_cold_snap",
    "rapid_warmup",
    "pressure_drop_event",
]

LAGGED_COLUMNS = [
    "precip_lag1",
    "precip_lag2",
    "snowfall_lag1",
    "snowfall_lag2",
    "snowfall_lag3",
    "precip_3d_total",
    "precip_7d_total",
    "snowfall_3d_total",
    "snowfall_7d_total",
    "cold_anomaly_days_3d",
    "warm_anomaly_days_3d",
    "wet_days_3d",
    "snow_days_3d",
    "day_after_major_snow",
    "two_days_after_major_snow",
    "three_days_after_major_snow",
    "days_since_major_snow_capped",
]

COMPOUND_COLUMNS = [
    "cold_anomaly_event",
    "warm_anomaly_event",
    "windy_anomaly_event",
    "major_snow_event",
    "heavy_precip_event",
    "freeze_thaw_day",
    "post_snow_thaw",
    "refreeze_after_thaw",
    "cold_windy_event",
    "snow_wind_event",
    "thermal_cold_stress",
    "thermal_heat_stress",
    "storm_severity_index",
    "travel_disruption_index",
]

ALL_ENGINEERED_COLUMNS = [
    *ANOMALY_COLUMNS,
    *SHOCK_COLUMNS,
    *LAGGED_COLUMNS,
    *COMPOUND_COLUMNS,
]


@dataclass(frozen=True)
class WeatherClimatology:
    expected: dict[str, pd.Series]
    scale: dict[str, float]


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def aggregate_hourly_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the existing hourly Open-Meteo schema to one row per local day."""
    frame = hourly.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], format="mixed", errors="coerce")
    if getattr(frame["ds"].dt, "tz", None) is not None:
        frame["ds"] = frame["ds"].dt.tz_convert("America/Montreal").dt.tz_localize(None)
    frame = frame.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")
    frame["day"] = frame["ds"].dt.normalize()

    for column in frame.columns:
        if column not in {"ds", "day"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    grouped = frame.groupby("day", sort=True)
    out = pd.DataFrame({"ds": grouped.size().index, "weather_hours": grouped.size().values})

    def add(name: str, source: str, aggregation: str) -> None:
        if source not in frame.columns:
            out[name] = np.nan
            return
        values = grouped[source].agg(aggregation)
        out[name] = values.reindex(out["ds"]).to_numpy(dtype="float64")

    add("temp_mean", "temperature_2m", "mean")
    add("temp_min", "temperature_2m", "min")
    add("temp_max", "temperature_2m", "max")
    add("apparent_temp_mean", "apparent_temperature", "mean")
    add("apparent_temp_min", "apparent_temperature", "min")
    add("apparent_temp_max", "apparent_temperature", "max")
    add("precip_sum", "precipitation", "sum")
    add("rain_sum", "rain", "sum")
    add("snowfall_sum", "snowfall", "sum")
    add("snow_depth_max", "snow_depth", "max")
    add("precip_probability_max", "precipitation_probability", "max")
    add("wind_mean", "wind_speed_10m", "mean")
    add("wind_max", "wind_speed_10m", "max")
    add("gust_max", "wind_gusts_10m", "max")
    add("humidity_mean", "relative_humidity_2m", "mean")
    add("pressure_mean", "pressure_msl", "mean")
    add("pressure_min", "pressure_msl", "min")
    add("pressure_max", "pressure_msl", "max")
    add("cloud_mean", "cloud_cover", "mean")
    out["pressure_range"] = out["pressure_max"] - out["pressure_min"]

    return out.sort_values("ds").reset_index(drop=True)


def _smooth_day_of_year(values: pd.Series, window: int = 31) -> pd.Series:
    """Circularly smooth a 1..366 day-of-year climatology."""
    values = values.reindex(range(1, 367)).astype("float64")
    values = values.interpolate(limit_direction="both")
    radius = window // 2
    wrapped = pd.concat([values.iloc[-radius:], values, values.iloc[:radius]], ignore_index=True)
    smoothed = wrapped.rolling(window=window, center=True, min_periods=1).mean()
    result = smoothed.iloc[radius : radius + 366].copy()
    result.index = range(1, 367)
    return result


def fit_climatology(history_weather: pd.DataFrame) -> WeatherClimatology:
    """Fit a smooth seasonal expectation using weather available through a cutoff."""
    frame = history_weather.copy().sort_values("ds")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame = frame.dropna(subset=["ds"])
    if len(frame) < 60:
        raise ValueError("At least 60 daily weather rows are required to fit climatology")
    doy = frame["ds"].dt.dayofyear

    expected: dict[str, pd.Series] = {}
    scale: dict[str, float] = {}
    for column in CLIMATOLOGY_SOURCE_COLUMNS:
        values = _numeric(frame, column)
        seasonal = values.groupby(doy).mean()
        seasonal = _smooth_day_of_year(seasonal)
        expected[column] = seasonal
        fitted = doy.map(seasonal).astype("float64")
        residual = values - fitted
        robust_scale = float(residual.std(skipna=True))
        if not np.isfinite(robust_scale) or robust_scale < 1e-6:
            robust_scale = float(values.std(skipna=True))
        if not np.isfinite(robust_scale) or robust_scale < 1e-6:
            robust_scale = 1.0
        scale[column] = robust_scale
    return WeatherClimatology(expected=expected, scale=scale)


def _days_since_event(event: pd.Series, cap: int = 7) -> pd.Series:
    result = np.full(len(event), cap + 1, dtype="float64")
    last = None
    for idx, active in enumerate(event.fillna(0).astype(bool).to_numpy()):
        if active:
            last = idx
            result[idx] = 0
        elif last is not None:
            result[idx] = min(idx - last, cap + 1)
    return pd.Series(result, index=event.index)


def add_weather_features(
    daily_weather: pd.DataFrame,
    climatology: WeatherClimatology,
) -> pd.DataFrame:
    """Add anomaly, shock, persistence, lag/recovery, and compound weather features."""
    out = daily_weather.copy().sort_values("ds").reset_index(drop=True)
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.normalize()
    doy = out["ds"].dt.dayofyear

    anomaly_map = {
        "temp_mean": "temp",
        "temp_min": "temp_min",
        "temp_max": "temp_max",
        "apparent_temp_mean": "apparent_temp",
        "precip_sum": "precip",
        "snowfall_sum": "snowfall",
        "wind_max": "wind",
        "gust_max": "gust",
        "pressure_mean": "pressure",
    }
    for source, prefix in anomaly_map.items():
        expected = doy.map(climatology.expected[source]).astype("float64")
        anomaly = _numeric(out, source) - expected
        out[f"{prefix}_anomaly"] = anomaly
        if prefix in {"temp", "apparent_temp", "precip", "snowfall", "wind", "gust", "pressure"}:
            out[f"{prefix}_anomaly_z"] = anomaly / climatology.scale[source]

    temp = _numeric(out, "temp_mean")
    apparent = _numeric(out, "apparent_temp_mean")
    pressure = _numeric(out, "pressure_mean")
    wind = _numeric(out, "wind_max")
    gust = _numeric(out, "gust_max")
    precip = _numeric(out, "precip_sum").fillna(0)
    snow = _numeric(out, "snowfall_sum").fillna(0)
    snow_depth = _numeric(out, "snow_depth_max").fillna(0)

    out["temp_change_1d"] = temp.diff(1)
    out["temp_change_3d"] = temp - temp.shift(1).rolling(3, min_periods=1).mean()
    out["apparent_temp_change_1d"] = apparent.diff(1)
    out["pressure_change_1d"] = pressure.diff(1)
    out["pressure_change_3d"] = pressure - pressure.shift(1).rolling(3, min_periods=1).mean()
    out["wind_change_1d"] = wind.diff(1)
    out["gust_change_1d"] = gust.diff(1)
    out["rapid_cold_snap"] = (out["temp_change_1d"] <= -7.0).astype(np.int8)
    out["rapid_warmup"] = (out["temp_change_1d"] >= 7.0).astype(np.int8)
    out["pressure_drop_event"] = (out["pressure_change_1d"] <= -8.0).astype(np.int8)

    out["precip_lag1"] = precip.shift(1)
    out["precip_lag2"] = precip.shift(2)
    out["snowfall_lag1"] = snow.shift(1)
    out["snowfall_lag2"] = snow.shift(2)
    out["snowfall_lag3"] = snow.shift(3)
    out["precip_3d_total"] = precip.rolling(3, min_periods=1).sum()
    out["precip_7d_total"] = precip.rolling(7, min_periods=1).sum()
    out["snowfall_3d_total"] = snow.rolling(3, min_periods=1).sum()
    out["snowfall_7d_total"] = snow.rolling(7, min_periods=1).sum()

    out["cold_anomaly_event"] = (out["temp_anomaly_z"] <= -1.5).astype(np.int8)
    out["warm_anomaly_event"] = (out["temp_anomaly_z"] >= 1.5).astype(np.int8)
    out["windy_anomaly_event"] = (
        (out["gust_anomaly_z"] >= 1.5) | (gust >= 45.0)
    ).astype(np.int8)
    out["major_snow_event"] = (snow >= 5.0).astype(np.int8)
    out["heavy_precip_event"] = (precip >= 15.0).astype(np.int8)

    out["cold_anomaly_days_3d"] = out["cold_anomaly_event"].rolling(3, min_periods=1).sum()
    out["warm_anomaly_days_3d"] = out["warm_anomaly_event"].rolling(3, min_periods=1).sum()
    out["wet_days_3d"] = (precip >= 1.0).astype(int).rolling(3, min_periods=1).sum()
    out["snow_days_3d"] = (snow >= 0.5).astype(int).rolling(3, min_periods=1).sum()

    major_snow = out["major_snow_event"].astype(bool)
    out["day_after_major_snow"] = major_snow.shift(1, fill_value=False).astype(np.int8)
    out["two_days_after_major_snow"] = major_snow.shift(2, fill_value=False).astype(np.int8)
    out["three_days_after_major_snow"] = major_snow.shift(3, fill_value=False).astype(np.int8)
    out["days_since_major_snow_capped"] = _days_since_event(out["major_snow_event"], cap=7)

    temp_min = _numeric(out, "temp_min")
    temp_max = _numeric(out, "temp_max")
    out["freeze_thaw_day"] = ((temp_min < 0) & (temp_max > 0)).astype(np.int8)
    out["post_snow_thaw"] = (
        (out["snowfall_3d_total"] >= 1.0) & (temp_max > 1.0)
    ).astype(np.int8)
    prior_thaw = temp_max.shift(1) > 1.0
    out["refreeze_after_thaw"] = (
        prior_thaw & (temp_min < -2.0) & ((snow_depth > 0) | (out["snowfall_3d_total"] > 0))
    ).astype(np.int8)
    out["cold_windy_event"] = (
        out["cold_anomaly_event"].astype(bool) & out["windy_anomaly_event"].astype(bool)
    ).astype(np.int8)
    out["snow_wind_event"] = ((snow >= 1.0) & (gust >= 35.0)).astype(np.int8)

    out["thermal_cold_stress"] = (-out["apparent_temp_anomaly_z"]).clip(lower=0)
    out["thermal_heat_stress"] = out["apparent_temp_anomaly_z"].clip(lower=0)
    out["storm_severity_index"] = (
        out["precip_anomaly_z"].clip(lower=0).fillna(0)
        + out["gust_anomaly_z"].clip(lower=0).fillna(0)
        + (snow / 5.0).clip(lower=0)
        + (-out["pressure_anomaly_z"]).clip(lower=0).fillna(0) * 0.5
    )
    out["travel_disruption_index"] = (
        out["storm_severity_index"]
        + out["thermal_cold_stress"].fillna(0) * 0.5
        + out["freeze_thaw_day"].astype(float)
        + out["refreeze_after_thaw"].astype(float)
    )

    return out
