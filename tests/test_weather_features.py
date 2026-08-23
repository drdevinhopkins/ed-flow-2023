#!/usr/bin/env python3
"""Lightweight tests for weather feature engineering; runnable without pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from weather_features import (  # noqa: E402
    add_weather_features,
    aggregate_hourly_weather,
    fit_climatology,
)


def synthetic_hourly(days: int = 420) -> pd.DataFrame:
    ds = pd.date_range("2025-01-01", periods=days * 24, freq="h")
    doy = ds.dayofyear.to_numpy()
    hour = ds.hour.to_numpy()
    seasonal = 8.0 + 17.0 * np.sin(2 * np.pi * (doy - 105) / 365.25)
    diurnal = 3.0 * np.sin(2 * np.pi * (hour - 8) / 24)
    temp = seasonal + diurnal
    frame = pd.DataFrame(
        {
            "ds": ds,
            "temperature_2m": temp,
            "apparent_temperature": temp - 1.5,
            "precipitation_probability": 10.0,
            "precipitation": 0.0,
            "rain": 0.0,
            "snowfall": 0.0,
            "snow_depth": 0.0,
            "wind_speed_10m": 12.0,
            "wind_gusts_10m": 22.0,
            "relative_humidity_2m": 65.0,
            "pressure_msl": 1015.0,
            "cloud_cover": 40.0,
        }
    )
    return frame


def test_daily_aggregation() -> None:
    hourly = synthetic_hourly(days=2)
    hourly.loc[hourly["ds"].dt.day.eq(1), "precipitation"] = 1.0
    daily = aggregate_hourly_weather(hourly)
    assert len(daily) == 2
    assert daily.loc[0, "weather_hours"] == 24
    assert np.isclose(daily.loc[0, "precip_sum"], 24.0)
    assert daily.loc[0, "temp_max"] > daily.loc[0, "temp_min"]


def test_anomaly_and_event_features() -> None:
    hourly = synthetic_hourly()
    daily = aggregate_hourly_weather(hourly)
    cutoff_idx = 365
    climatology = fit_climatology(daily.iloc[:cutoff_idx])

    # Create an unseasonably cold, windy, snowy day after the climatology period.
    event_idx = cutoff_idx + 10
    daily.loc[event_idx, "temp_mean"] -= 22.0
    daily.loc[event_idx, "temp_min"] -= 22.0
    daily.loc[event_idx, "temp_max"] -= 22.0
    daily.loc[event_idx, "apparent_temp_mean"] -= 25.0
    daily.loc[event_idx, "apparent_temp_min"] -= 25.0
    daily.loc[event_idx, "apparent_temp_max"] -= 25.0
    daily.loc[event_idx, "gust_max"] = 70.0
    daily.loc[event_idx, "wind_max"] = 50.0
    daily.loc[event_idx, "snowfall_sum"] = 10.0

    featured = add_weather_features(daily.iloc[: event_idx + 4], climatology)
    row = featured.iloc[event_idx]
    assert row["temp_anomaly"] < -10
    assert row["cold_anomaly_event"] == 1
    assert row["windy_anomaly_event"] == 1
    assert row["cold_windy_event"] == 1
    assert row["major_snow_event"] == 1
    assert row["snow_wind_event"] == 1
    assert featured.iloc[event_idx + 1]["day_after_major_snow"] == 1
    assert featured.iloc[event_idx + 2]["two_days_after_major_snow"] == 1
    assert featured.iloc[event_idx + 3]["three_days_after_major_snow"] == 1


def test_freeze_thaw_and_refreeze() -> None:
    hourly = synthetic_hourly()
    daily = aggregate_hourly_weather(hourly)
    climatology = fit_climatology(daily.iloc[:365])
    start = 370
    daily.loc[start, ["temp_min", "temp_max", "temp_mean"]] = [-3.0, 4.0, 1.0]
    daily.loc[start, "snowfall_sum"] = 4.0
    daily.loc[start, "snow_depth_max"] = 0.05
    daily.loc[start + 1, ["temp_min", "temp_max", "temp_mean"]] = [-8.0, -1.0, -4.0]
    daily.loc[start + 1, "snow_depth_max"] = 0.04
    featured = add_weather_features(daily.iloc[: start + 2], climatology)
    assert featured.iloc[start]["freeze_thaw_day"] == 1
    assert featured.iloc[start]["post_snow_thaw"] == 1
    assert featured.iloc[start + 1]["refreeze_after_thaw"] == 1


def main() -> None:
    test_daily_aggregation()
    test_anomaly_and_event_features()
    test_freeze_thaw_and_refreeze()
    print("weather feature tests passed")


if __name__ == "__main__":
    main()
