#!/usr/bin/env python3
"""Regression checks for the promoted native Chronos-2 daily-arrivals forecast."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from daily_weather_feature_set import (  # noqa: E402
    PROMOTED_WEATHER_FEATURE_SET,
    RAW_PLUS_SNOW_COLUMNS,
)
from forecast_daily_visits import build_forecast_frames  # noqa: E402


def synthetic_daily(start: str = "2025-01-01", days: int = 220) -> pd.DataFrame:
    ds = pd.date_range(start, periods=days, freq="D")
    visits = 235 + 12 * np.sin(np.arange(days) * 2 * np.pi / 7)
    return pd.DataFrame(
        {
            "ds": ds,
            "daily_visits": visits.astype(float),
            "observed_rows": 24,
            "numeric_rows": 24,
            "is_complete": True,
        }
    )


def synthetic_hourly_weather(start: str = "2024-12-15", days: int = 260) -> pd.DataFrame:
    ds = pd.date_range(start, periods=days * 24, freq="h")
    hour = np.arange(len(ds))
    seasonal = -4 + 15 * np.sin(hour / (24 * 365) * 2 * np.pi)
    diurnal = 4 * np.sin((hour % 24) / 24 * 2 * np.pi)
    temp = seasonal + diurnal
    snow = np.zeros(len(ds), dtype=float)

    # Add two synthetic snow events so lag/recovery and surface features are exercised.
    day_index = ((ds - ds.min()) / pd.Timedelta(days=1)).astype(int)
    snow[(day_index == 205) & ((hour % 24) < 10)] = 0.8
    snow[(day_index == 225) & ((hour % 24) < 8)] = 1.0

    return pd.DataFrame(
        {
            "ds": ds,
            "temperature_2m": temp,
            "apparent_temperature": temp - 2.0,
            "precipitation": snow,
            "rain": np.zeros(len(ds)),
            "snowfall": snow,
            "snow_depth": pd.Series(snow).rolling(72, min_periods=1).sum().to_numpy() / 100,
            "cloud_cover": 45.0,
            "wind_speed_10m": 15.0,
            "wind_gusts_10m": 28.0,
            "relative_humidity_2m": 72.0,
            "pressure_msl": 1014.0 + 3 * np.sin(hour / 96),
        }
    )


def test_feature_contract() -> None:
    assert PROMOTED_WEATHER_FEATURE_SET == "raw_plus_snow"
    required = {
        "temp_mean",
        "snowfall_sum",
        "snowfall_lag1",
        "snowfall_lag2",
        "snowfall_lag3",
        "snowfall_7d_total",
        "day_after_major_snow",
        "two_days_after_major_snow",
        "three_days_after_major_snow",
        "freeze_thaw_day",
        "post_snow_thaw",
        "refreeze_after_thaw",
    }
    assert required.issubset(RAW_PLUS_SNOW_COLUMNS)

    # Losing standalone anomaly/shock blocks must not leak into the promoted set.
    excluded = {
        "temp_anomaly",
        "temp_anomaly_z",
        "pressure_change_1d",
        "rapid_cold_snap",
        "rapid_warmup",
        "storm_severity_index",
        "travel_disruption_index",
    }
    assert not excluded.intersection(RAW_PLUS_SNOW_COLUMNS)


def test_frames_are_complete_and_future_known() -> None:
    daily = synthetic_daily()
    weather = synthetic_hourly_weather()
    cutoff, history, future = build_forecast_frames(
        daily,
        weather,
        horizon_days=7,
        context_days=180,
        min_history_days=120,
    )

    assert cutoff == daily["ds"].max()
    assert len(history) == 180
    assert len(future) == 7
    assert future["ds"].min() == cutoff + pd.Timedelta(days=1)
    assert future["ds"].max() == cutoff + pd.Timedelta(days=7)
    assert not future.isna().any().any()
    assert not history.isna().any().any()

    for column in RAW_PLUS_SNOW_COLUMNS:
        assert column in history.columns
        assert column in future.columns

    assert "is_system_closed_day" in history.columns
    assert "is_system_closed_day" in future.columns
    assert "temp_anomaly" not in future.columns
    assert "pressure_change_1d" not in future.columns


if __name__ == "__main__":
    test_feature_contract()
    test_frames_are_complete_and_future_known()
    print("daily visit forecast tests passed")
