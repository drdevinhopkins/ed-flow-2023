#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hourly_weather_features import (  # noqa: E402
    ENGINEERED_HOURLY_WEATHER_COLUMNS,
    RAW_PLUS_SNOW_HOURLY_COLUMNS,
    add_hourly_weather_features,
    fit_hourly_climatology,
)


def synthetic_weather(days: int = 70) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    hour = ds.hour.to_numpy()
    temp = -5 + 4 * np.sin((hour - 6) / 24 * 2 * np.pi)
    return pd.DataFrame(
        {
            "ds": ds,
            "temperature_2m": temp,
            "apparent_temperature": temp - 2,
            "precipitation": 0.0,
            "rain": 0.0,
            "snowfall": 0.0,
            "snow_depth": 0.10,
            "wind_speed_10m": 12.0,
            "wind_gusts_10m": 20.0,
            "relative_humidity_2m": 75.0,
            "pressure_msl": 1015.0,
            "cloud_cover": 50.0,
        }
    )


def test_feature_contract_and_snow_recovery() -> None:
    weather = synthetic_weather()
    cutoff_idx = 60 * 24 - 1
    climatology = fit_hourly_climatology(weather.iloc[: cutoff_idx + 1])

    # Inject a 9 mm snow event over six hours followed by a thaw/refreeze.
    start = 61 * 24
    weather.loc[start : start + 5, "snowfall"] = 1.5
    weather.loc[start : start + 5, "wind_gusts_10m"] = 45.0
    weather.loc[start + 8 : start + 12, "temperature_2m"] = 2.0
    weather.loc[start + 13 : start + 16, "temperature_2m"] = -4.0

    featured = add_hourly_weather_features(weather, climatology)
    assert set(RAW_PLUS_SNOW_HOURLY_COLUMNS).issubset(featured.columns)
    assert set(ENGINEERED_HOURLY_WEATHER_COLUMNS).issubset(featured.columns)
    assert featured.loc[start + 5, "major_snow_24h_event"] == 1.0
    assert featured.loc[start + 5, "snow_wind_event"] == 1.0
    assert featured.loc[start + 8 : start + 12, "post_snow_thaw"].max() == 1.0
    assert featured.loc[start + 13 : start + 16, "refreeze_after_thaw"].max() == 1.0


def test_anomalies_use_cutoff_fitted_climatology() -> None:
    weather = synthetic_weather()
    history = weather.iloc[: 60 * 24].copy()
    future = weather.iloc[60 * 24 :].copy()
    future["temperature_2m"] += 12.0
    future["apparent_temperature"] += 12.0
    combined = pd.concat([history, future], ignore_index=True)

    climatology = fit_hourly_climatology(history)
    featured = add_hourly_weather_features(combined, climatology)
    future_featured = featured.loc[featured["ds"] > history["ds"].max()]
    assert future_featured["temperature_anomaly_z"].median() > 1.5
    assert future_featured["warm_anomaly_event"].mean() > 0.5


if __name__ == "__main__":
    test_feature_contract_and_snow_recovery()
    test_anomalies_use_cutoff_fitted_climatology()
    print("hourly weather feature tests passed")
