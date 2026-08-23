"""Current Open-Meteo weather forecast for operational ED forecasting."""

from __future__ import annotations

import time

import pandas as pd
import requests

from build_weather_history import (
    HOURLY_VARIABLES,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    fetch_history,
)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_live_forecast(
    *,
    forecast_days: int = 8,
    past_days: int = 3,
    attempts: int = 5,
) -> pd.DataFrame:
    """Fetch current forecast with explicit Montreal-local timestamps."""
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "forecast_days": forecast_days,
        "past_days": past_days,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    session = requests.Session()
    headers = {"User-Agent": "ed-flow-2023 operational daily visits forecast"}
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(
                FORECAST_URL,
                params=params,
                headers=headers,
                timeout=(20, 90),
            )
            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == attempts:
                    response.raise_for_status()
                delay = min(2**attempt, 30)
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            hourly = payload.get("hourly")
            if not hourly or "time" not in hourly:
                raise ValueError(f"Open-Meteo forecast returned no hourly data: {payload}")
            frame = pd.DataFrame(hourly).rename(columns={"time": "ds"})
            # With timezone=America/Montreal, JSON time strings are local wall-clock.
            frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
            return frame.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == attempts:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("Open-Meteo live forecast retries exhausted")


def build_operational_weather(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    forecast_days: int = 8,
) -> pd.DataFrame:
    """Combine historical forecast context with today's actual live forecast."""
    historical = fetch_history(start, end)
    live = fetch_live_forecast(forecast_days=forecast_days)
    combined = pd.concat([historical, live], ignore_index=True, sort=False)
    combined["ds"] = pd.to_datetime(combined["ds"], format="mixed", errors="coerce")
    return (
        combined.dropna(subset=["ds"])
        .sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )
