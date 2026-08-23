#!/usr/bin/env python3
"""Build a continuous hourly weather table for daily-visit backtesting.

The repository's rolling ``weather.csv`` contains the latest Open-Meteo forecast plus a
sparse history accumulated from periodic updates.  That is sufficient for production
forecasting but leaves gaps that make historical weather ablation fragile.

This helper backfills those gaps from Open-Meteo's Historical Forecast API, whose schema
matches the Forecast API.  Historical Forecast is a stitched near-analysis time series,
not the exact 1-7 day forecast snapshot that would have been available at each ED
forecast cutoff.  It is therefore appropriate for weather-signal/feature screening, but
not for a leakage-free evaluation of weather forecast skill at fixed lead times.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

LIVE_WEATHER_URL = (
    "https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/"
    "weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1"
)
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LATITUDE = 45.5088
LONGITUDE = -73.5878
TIMEZONE = "America/Montreal"
HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
    "apparent_temperature",
    "relative_humidity_2m",
    "pressure_msl",
]


def fetch_chunk(start: pd.Timestamp, end: pd.Timestamp, attempts: int = 5) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": TIMEZONE,
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    headers = {"User-Agent": "ed-flow-2023 weather feature backtest"}
    for attempt in range(1, attempts + 1):
        response = requests.get(
            HISTORICAL_FORECAST_URL,
            params=params,
            headers=headers,
            timeout=120,
        )
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == attempts:
                response.raise_for_status()
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(2 ** attempt, 30)
            except ValueError:
                delay = min(2 ** attempt, 30)
            print(
                f"Historical Forecast API HTTP {response.status_code}; "
                f"retrying in {delay:.0f}s"
            )
            time.sleep(delay)
            continue
        response.raise_for_status()
        payload = response.json()
        hourly = payload.get("hourly")
        if not hourly or "time" not in hourly:
            raise ValueError(f"Historical Forecast API returned no hourly data: {payload}")
        frame = pd.DataFrame(hourly).rename(columns={"time": "ds"})
        frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
        return frame.dropna(subset=["ds"])
    raise RuntimeError("Historical Forecast API retries exhausted")


def fetch_history(start: pd.Timestamp, end: pd.Timestamp, chunk_days: int = 180) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    current = start.normalize()
    end = end.normalize()
    chunk_number = 0
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end)
        chunk_number += 1
        print(
            f"Fetching historical forecast chunk {chunk_number}: "
            f"{current.date()} to {chunk_end.date()}"
        )
        chunks.append(fetch_chunk(current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
        if current <= end:
            time.sleep(0.5)
    if not chunks:
        return pd.DataFrame(columns=["ds", *HOURLY_VARIABLES])
    return pd.concat(chunks, ignore_index=True)


def build_weather_history(
    *,
    live_url: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    historical = fetch_history(start, end)
    live = pd.read_csv(live_url)
    live["ds"] = pd.to_datetime(live["ds"], format="mixed", errors="coerce")
    live = live.dropna(subset=["ds"])

    # Align schemas.  Historical Forecast intentionally requests the columns used by the
    # daily feature aggregator; extra live columns are retained because they are harmless
    # and may be useful later.
    combined = pd.concat([historical, live], ignore_index=True, sort=False)
    combined["ds"] = pd.to_datetime(combined["ds"], format="mixed", errors="coerce")
    combined = (
        combined.dropna(subset=["ds"])
        .sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument(
        "--end-date",
        default=(pd.Timestamp.now(tz=TIMEZONE).normalize() - pd.Timedelta(days=1)).strftime(
            "%Y-%m-%d"
        ),
    )
    parser.add_argument("--live-url", default=LIVE_WEATHER_URL)
    parser.add_argument("--output", type=Path, default=Path("weather_backfilled.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("end-date must be on or after start-date")
    combined = build_weather_history(live_url=args.live_url, start=start, end=end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"Wrote {len(combined):,} hourly rows to {args.output}")
    print(f"Weather range: {combined['ds'].min()} to {combined['ds'].max()}")


if __name__ == "__main__":
    main()
