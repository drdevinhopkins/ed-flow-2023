#!/usr/bin/env python3
"""Regression checks for grouped Chronos-2 daily-arrival explainability."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import explain_daily_visits_forecast as explain  # noqa: E402
import forecast_daily_visits as forecast  # noqa: E402


def synthetic_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.date_range("2025-01-01", periods=420, freq="D")
    history = pd.DataFrame(
        {
            "id": forecast.SERIES_ID,
            "ds": ds,
            forecast.TARGET: 230.0
            + 8.0 * np.sin(np.arange(len(ds)) * 2 * np.pi / 7),
        }
    )
    future = pd.DataFrame(
        {
            "id": forecast.SERIES_ID,
            "ds": pd.date_range(ds.max() + pd.Timedelta(days=1), periods=7, freq="D"),
        }
    )

    covariates = list(
        dict.fromkeys(
            column
            for columns in explain.FEATURE_GROUPS.values()
            for column in columns
        )
    )
    day = np.arange(len(history), dtype=float)
    for column in covariates:
        if column == "days_since_major_snow_capped":
            history[column] = 8.0
            future[column] = 8.0
        elif column in explain.ZERO_NEUTRAL_COLUMNS:
            history[column] = 0.0
            future[column] = 0.0
        elif column.startswith(("temp", "apparent_temp")):
            history[column] = 10.0 + 12.0 * np.sin(day * 2 * np.pi / 365.25)
            future[column] = 30.0
        elif column in {"precip_sum", "rain_sum", "snowfall_sum", "snow_depth_max"}:
            history[column] = 0.0
            future[column] = 0.0
        elif column in {"wind_mean", "wind_max", "gust_max"}:
            history[column] = 12.0
            future[column] = 25.0
        elif column.startswith("pressure"):
            history[column] = 1013.0
            future[column] = 1005.0
        elif column == "humidity_mean":
            history[column] = 65.0
            future[column] = 80.0
        elif column == "cloud_mean":
            history[column] = 50.0
            future[column] = 75.0
        else:
            history[column] = 0.0
            future[column] = 0.0

    future.loc[0, "is_qc_holiday"] = 1.0
    future.loc[0, "is_system_closed_day"] = 1.0
    future.loc[1, "major_snow_event"] = 1.0
    future.loc[1, "snowfall_sum"] = 12.0
    return history, future


def test_neutral_future_is_complete_and_event_free() -> None:
    history, future = synthetic_frames()
    neutral = explain.build_neutral_future(history, future)

    assert len(neutral) == len(future)
    assert not neutral.isna().any().any()
    assert neutral["is_qc_holiday"].eq(0.0).all()
    assert neutral["is_system_closed_day"].eq(0.0).all()
    assert neutral["major_snow_event"].eq(0.0).all()
    assert neutral["day_after_major_snow"].eq(0.0).all()
    assert neutral["days_since_major_snow_capped"].gt(0.0).all()

    # Continuous weather is seasonally neutralized rather than zeroed.
    assert neutral["temp_mean"].abs().max() > 1.0
    assert not np.allclose(neutral["temp_mean"], future["temp_mean"])


def test_weekday_baseline_uses_recent_same_weekdays() -> None:
    history, future = synthetic_frames()
    baseline = explain.weekday_baseline(history, future["ds"], weeks=8)
    assert len(baseline) == 7
    assert baseline.notna().all()

    first_date = future["ds"].iloc[0]
    mask = history["ds"].dt.weekday.eq(first_date.weekday())
    expected = history.loc[mask, forecast.TARGET].tail(8).mean()
    assert np.isclose(baseline.iloc[0], expected)


def test_enriched_output_ranks_driver_effects() -> None:
    history, future = synthetic_frames()
    formatted = pd.DataFrame(
        {
            "ds": future["ds"],
            "daily_visits_prediction": np.arange(7, dtype=float) + 240.0,
            "data_cutoff": history["ds"].max(),
            "horizon_day": np.arange(1, 8),
        }
    )
    rows = []
    effects = {
        "temperature": 7.5,
        "calendar_closure": -4.0,
        "wind": 1.5,
    }
    for horizon, ds in enumerate(future["ds"], start=1):
        for group, effect in effects.items():
            rows.append(
                {
                    "ds": ds,
                    "horizon_day": horizon,
                    "driver_group": group,
                    "effect_visits": effect,
                    "full_prediction": float(
                        formatted.loc[horizon - 1, "daily_visits_prediction"]
                    ),
                    "counterfactual_prediction": float(
                        formatted.loc[horizon - 1, "daily_visits_prediction"] - effect
                    ),
                    "representative_feature": (
                        "temp_mean" if group == "temperature" else None
                    ),
                    "feature_value": 30.0 if group == "temperature" else None,
                    "neutral_value": 18.0 if group == "temperature" else None,
                    "explainability_method": explain.EXPLAINABILITY_METHOD,
                }
            )
    explanations = pd.DataFrame(rows)
    enriched = explain.enrich_forecast(formatted, history, explanations)

    assert enriched["top_driver_1"].eq("temperature").all()
    assert enriched["top_driver_2"].eq("calendar_closure").all()
    assert enriched["top_driver_3"].eq("wind").all()
    assert np.allclose(enriched["top_driver_1_effect"], 7.5)
    assert enriched["explanation_text"].str.contains("same-weekday baseline").all()
    assert enriched["explainability_method"].eq(explain.EXPLAINABILITY_METHOD).all()


if __name__ == "__main__":
    test_neutral_future_is_complete_and_event_free()
    test_weekday_baseline_uses_recent_same_weekdays()
    test_enriched_output_ranks_driver_effects()
    print("daily visit explainability tests passed")
