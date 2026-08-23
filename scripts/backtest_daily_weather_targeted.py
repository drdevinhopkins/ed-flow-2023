#!/usr/bin/env python3
"""Targeted second-pass weather ablation for native Chronos-2 daily ED visits.

The broad 16-cutoff experiment showed that raw weather helps modestly, the full
engineered set helps more, but anomaly-only and shock-only blocks add noise. This pass
keeps the calendar-closure control and compares smaller combinations aimed at the
segments where weather produced clear gains: cold/thermal stress, freeze-thaw/refreeze,
and post-snow recovery.
"""

from __future__ import annotations

import backtest_daily_weather_features as base
from backtest_daily_weather_features_dense import select_cutoffs
from weather_features import (
    ANOMALY_COLUMNS,
    COMPOUND_COLUMNS,
    LAGGED_COLUMNS,
    RAW_DAILY_COLUMNS,
)

TARGETED_THERMAL = [
    "temp_anomaly",
    "temp_anomaly_z",
    "temp_min_anomaly",
    "temp_max_anomaly",
    "apparent_temp_anomaly",
    "apparent_temp_anomaly_z",
    "wind_anomaly_z",
    "gust_anomaly_z",
    "cold_anomaly_event",
    "warm_anomaly_event",
    "windy_anomaly_event",
    "cold_windy_event",
    "thermal_cold_stress",
    "thermal_heat_stress",
]

TARGETED_SNOW_RECOVERY = [
    "snowfall_anomaly",
    "snowfall_anomaly_z",
    "snowfall_lag1",
    "snowfall_lag2",
    "snowfall_lag3",
    "snowfall_3d_total",
    "snowfall_7d_total",
    "snow_days_3d",
    "major_snow_event",
    "day_after_major_snow",
    "two_days_after_major_snow",
    "three_days_after_major_snow",
    "days_since_major_snow_capped",
    "snow_wind_event",
]

TARGETED_SURFACE = [
    "freeze_thaw_day",
    "post_snow_thaw",
    "refreeze_after_thaw",
]

TARGETED_STORM = [
    "precip_anomaly_z",
    "gust_anomaly_z",
    "pressure_anomaly_z",
    "storm_severity_index",
    "travel_disruption_index",
]

TARGETED_COLUMNS = list(
    dict.fromkeys(
        [
            *RAW_DAILY_COLUMNS,
            *TARGETED_THERMAL,
            *TARGETED_SNOW_RECOVERY,
            *TARGETED_SURFACE,
            *TARGETED_STORM,
        ]
    )
)

RAW_PLUS_COMPOUND = list(dict.fromkeys([*RAW_DAILY_COLUMNS, *COMPOUND_COLUMNS]))
RAW_PLUS_SNOW = list(
    dict.fromkeys([*RAW_DAILY_COLUMNS, *TARGETED_SNOW_RECOVERY, *TARGETED_SURFACE])
)
RAW_PLUS_THERMAL = list(dict.fromkeys([*RAW_DAILY_COLUMNS, *TARGETED_THERMAL]))
RAW_PLUS_LAGGED = list(dict.fromkeys([*RAW_DAILY_COLUMNS, *LAGGED_COLUMNS]))

SCENARIOS = [
    "baseline",
    "calendar_closure",
    "raw_weather",
    "raw_plus_thermal",
    "raw_plus_snow",
    "raw_plus_compound",
    "raw_plus_lagged",
    "targeted_weather",
    "all_weather",
]

_original_columns_for_scenario = base._weather_columns_for_scenario


def weather_columns_for_scenario(scenario: str) -> list[str]:
    if scenario == "raw_plus_thermal":
        return RAW_PLUS_THERMAL
    if scenario == "raw_plus_snow":
        return RAW_PLUS_SNOW
    if scenario == "raw_plus_compound":
        return RAW_PLUS_COMPOUND
    if scenario == "raw_plus_lagged":
        return RAW_PLUS_LAGGED
    if scenario == "targeted_weather":
        return TARGETED_COLUMNS
    return _original_columns_for_scenario(scenario)


if __name__ == "__main__":
    base.select_cutoffs = select_cutoffs
    base._weather_columns_for_scenario = weather_columns_for_scenario
    base.SCENARIOS = SCENARIOS
    base.main()
