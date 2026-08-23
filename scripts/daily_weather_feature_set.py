"""Promoted weather covariates for native Chronos-2 daily ED arrival forecasts.

This is the production feature set selected by the 16-cutoff / 112-day targeted
weather ablation. Keep this definition shared by backtests and operational forecasts so
feature selection cannot silently drift between validation and production.
"""

from weather_features import RAW_DAILY_COLUMNS

SNOW_RECOVERY_COLUMNS = [
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

SURFACE_CONDITION_COLUMNS = [
    "freeze_thaw_day",
    "post_snow_thaw",
    "refreeze_after_thaw",
]

RAW_PLUS_SNOW_COLUMNS = list(
    dict.fromkeys(
        [
            *RAW_DAILY_COLUMNS,
            *SNOW_RECOVERY_COLUMNS,
            *SURFACE_CONDITION_COLUMNS,
        ]
    )
)

PROMOTED_WEATHER_FEATURE_SET = "raw_plus_snow"
