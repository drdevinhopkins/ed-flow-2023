"""Leakage-safe long-term covariates for hourly ED Chronos-2 forecasts.

Chronos-2 can only consume 8,192 raw hourly observations (~341 days), so this
module exposes information from older history explicitly as known covariates:

* deterministic annual Fourier phase;
* same-local-hour target values one, two, and three calendar years earlier;
* seasonally aligned multi-year level and growth summaries; and
* slow secular level/growth estimates from trailing 90d and 365d means.

For historical rows, secular features are calculated strictly from observations
before that row. For forecast rows, secular features are frozen at the forecast
cutoff, so no future target observations are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUAL_HARMONICS = 3
ANNUAL_LAG_YEARS = (1, 2, 3)
TREND_WINDOWS_HOURS = {"90d": 90 * 24, "365d": 365 * 24}


def _timestamp_minus_years(ts: pd.Timestamp, years: int) -> pd.Timestamp:
    return ts - pd.DateOffset(years=years)


def annual_fourier_frame(timestamps: pd.Series, harmonics: int = ANNUAL_HARMONICS) -> pd.DataFrame:
    """Return smooth known-future annual phase features."""
    ds = pd.to_datetime(timestamps, errors="coerce")
    year_start = pd.to_datetime(ds.dt.year.astype("Int64").astype(str) + "-01-01")
    next_year = pd.to_datetime((ds.dt.year + 1).astype("Int64").astype(str) + "-01-01")
    elapsed = (ds - year_start) / pd.Timedelta(hours=1)
    year_hours = (next_year - year_start) / pd.Timedelta(hours=1)
    phase = 2.0 * np.pi * elapsed / year_hours

    out = pd.DataFrame({"ds": ds})
    for k in range(1, harmonics + 1):
        out[f"annual_sin_{k}"] = np.sin(k * phase).astype(float)
        out[f"annual_cos_{k}"] = np.cos(k * phase).astype(float)
    return out


def annual_target_memory_frame(
    flow: pd.DataFrame,
    timestamps: pd.Series,
    targets: list[str],
) -> pd.DataFrame:
    """Return exact calendar-year lags plus seasonally aligned growth summaries."""
    indexed = flow.set_index("ds")[targets].sort_index()
    ds = pd.to_datetime(timestamps, errors="coerce")
    out = pd.DataFrame({"ds": ds})

    for target in targets:
        lag_columns: list[str] = []
        for years in ANNUAL_LAG_YEARS:
            lookup = pd.DatetimeIndex(
                [_timestamp_minus_years(ts, years) for ts in ds]
            )
            column = f"{target}__lag_{years}y"
            out[column] = indexed[target].reindex(lookup).to_numpy(dtype=float)
            lag_columns.append(column)

        out[f"{target}__annual_level_mean"] = out[lag_columns].mean(axis=1)
        out[f"{target}__annual_growth_recent"] = (
            out[f"{target}__lag_1y"] - out[f"{target}__lag_2y"]
        )
        out[f"{target}__annual_growth_3y"] = (
            out[f"{target}__lag_1y"] - out[f"{target}__lag_3y"]
        ) / 2.0
    return out


def _historical_secular_frame(flow: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Build as-of rolling level and year-over-year growth features for every history row."""
    ordered = flow[["ds", *targets]].sort_values("ds").reset_index(drop=True)
    out = pd.DataFrame({"ds": ordered["ds"]})

    for target in targets:
        values = pd.to_numeric(ordered[target], errors="coerce")
        for label, hours in TREND_WINDOWS_HOURS.items():
            # shift(1) guarantees that the feature at t only uses values before t.
            level = values.shift(1).rolling(hours, min_periods=hours).mean()
            level_name = f"{target}__level_{label}"
            yoy_name = f"{target}__growth_{label}_yoy"
            out[level_name] = level
            out[yoy_name] = level - level.shift(365 * 24)
    return out


def secular_trend_frame(
    flow: pd.DataFrame,
    timestamps: pd.Series,
    targets: list[str],
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Return historical as-of trend features and freeze them after ``cutoff``."""
    ds = pd.to_datetime(timestamps, errors="coerce")
    historical = _historical_secular_frame(flow, targets).set_index("ds")
    result = historical.reindex(pd.DatetimeIndex(ds)).reset_index(drop=True)
    result.insert(0, "ds", ds.reset_index(drop=True))

    feature_cols = [c for c in result.columns if c != "ds"]
    cutoff_row = historical.reindex([pd.Timestamp(cutoff)])
    if cutoff_row[feature_cols].isna().any().any():
        missing = cutoff_row[feature_cols].columns[cutoff_row[feature_cols].isna().any()].tolist()
        raise ValueError(f"Insufficient history for secular trend features at {cutoff}: {missing}")

    future_mask = result["ds"] > cutoff
    if future_mask.any():
        frozen = cutoff_row.iloc[0][feature_cols]
        result.loc[future_mask, feature_cols] = frozen.to_numpy(dtype=float)
    return result


def build_long_term_feature_frame(
    flow: pd.DataFrame,
    timestamps: pd.Series,
    targets: list[str],
    *,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Build all annual-memory and secular-growth features for one forecast cutoff."""
    fourier = annual_fourier_frame(timestamps)
    annual = annual_target_memory_frame(flow, timestamps, targets)
    trend = secular_trend_frame(flow, timestamps, targets, cutoff=cutoff)
    out = fourier.merge(annual, on="ds", how="left").merge(trend, on="ds", how="left")
    return out


def scenario_columns(scenario: str, targets: list[str]) -> list[str]:
    """Columns used by each long-term-memory ablation scenario."""
    fourier = [
        name
        for k in range(1, ANNUAL_HARMONICS + 1)
        for name in (f"annual_sin_{k}", f"annual_cos_{k}")
    ]
    lag_columns = [
        f"{target}__lag_{years}y"
        for target in targets
        for years in ANNUAL_LAG_YEARS
    ]
    annual_level = [f"{target}__annual_level_mean" for target in targets]
    annual_growth = [
        name
        for target in targets
        for name in (
            f"{target}__annual_growth_recent",
            f"{target}__annual_growth_3y",
        )
    ]
    secular = [
        name
        for target in targets
        for label in TREND_WINDOWS_HOURS
        for name in (
            f"{target}__level_{label}",
            f"{target}__growth_{label}_yoy",
        )
    ]

    if scenario == "baseline":
        return []
    if scenario == "annual_fourier":
        return fourier
    if scenario == "annual_memory":
        return [*lag_columns, *annual_level]
    if scenario == "annual_memory_growth":
        return [*lag_columns, *annual_level, *annual_growth]
    if scenario == "secular_growth":
        return secular
    if scenario == "annual_plus_growth":
        return [*fourier, *lag_columns, *annual_level, *annual_growth, *secular]
    raise ValueError(f"Unknown long-term feature scenario: {scenario}")
