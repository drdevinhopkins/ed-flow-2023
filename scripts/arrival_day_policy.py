"""Opt-in report-day targets for interval-ending hourly arrival counts.

Keep ds as the observed endpoint. A separate interval-start view assigns flow
counts to the source report day (hours 1..24). Never shift occupancy/state data,
impute arrivals, trust the cumulative counter, or change production defaults.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from daily_arrival_quality import build_daily_inflow_outputs

INTERVAL_TARGET_VERSION = "arrival-day-interval-end-v2"
INTERVAL_QUALITY_VERSION = "daily-arrivals-interval-quality-v2"
INTERVAL_MODEL_VERSION = "intraday-ensemble-interval-v2-2026-10-08"


def interval_view(hourly, *, now):
    """Validate endpoint clocks before creating a detached arrival-only view."""
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    source = hourly.copy(deep=True)
    source["ds"] = pd.to_datetime(source["ds"], format="mixed", errors="raise")
    if not pd.api.types.is_datetime64_dtype(source.ds.dtype):
        raise ValueError("Hourly ds must contain naive Montreal wall-clock timestamps")
    if source.empty or source.ds.isna().any() or source.ds.ne(source.ds.dt.floor("h")).any():
        raise ValueError("Hourly endpoints must be valid whole clock hours")
    if source.ds.max() > now.tz_convert("America/Montreal").tz_localize(None):
        raise ValueError("Hourly history contains future endpoints")
    source["interval_end_ds"] = source.ds
    source["ds"] = source.ds - pd.Timedelta(hours=1)
    return source


def build_interval_daily_outputs(hourly, *, now):
    """Use the same completeness rules on arrival intervals, not state timestamps."""
    view = interval_view(hourly, now=now)
    daily, quality = build_daily_inflow_outputs(view, now=now)
    daily.attrs["target_quality_version"] = INTERVAL_QUALITY_VERSION
    daily.attrs["target_definition_version"] = INTERVAL_TARGET_VERSION
    quality["quality_version"] = INTERVAL_QUALITY_VERSION
    quality["target_definition_version"] = INTERVAL_TARGET_VERSION
    quality["source_hourly_cutoff"] = view.interval_end_ds.max().isoformat()
    quality["required_first_endpoint"] = quality.ds + pd.Timedelta(hours=1)
    quality["required_last_endpoint"] = quality.ds + pd.Timedelta(days=1)
    quality["missing_interval_start_hours"] = quality.missing_clock_hours
    return daily, quality


def interval_day_flow(hourly, *, now):
    """Prepare training/live flow with original endpoints and report-day grouping."""
    view = interval_view(hourly, now=now)
    _, quality = build_interval_daily_outputs(hourly, now=now)
    flow = hourly.copy(deep=True)
    flow["ds"] = view.interval_end_ds
    flow["Inflow_Total"] = pd.to_numeric(flow.Inflow_Total, errors="coerce")
    flow["day"] = view.ds.dt.normalize()
    flow["_source_order"] = np.arange(len(flow))
    # The closing midnight belongs to the preceding report day, with progress 24/24.
    flow["_arrival_report_hour"] = flow.ds.dt.hour.replace(0, 24)
    eligible = quality.set_index("ds").audit_eligible
    flow["is_complete_day"] = flow.day.map(eligible).fillna(False).astype(bool)
    flow = flow.sort_values(["ds", "_source_order"]).reset_index(drop=True)
    flow.attrs["target_definition_version"] = INTERVAL_TARGET_VERSION
    flow.attrs["target_quality_version"] = INTERVAL_QUALITY_VERSION
    return flow


def select_interval_issues(archive):
    """Keep v2 records only; never reinterpret legacy forecasts as new-target evidence."""
    required = {"target_definition_version", "target_quality_version"}
    if required - set(archive):
        return archive.iloc[:0].copy()
    return archive.loc[archive.target_definition_version.eq(INTERVAL_TARGET_VERSION)
                       & archive.target_quality_version.eq(INTERVAL_QUALITY_VERSION)].copy()
