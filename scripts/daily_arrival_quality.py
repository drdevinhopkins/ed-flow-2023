"""Shared daily-arrival completeness policy; never impute unobserved arrivals.

Keep the existing calendar grouping by the naive Montreal ds date. DST remains
unverified until the upstream reporting convention is established. The established
two-column daily contract is preserved; incomplete targets are NaN, with observed
partial sums and coverage available in a separate quality companion.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pandas as pd

from evaluation.audit_flow_history import TIMEZONE, audit_history

QUALITY_VERSION = "daily-arrivals-quality-v1"
DAILY_PATH = "/daily_inflow.csv"
HOURLY_PATH = "/allData.csv"


def verified_daily_targets(daily, hourly, *, now=None):
    """Verify even legacy numeric totals against authoritative hourly coverage."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    summary, quality, _ = audit_history(hourly, daily, now=now)
    quality["quality_version"] = QUALITY_VERSION
    quality["source_hourly_cutoff"] = summary["hourly_end"]
    quality["verified_at_utc"] = now.tz_convert("UTC").isoformat()
    today = now.tz_convert(TIMEZONE).tz_localize(None).normalize()
    past = quality.loc[quality["ds"].lt(today)].copy()
    verified = past[["ds", "Daily_Inflow_Total"]].copy()
    verified["Daily_Inflow_Total"] = verified["Daily_Inflow_Total"].where(past["audit_eligible"])
    verified.attrs["target_quality_version"] = QUALITY_VERSION
    return verified.reset_index(drop=True), quality


def build_daily_inflow_outputs(hourly, *, now=None):
    """Build masked legacy targets and explicit quality metadata from one snapshot."""
    source = hourly[["ds", "Inflow_Total"]].copy()
    source["ds"] = pd.to_datetime(source["ds"], format="mixed", errors="raise")
    if not pd.api.types.is_datetime64_dtype(source["ds"].dtype):
        raise ValueError("Hourly ds must contain naive Montreal wall-clock timestamps")
    source["Inflow_Total"] = pd.to_numeric(source["Inflow_Total"], errors="coerce")
    observed = source.groupby(source["ds"].dt.normalize())["Inflow_Total"].sum(min_count=1)
    daily = observed.rename("Daily_Inflow_Total").reset_index()
    return verified_daily_targets(daily, hourly, now=now)


def load_verified_daily(dbx, *, now=None):
    """Validate targets during mixed-version rollout; never trust numeric sums alone."""
    _, response = dbx.files_download(DAILY_PATH)
    daily = pd.read_csv(io.BytesIO(response.content))
    _, response = dbx.files_download(HOURLY_PATH)
    hourly = pd.read_csv(io.BytesIO(response.content))
    verified, quality = verified_daily_targets(daily, hourly, now=now)
    print(f"Daily quality {QUALITY_VERSION}: {verified['Daily_Inflow_Total'].notna().sum()} "
          f"verified prior dates; {verified['Daily_Inflow_Total'].isna().sum()} unverified dates",
          flush=True)
    return verified, quality


def require_latest_completed_day(daily, *, now=None):
    """Do not quietly forecast from an older cutoff when yesterday is unverified."""
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    yesterday = now.tz_convert(TIMEZONE).tz_localize(None).normalize() - pd.Timedelta(days=1)
    values = daily.loc[daily["ds"].eq(yesterday), "Daily_Inflow_Total"]
    if len(values) != 1 or not np.isfinite(values.iloc[0]):
        raise ValueError(f"Latest completed Montreal day {yesterday.date()} is unverified or missing")


def target_history_fingerprint(history, *, target="daily_visits"):
    """Fingerprint actual historical targets, independent of generated weather."""
    rows = history[["ds", target]].sort_values("ds")
    data = "\n".join(f"{pd.Timestamp(ds).date()}:{float(value):.17g}"
                     for ds, value in rows.itertuples(index=False, name=None))
    return hashlib.sha256(data.encode()).hexdigest()


def verify_explanation_context(formatted, history, *, quality_version=QUALITY_VERSION):
    """Do not attribute an old forecast to a rebuilt, different target context."""
    if "history_days" not in formatted or not pd.to_numeric(
        formatted["history_days"], errors="coerce").eq(len(history)).all():
        raise ValueError("Persisted forecast history length does not match verified context")
    if "target_quality_version" not in formatted or not formatted[
        "target_quality_version"].eq(quality_version).all():
        raise ValueError("Persisted forecast predates verified daily target handling; rerun forecast")
    if "target_history_sha256" not in formatted or not formatted[
        "target_history_sha256"].eq(target_history_fingerprint(history)).all():
        raise ValueError("Persisted forecast targets differ from verified context; rerun forecast")
