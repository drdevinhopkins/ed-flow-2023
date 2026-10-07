#!/usr/bin/env python3
"""Read-only audit of hourly coverage and the daily totals derived from it.

Keep the established calendar-day aggregation (ds date, including 00:00).
Naive wall-clock data cannot resolve DST or prove missing-hour arrivals. Never
impute arrivals, overwrite history, or use this diagnostic as a replacement target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TIMEZONE = "America/Montreal"


def audit_history(hourly: pd.DataFrame, daily: pd.DataFrame, *, now: pd.Timestamp):
    """Return summary, per-day quality and missing internal clock-hour slots."""
    for frame, required in [(hourly, {"ds", "Inflow_Total"}),
                            (daily, {"ds", "Daily_Inflow_Total"})]:
        if required - set(frame.columns):
            raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    h = hourly[["ds", "Inflow_Total"]].copy()
    # The source currently stores local naive wall times. Do not silently drop
    # offsets or mix timestamp conventions when this format changes.
    h["ds"] = pd.to_datetime(h["ds"], format="mixed", errors="coerce")
    if not pd.api.types.is_datetime64_dtype(h["ds"].dtype):
        raise ValueError("Hourly ds must contain naive Montreal wall-clock timestamps")
    if h["ds"].isna().any() or h.empty:
        raise ValueError("Hourly history contains invalid timestamps or no rows")
    if h["ds"].ne(h["ds"].dt.floor("h")).any():
        raise ValueError("Hourly timestamps must be on clock-hour boundaries")
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    today = now.tz_convert(TIMEZONE).tz_localize(None).normalize()
    if h["ds"].max() > now.tz_convert(TIMEZONE).tz_localize(None):
        raise ValueError("Hourly history contains future timestamps")
    h["Inflow_Total"] = pd.to_numeric(h["Inflow_Total"], errors="coerce")
    h["valid_inflow"] = np.isfinite(h["Inflow_Total"]) & h["Inflow_Total"].ge(0)
    h["duplicate"] = h["ds"].duplicated(keep=False)
    h["day"] = h["ds"].dt.normalize()
    grouped = h.groupby("day").agg(
        observed_rows=("ds", "size"), unique_hours=("ds", "nunique"),
        valid_inflow_rows=("valid_inflow", "sum"),
        duplicate_rows=("duplicate", "sum"),
        observed_inflow_sum=("Inflow_Total", lambda x: x.sum(min_count=1)),
    )
    dates = pd.date_range(h["day"].min(), h["day"].max(), freq="D", name="ds")
    result = grouped.reindex(dates).reset_index()
    for col in ["observed_rows", "unique_hours", "valid_inflow_rows", "duplicate_rows"]:
        result[col] = result[col].fillna(0).astype(int)
    observed = set(h["ds"])
    result["missing_clock_hours"] = [
        ";".join(f"{hour:02d}" for hour in range(24)
                 if day + pd.Timedelta(hours=hour) not in observed)
        for day in dates
    ]
    # An offset change requires source verification even when all 24 naive
    # slots are present. A generic 23..25-row check also hides ordinary gaps.
    tz = ZoneInfo(TIMEZONE)
    result["dst_transition"] = [
        day.to_pydatetime().replace(tzinfo=tz).utcoffset()
        != (day + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=tz).utcoffset()
        for day in dates
    ]
    result["clock_coverage_complete"] = (
        result["unique_hours"].eq(24) & result["observed_rows"].eq(24)
        & result["valid_inflow_rows"].eq(24) & result["duplicate_rows"].eq(0)
    )
    result["quality_status"] = "complete_clock_coverage"
    result.loc[~result["clock_coverage_complete"], "quality_status"] = "incomplete_or_invalid"
    result.loc[result["dst_transition"], "quality_status"] = "dst_requires_verification"
    result.loc[(result["ds"].eq(dates.min())) & ~result["clock_coverage_complete"],
               "quality_status"] = "leading_partial"
    result.loc[result["ds"].ge(today), "quality_status"] = "current_day_partial"
    # This flag is a conservative audit eligibility rule, not a new model route.
    result["audit_eligible"] = (
        result["quality_status"].eq("complete_clock_coverage") & result["ds"].lt(today)
    )

    d = daily[["ds", "Daily_Inflow_Total"]].copy()
    d["ds"] = pd.to_datetime(d["ds"], format="mixed", errors="coerce")
    if not pd.api.types.is_datetime64_dtype(d["ds"].dtype) or d["ds"].isna().any():
        raise ValueError("Daily ds must contain valid naive dates")
    if d["ds"].ne(d["ds"].dt.normalize()).any() or d["ds"].duplicated().any():
        raise ValueError("Daily ds must contain unique midnight dates")
    d["Daily_Inflow_Total"] = pd.to_numeric(d["Daily_Inflow_Total"], errors="coerce")
    result = result.merge(d, on="ds", how="outer", validate="one_to_one").sort_values("ds")
    result["quality_status"] = result["quality_status"].fillna("no_hourly_source")
    result["audit_eligible"] = result["audit_eligible"].eq(True)
    result["daily_sum_matches"] = (
        result["Daily_Inflow_Total"].notna() & result["observed_inflow_sum"].notna()
        & np.isclose(result["Daily_Inflow_Total"], result["observed_inflow_sum"], rtol=0, atol=1e-9)
    )
    published = result["Daily_Inflow_Total"].notna()
    result["published_sum_mismatch"] = published & ~result["daily_sum_matches"]
    result["audit_eligible"] &= (
        published & np.isfinite(result["Daily_Inflow_Total"])
        & result["Daily_Inflow_Total"].ge(0) & result["daily_sum_matches"]
    )
    result["published_unverified_total"] = published & ~result["audit_eligible"]

    missing = pd.date_range(h["ds"].min(), h["ds"].max(), freq="h").difference(h["ds"])
    gaps = pd.DataFrame({"ds": missing})
    gaps["day"] = gaps["ds"].dt.normalize()
    gaps = gaps.merge(result[["ds", "dst_transition"]].rename(columns={"ds": "day"}),
                      on="day", how="left")
    gaps["episode"] = (gaps["ds"].diff().ne(pd.Timedelta(hours=1))).cumsum()
    ordinary = result["quality_status"].eq("incomplete_or_invalid")
    previous = result.loc[result["ds"].lt(today)]
    tail = 0
    for eligible in previous["audit_eligible"].iloc[::-1]:
        if not eligible:
            break
        tail += 1
    summary = {
        "audit_at_utc": now.tz_convert("UTC").isoformat(),
        "timezone": TIMEZONE,
        "hourly_start": h["ds"].min().isoformat(),
        "hourly_end": h["ds"].max().isoformat(),
        "hourly_rows": len(h), "duplicate_hourly_rows": int(h["duplicate"].sum()),
        "invalid_inflow_rows": int((~h["valid_inflow"]).sum()),
        "missing_internal_clock_hours": len(gaps),
        "gap_episodes": int(gaps["episode"].nunique()),
        "missing_slots_on_dst_dates": int(gaps["dst_transition"].sum()),
        "ordinary_incomplete_previous_days": int(ordinary.sum()),
        "ordinary_incomplete_days_in_1095_day_context": int(
            (ordinary & result["ds"].ge(today - pd.Timedelta(days=1095))).sum()),
        "published_ordinary_incomplete_totals": int((ordinary & published).sum()),
        "published_sum_mismatches": int(result["published_sum_mismatch"].sum()),
        "contiguous_audit_eligible_previous_days": tail,
        "latest_published_daily_date": d.loc[d["Daily_Inflow_Total"].notna(), "ds"].max().isoformat(),
        "daily_dates_without_hourly_source": int(result["quality_status"].eq("no_hourly_source").sum()),
        "interpretation": "Clock coverage audit only; no arrivals imputed or production files changed. "
        "DST and reporting-interval conventions need source verification.",
    }
    return summary, result, gaps


def audit_scoring(detail: pd.DataFrame, quality: pd.DataFrame, daily: pd.DataFrame):
    """Trace scored actuals and eight-week baselines to unverified daily totals.

    Mirror the current scorer's cutoff-only baseline selection. Do not recompute
    errors or retrospectively replace issued predictions.
    """
    required = {"ds", "data_cutoff", "actual"}
    if required - set(detail.columns):
        raise ValueError(f"Scored detail missing columns: {sorted(required - set(detail.columns))}")
    fields = [c for c in ["snapshot_name", "ds", "data_cutoff", "horizon_day", "actual"]
              if c in detail.columns]
    rows = detail[fields].copy()
    for col in ["ds", "data_cutoff"]:
        rows[col] = pd.to_datetime(rows[col], format="mixed", errors="coerce")
        if not pd.api.types.is_datetime64_dtype(rows[col].dtype) or rows[col].isna().any():
            raise ValueError("Scored dates must be valid naive dates")
        if rows[col].ne(rows[col].dt.normalize()).any():
            raise ValueError("Scored dates must be normalized calendar dates")
    totals = daily.copy()
    totals["ds"] = pd.to_datetime(totals["ds"], format="mixed")
    totals["Daily_Inflow_Total"] = pd.to_numeric(totals["Daily_Inflow_Total"], errors="coerce")
    if totals["ds"].duplicated().any():
        raise ValueError("Daily ds must contain unique dates")
    totals = totals.sort_values("ds")
    verified_dates = set(quality.loc[quality["audit_eligible"], "ds"])
    rows["actual_requires_review"] = ~rows["ds"].isin(verified_dates)
    lookup = totals.set_index("ds")["Daily_Inflow_Total"]
    rows["scored_actual_matches_daily"] = np.isclose(
        pd.to_numeric(rows["actual"], errors="coerce"), rows["ds"].map(lookup),
        rtol=0, atol=1e-9,
    )
    suspect_baselines = []
    for row in rows.itertuples(index=False):
        available = totals.loc[(totals["ds"] <= row.data_cutoff)
                               & totals["Daily_Inflow_Total"].notna()]
        baseline = available.loc[available["ds"].dt.weekday.eq(row.ds.weekday())].tail(8)
        if baseline.empty:
            baseline = available.tail(28)
        suspect_baselines.append(";".join(
            day.strftime("%Y-%m-%d") for day in baseline["ds"] if day not in verified_dates))
    rows["baseline_dates_requiring_review"] = suspect_baselines
    rows["baseline_requires_review"] = rows["baseline_dates_requiring_review"].ne("")
    summary = {
        "scored_rows": len(rows),
        "scored_actual_rows_requiring_review": int(rows["actual_requires_review"].sum()),
        "scored_actual_dates_requiring_review": sorted(
            rows.loc[rows["actual_requires_review"], "ds"].dt.strftime("%Y-%m-%d").unique()),
        "scored_baseline_rows_requiring_review": int(rows["baseline_requires_review"].sum()),
        "scored_actual_mismatches": int((~rows["scored_actual_matches_daily"]).sum()),
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", required=True, type=Path)
    parser.add_argument("--daily", required=True, type=Path)
    parser.add_argument("--scored-detail", type=Path, help="Optional current prospective detail CSV")
    parser.add_argument("--now", help="Aware ISO timestamp; defaults to actual UTC now")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    # Do not allow any diagnostic output to overwrite an input by accident.
    output_names = ["summary.json", "daily_quality.csv", "missing_hours.csv"]
    if args.scored_detail:
        output_names.append("scoring_quality.csv")
    outputs = [(args.output_dir / n).resolve() for n in output_names]
    inputs = [args.hourly, args.daily] + ([args.scored_detail] if args.scored_detail else [])
    if {p.resolve() for p in inputs}.intersection(outputs):
        raise ValueError("Audit outputs must not overwrite inputs")
    summary, quality, gaps = audit_history(
        pd.read_csv(args.hourly), pd.read_csv(args.daily),
        now=pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC"),
    )
    summary["input_sha256"] = {
        "hourly": hashlib.sha256(args.hourly.read_bytes()).hexdigest(),
        "daily": hashlib.sha256(args.daily.read_bytes()).hexdigest(),
    }
    if args.scored_detail:
        scoring_summary, scoring_rows = audit_scoring(
            pd.read_csv(args.scored_detail), quality, pd.read_csv(args.daily))
        summary.update(scoring_summary)
        summary["input_sha256"]["scored_detail"] = hashlib.sha256(
            args.scored_detail.read_bytes()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(summary, indent=2) + "\n")
    quality.to_csv(outputs[1], index=False)
    gaps.to_csv(outputs[2], index=False)
    if args.scored_detail:
        scoring_rows.to_csv(outputs[3], index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
