#!/usr/bin/env python3
"""Summarize prospective candidate routes using only complete issue dates.

Repeated intraday shadow runs are correlated and a recent issue date can be only partly
matured. This diagnostic keeps partial dates visible but excludes them from formal
issue-date-balanced evidence until the full local calendar day plus the horizon band's
maximum lead has matured.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Toronto"
BAND_MAX_HORIZON = {
    "h01_04": 4,
    "h05_08": 8,
    "h09_12": 12,
    "h13_24": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", type=str, default=None)
    return parser.parse_args()


def _as_of_local(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=LOCAL_TZ)
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return stamp.tz_convert(LOCAL_TZ)


def completion_time(issue_date: object, horizon_band: str) -> pd.Timestamp:
    if horizon_band not in BAND_MAX_HORIZON:
        raise ValueError(f"Unknown horizon band: {horizon_band}")
    day = pd.Timestamp(issue_date)
    if day.tzinfo is None:
        day = day.tz_localize(LOCAL_TZ)
    else:
        day = day.tz_convert(LOCAL_TZ)
    return day.normalize() + pd.Timedelta(days=1, hours=BAND_MAX_HORIZON[horizon_band])


def by_issue_date(active: pd.DataFrame, as_of_local: pd.Timestamp) -> pd.DataFrame:
    group = ["target_name", "horizon_band", "candidate_scenario", "forecast_issue_date"]
    out = active.groupby(group, as_index=False).agg(
        n=("actual", "size"),
        n_runs=("forecast_run_id", "nunique"),
        n_unique_target_hours=("target_ds", "nunique"),
        baseline_mae=("baseline_absolute_error", "mean"),
        candidate_mae=("candidate_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        candidate_win_rate=("candidate_wins", "mean"),
    )
    out["mae_improvement_pct"] = np.where(
        out["baseline_mae"].ne(0),
        out["mean_paired_mae_delta"] / out["baseline_mae"] * 100,
        np.nan,
    )
    out["candidate_better_on_issue_date"] = out["mean_paired_mae_delta"] > 0
    out["issue_date_complete_at"] = [
        completion_time(date, band).isoformat()
        for date, band in zip(out["forecast_issue_date"], out["horizon_band"])
    ]
    out["issue_date_complete"] = [
        as_of_local >= completion_time(date, band)
        for date, band in zip(out["forecast_issue_date"], out["horizon_band"])
    ]
    return out.sort_values(group, ignore_index=True)


def balanced_complete(by_date: pd.DataFrame) -> pd.DataFrame:
    complete = by_date.loc[by_date["issue_date_complete"].astype(bool)].copy()
    if complete.empty:
        return pd.DataFrame()
    group = ["target_name", "horizon_band", "candidate_scenario"]
    out = complete.groupby(group, as_index=False).agg(
        n_complete_issue_dates=("forecast_issue_date", "nunique"),
        n_unique_target_hours=("n_unique_target_hours", "sum"),
        mean_issue_date_mae_delta=("mean_paired_mae_delta", "mean"),
        median_issue_date_mae_delta=("mean_paired_mae_delta", "median"),
        mean_issue_date_mae_improvement_pct=("mae_improvement_pct", "mean"),
        median_issue_date_mae_improvement_pct=("mae_improvement_pct", "median"),
        issue_date_win_rate=("candidate_better_on_issue_date", "mean"),
        harmful_issue_date_rate=("candidate_better_on_issue_date", lambda x: float((~x.astype(bool)).mean())),
        worst_issue_date_mae_improvement_pct=("mae_improvement_pct", "min"),
    )
    return out.sort_values(group, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    required = {
        "target_name", "horizon_band", "candidate_scenario", "forecast_issue_date",
        "candidate_route_active", "forecast_run_id", "target_ds", "actual",
        "baseline_absolute_error", "candidate_absolute_error", "paired_absolute_error_delta",
        "candidate_wins",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Candidate detail missing columns: {sorted(missing)}")
    active = detail.loc[detail["candidate_route_active"].astype(bool)].copy()
    if active.empty:
        print("No matured active candidate-route rows; complete-date summary skipped.")
        return
    as_of_local = _as_of_local(args.as_of)
    issue = by_issue_date(active, as_of_local)
    issue.to_csv(args.output_dir / "candidate-by-issue-date-completeness.csv", index=False)
    balanced = balanced_complete(issue)
    balanced.to_csv(args.output_dir / "candidate-complete-issue-date-balanced.csv", index=False)
    print(
        f"Candidate complete-date diagnostic: {issue['forecast_issue_date'].nunique()} visible issue date(s), "
        f"{int(issue.loc[issue['issue_date_complete'].astype(bool), 'forecast_issue_date'].nunique())} complete."
    )
    if not balanced.empty:
        print(balanced.to_string(index=False))


if __name__ == "__main__":
    main()
