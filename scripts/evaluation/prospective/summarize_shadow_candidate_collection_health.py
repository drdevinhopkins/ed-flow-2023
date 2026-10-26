#!/usr/bin/env python3
"""Summarize continuity and structural health of the prospective candidate archive.

Diagnostic only: this never changes routing or production forecast artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

EXPECTED_ROWS_PER_RUN = 120
BURST_GAP_MINUTES = 10.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    parser.add_argument("--output", default="candidate-collection-health.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.archive)
    required = {"forecast_run_id", "target_name", "horizon_hour"}
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    # The prospective archive uses forecast_issued_at. Keep the older issued_at
    # alias accepted so synthetic/legacy diagnostic inputs remain readable.
    timestamp_column = (
        "forecast_issued_at"
        if "forecast_issued_at" in frame.columns
        else "issued_at"
        if "issued_at" in frame.columns
        else None
    )
    if timestamp_column is None:
        raise SystemExit("Missing required timestamp column: forecast_issued_at")
    frame["issued_at"] = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")

    per_run = (
        frame.groupby("forecast_run_id", as_index=False)
        .agg(
            issued_at=("issued_at", "min"),
            n_rows=("forecast_run_id", "size"),
            n_targets=("target_name", "nunique"),
            min_horizon=("horizon_hour", "min"),
            max_horizon=("horizon_hour", "max"),
        )
        .sort_values("issued_at")
        .reset_index(drop=True)
    )
    per_run["gap_hours_from_prior_run"] = per_run["issued_at"].diff().dt.total_seconds() / 3600.0
    per_run["issue_hour_utc"] = per_run["issued_at"].dt.floor("h")
    per_run["structurally_complete"] = (
        per_run["n_rows"].eq(EXPECTED_ROWS_PER_RUN)
        & per_run["n_targets"].eq(5)
        & per_run["min_horizon"].eq(1)
        & per_run["max_horizon"].eq(24)
    )

    gaps = per_run["gap_hours_from_prior_run"].dropna()
    issue_hour_counts = per_run.groupby("issue_hour_utc").size()
    short_gap_threshold_hours = BURST_GAP_MINUTES / 60.0
    short_gaps = gaps < short_gap_threshold_hours
    summary = pd.DataFrame([
        {
            "n_runs": len(per_run),
            "first_issued_at": per_run["issued_at"].min(),
            "last_issued_at": per_run["issued_at"].max(),
            "span_hours": (per_run["issued_at"].max() - per_run["issued_at"].min()).total_seconds() / 3600.0 if len(per_run) > 1 else 0.0,
            "structurally_complete_runs": int(per_run["structurally_complete"].sum()),
            "incomplete_runs": int((~per_run["structurally_complete"]).sum()),
            "n_unique_issue_hours": int(per_run["issue_hour_utc"].nunique()),
            "max_runs_per_issue_hour": int(issue_hour_counts.max()) if not issue_hour_counts.empty else 0,
            "mean_runs_per_issue_hour": float(issue_hour_counts.mean()) if not issue_hour_counts.empty else float("nan"),
            "gaps_under_10m": int(short_gaps.sum()),
            "fraction_interrun_gaps_under_10m": float(short_gaps.mean()) if not gaps.empty else float("nan"),
            "median_gap_hours": gaps.median() if not gaps.empty else float("nan"),
            "max_gap_hours": gaps.max() if not gaps.empty else float("nan"),
            "gaps_over_2h": int((gaps > 2.0).sum()),
            "duplicate_issue_timestamps": int(per_run["issued_at"].duplicated().sum()),
        }
    ])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    per_run.to_csv(output.with_name(output.stem + "-by-run.csv"), index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
