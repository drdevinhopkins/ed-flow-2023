#!/usr/bin/env python3
"""Check candidate-route performance after equal-weighting forecast issue hours.

PR-triggered shadow collection can create several forecast runs within the same UTC hour.
Those burst runs are useful for prospective collection but should not receive extra weight in
an independence sensitivity analysis. For each target/horizon/candidate route this diagnostic
keeps either the earliest or latest forecast run in each forecast issue hour, summarizes that
run's matured active rows, and then gives every issue hour equal weight.

This is diagnostic only. It does not alter candidate routing, promotion criteria, production
workflows, or production forecast artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = ["target_name", "horizon_band", "candidate_scenario"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "validation/prospective-candidates/latest-score/"
            "candidate-route-issue-hour-sensitivity.csv"
        ),
    )
    return parser.parse_args()


def _select_one_run_per_issue_hour(frame: pd.DataFrame, keep: str) -> pd.DataFrame:
    run_times = (
        frame[[*GROUP, "issue_hour", "forecast_run_id", "forecast_issued_at"]]
        .drop_duplicates()
        .sort_values([*GROUP, "issue_hour", "forecast_issued_at", "forecast_run_id"])
    )
    chosen = run_times.drop_duplicates([*GROUP, "issue_hour"], keep=keep)
    return frame.merge(
        chosen[[*GROUP, "issue_hour", "forecast_run_id"]],
        on=[*GROUP, "issue_hour", "forecast_run_id"],
        how="inner",
    )


def _summarize_selected(frame: pd.DataFrame, selection: str) -> pd.DataFrame:
    # First collapse all matured active rows in a selected run to a run-level score.
    # Then average those run scores, so each forecast issue hour contributes one vote.
    by_hour = frame.groupby([*GROUP, "issue_hour"], as_index=False).agg(
        baseline_mae=("baseline_absolute_error", "mean"),
        candidate_mae=("candidate_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        candidate_win_rate=("candidate_wins", "mean"),
        baseline_bias=("baseline_error", "mean"),
        candidate_bias=("candidate_error", "mean"),
        n_matured_rows=("target_ds", "size"),
    )
    out = by_hour.groupby(GROUP, as_index=False).agg(
        n_unique_issue_hours=("issue_hour", "nunique"),
        n_matured_rows=("n_matured_rows", "sum"),
        baseline_mae=("baseline_mae", "mean"),
        candidate_mae=("candidate_mae", "mean"),
        mean_paired_mae_delta=("mean_paired_mae_delta", "mean"),
        median_issue_hour_mae_delta=("mean_paired_mae_delta", "median"),
        candidate_win_rate=("candidate_win_rate", "mean"),
        baseline_bias=("baseline_bias", "mean"),
        candidate_bias=("candidate_bias", "mean"),
    )
    out["mae_improvement_pct"] = (
        (out["baseline_mae"] - out["candidate_mae"])
        / out["baseline_mae"].replace(0, np.nan)
        * 100.0
    )
    out.insert(3, "selection", selection)
    return out


def summarize_issue_hour_sensitivity(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_run_id",
        "forecast_issued_at",
        "target_ds",
        "target_name",
        "horizon_band",
        "candidate_scenario",
        "candidate_route_active",
        "baseline_absolute_error",
        "candidate_absolute_error",
        "paired_absolute_error_delta",
        "candidate_wins",
        "baseline_error",
        "candidate_error",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    frame = detail.loc[detail["candidate_route_active"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["forecast_issued_at"] = pd.to_datetime(
        frame["forecast_issued_at"], utc=True, errors="coerce"
    )
    if frame["forecast_issued_at"].isna().any():
        raise ValueError("Invalid forecast_issued_at in candidate detail")
    frame["issue_hour"] = frame["forecast_issued_at"].dt.floor("h")

    earliest = _select_one_run_per_issue_hour(frame, "first")
    latest = _select_one_run_per_issue_hour(frame, "last")
    result = pd.concat(
        [
            _summarize_selected(earliest, "earliest_run_per_issue_hour"),
            _summarize_selected(latest, "latest_run_per_issue_hour"),
        ],
        ignore_index=True,
    )

    pivot = result.pivot_table(
        index=GROUP,
        columns="selection",
        values="mae_improvement_pct",
        aggfunc="first",
    ).reset_index()
    early = "earliest_run_per_issue_hour"
    late = "latest_run_per_issue_hour"
    if early in pivot.columns and late in pivot.columns:
        pivot["selection_direction_agrees"] = (
            np.sign(pivot[early].fillna(0)) == np.sign(pivot[late].fillna(0))
        )
        result = result.merge(
            pivot[[*GROUP, "selection_direction_agrees"]],
            on=GROUP,
            how="left",
        )
    else:
        result["selection_direction_agrees"] = False

    return result.sort_values([*GROUP, "selection"])


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    summary = summarize_issue_hour_sensitivity(detail)
    summary.to_csv(args.output, index=False)
    if summary.empty:
        print("No matured candidate-route rows yet; issue-hour sensitivity is empty.")
    else:
        print("Prospective candidate equal-weight issue-hour sensitivity:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
