#!/usr/bin/env python3
"""Build an issue-date-balanced cumulative trajectory for prospective weather routes.

This is diagnostic only. Each issue date receives equal weight so repeated runs on one day
cannot dominate the apparent trend. The output shows how the evidence evolves as new
independent forecast dates accumulate; it does not alter production routing or promotion
guardrails.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("by_issue_date_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _trajectory(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.sort_values("forecast_issue_date").reset_index(drop=True).copy()
    frame["cumulative_issue_dates"] = np.arange(1, len(frame) + 1)
    delta = pd.to_numeric(frame["mean_paired_mae_delta"], errors="coerce")
    pct = pd.to_numeric(frame["mae_improvement_pct"], errors="coerce")
    frame["cumulative_mean_paired_mae_delta"] = delta.expanding().mean()
    frame["cumulative_median_paired_mae_delta"] = delta.expanding().median()
    frame["cumulative_mean_mae_improvement_pct"] = pct.expanding().mean()
    frame["cumulative_median_mae_improvement_pct"] = pct.expanding().median()
    frame["cumulative_issue_date_win_rate"] = delta.gt(0).astype(float).expanding().mean()
    frame["cumulative_harmful_issue_date_rate"] = delta.lt(0).astype(float).expanding().mean()
    frame["cumulative_worst_issue_date_mae_improvement_pct"] = pct.expanding().min()
    return frame


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_date = pd.read_csv(args.by_issue_date_csv)
    required = {
        "target_name", "horizon_band", "forecast_issue_date",
        "mean_paired_mae_delta", "mae_improvement_pct",
    }
    missing = required - set(by_date.columns)
    if missing:
        raise ValueError(f"Issue-date summary missing columns: {sorted(missing)}")

    frames = []
    for _, group in by_date.groupby(["target_name", "horizon_band"], sort=True):
        frames.append(_trajectory(group))
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = args.output_dir / "weather-route-issue-date-trajectory.csv"
    output.to_csv(path, index=False)
    print(f"Wrote cumulative issue-date trajectory with {len(output)} row(s).")
    if not output.empty:
        latest = output.sort_values("forecast_issue_date").groupby(
            ["target_name", "horizon_band"], as_index=False
        ).tail(1)
        cols = [
            "target_name", "horizon_band", "forecast_issue_date", "cumulative_issue_dates",
            "cumulative_mean_mae_improvement_pct", "cumulative_issue_date_win_rate",
            "cumulative_harmful_issue_date_rate", "cumulative_worst_issue_date_mae_improvement_pct",
        ]
        print(latest[cols].to_string(index=False))


if __name__ == "__main__":
    main()
