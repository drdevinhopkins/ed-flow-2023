#!/usr/bin/env python3
"""Build a cumulative complete-issue-date trajectory for prospective candidate routes.

Diagnostic only. Partial issue dates are excluded so repeated intraday runs and incompletely
matured recent dates cannot dominate the apparent trend. Each complete issue date receives
one equal vote; this does not alter production routing or promotion guardrails.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("by_issue_date_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _trajectory(group: pd.DataFrame) -> pd.DataFrame:
    frame = group.sort_values("forecast_issue_date").reset_index(drop=True).copy()
    delta = pd.to_numeric(frame["mean_paired_mae_delta"], errors="coerce")
    pct = pd.to_numeric(frame["mae_improvement_pct"], errors="coerce")
    frame["cumulative_complete_issue_dates"] = np.arange(1, len(frame) + 1)
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
    by_date = pd.read_csv(args.by_issue_date_csv)
    required = {
        "target_name", "horizon_band", "candidate_scenario", "forecast_issue_date",
        "issue_date_complete", "mean_paired_mae_delta", "mae_improvement_pct",
    }
    missing = required - set(by_date.columns)
    if missing:
        raise ValueError(f"Candidate issue-date summary missing columns: {sorted(missing)}")

    complete = by_date.loc[by_date["issue_date_complete"].astype(bool)].copy()
    frames = []
    group_cols = ["target_name", "horizon_band", "candidate_scenario"]
    for _, group in complete.groupby(group_cols, sort=True):
        frames.append(_trajectory(group))
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote candidate complete-date trajectory with {len(output)} row(s).")
    if not output.empty:
        latest = output.sort_values("forecast_issue_date").groupby(group_cols, as_index=False).tail(1)
        cols = group_cols + [
            "forecast_issue_date", "cumulative_complete_issue_dates",
            "cumulative_mean_mae_improvement_pct", "cumulative_issue_date_win_rate",
            "cumulative_harmful_issue_date_rate", "cumulative_worst_issue_date_mae_improvement_pct",
        ]
        print(latest[cols].to_string(index=False))


if __name__ == "__main__":
    main()
