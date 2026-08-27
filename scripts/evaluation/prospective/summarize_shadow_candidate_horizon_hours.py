#!/usr/bin/env python3
"""Summarize prospective candidate-vs-baseline performance by exact lead hour.

Diagnostic only. This does not change candidate routing or promotion decisions; it is
intended to reveal within-band heterogeneity (for example, a 1-4h route that helps at
h3-h4 but not h1-h2).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _coverage(frame: pd.DataFrame, prefix: str) -> float:
    lower = f"{prefix}_lower"
    upper = f"{prefix}_upper"
    if lower not in frame or upper not in frame:
        return float("nan")
    valid = frame[[lower, upper, "actual"]].notna().all(axis=1)
    if not valid.any():
        return float("nan")
    return float(
        frame.loc[valid, "actual"].between(
            frame.loc[valid, lower], frame.loc[valid, upper], inclusive="both"
        ).mean()
    )


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_name", "horizon_hour", "candidate_scenario", "actual",
        "baseline_prediction", "candidate_prediction", "baseline_abs_error",
        "candidate_abs_error", "baseline_error", "candidate_error",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate detail is missing columns: {sorted(missing)}")

    active = frame.loc[
        frame["candidate_scenario"].notna()
        & frame["candidate_scenario"].ne("baseline")
    ].copy()
    if active.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (target, hour, scenario), group in active.groupby(
        ["target_name", "horizon_hour", "candidate_scenario"], sort=True
    ):
        baseline_mae = float(group["baseline_abs_error"].mean())
        candidate_mae = float(group["candidate_abs_error"].mean())
        delta = group["baseline_abs_error"] - group["candidate_abs_error"]
        rows.append(
            {
                "target_name": target,
                "horizon_hour": int(hour),
                "candidate_scenario": scenario,
                "n": len(group),
                "n_runs": group["forecast_run_id"].nunique() if "forecast_run_id" in group else np.nan,
                "n_issue_dates": group["forecast_issue_date"].nunique() if "forecast_issue_date" in group else np.nan,
                "n_unique_target_hours": pd.to_datetime(group["target_ds"]).nunique() if "target_ds" in group else np.nan,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
                "mean_paired_mae_delta": float(delta.mean()),
                "median_paired_mae_delta": float(delta.median()),
                "candidate_win_rate": float((delta > 0).mean()),
                "baseline_bias": float(group["baseline_error"].mean()),
                "candidate_bias": float(group["candidate_error"].mean()),
                "baseline_interval_coverage": _coverage(group, "baseline"),
                "candidate_interval_coverage": _coverage(group, "candidate"),
                "mae_improvement_pct": (
                    (baseline_mae - candidate_mae) / baseline_mae * 100
                    if baseline_mae else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["target_name", "horizon_hour", "candidate_scenario"], ignore_index=True
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("candidate-horizon-hour-summary.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.detail_csv)
    result = summarize(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Candidate horizon-hour diagnostic: {len(result)} row(s)")
    if not result.empty:
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
