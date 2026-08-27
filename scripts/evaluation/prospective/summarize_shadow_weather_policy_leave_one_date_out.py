#!/usr/bin/env python3
"""Leave-one-issue-date-out stability check for short-horizon weather policies.

Diagnostic only. This consumes the per-issue-date policy-grid output and asks a stricter
question than pooled or equal-weight summaries: if one independent issue date is withheld,
which policy looks best on the remaining dates, and how does that chosen policy perform on
the held-out date?

This is deliberately non-production and does not alter routing or promotion thresholds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_leave_one_date_out(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_issue_date",
        "policy",
        "mae_improvement_pct_vs_baseline",
        "mae_improvement_pct_vs_current_weather_policy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required policy-grid columns: {sorted(missing)}")

    dates = sorted(frame["forecast_issue_date"].astype(str).dropna().unique().tolist())
    if len(dates) < 2:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for held_out in dates:
        train = frame.loc[frame["forecast_issue_date"].astype(str).ne(held_out)].copy()
        test = frame.loc[frame["forecast_issue_date"].astype(str).eq(held_out)].copy()
        if train.empty or test.empty:
            continue

        train_summary = (
            train.groupby("policy", as_index=False)
            .agg(
                train_mean_improvement_pct=("mae_improvement_pct_vs_baseline", "mean"),
                train_median_improvement_pct=("mae_improvement_pct_vs_baseline", "median"),
                train_issue_date_win_rate=(
                    "mae_improvement_pct_vs_baseline",
                    lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()),
                ),
                train_mean_vs_current_pct=("mae_improvement_pct_vs_current_weather_policy", "mean"),
            )
        )
        train_summary = train_summary.sort_values(
            ["train_mean_improvement_pct", "train_median_improvement_pct", "policy"],
            ascending=[False, False, True],
            ignore_index=True,
        )
        selected = train_summary.iloc[0]
        policy = str(selected["policy"])
        held = test.loc[test["policy"].eq(policy)]
        if len(held) != 1:
            raise ValueError(
                f"Expected exactly one held-out row for policy={policy} date={held_out}; got {len(held)}"
            )
        held_row = held.iloc[0]
        held_improvement = float(held_row["mae_improvement_pct_vs_baseline"])
        held_vs_current = float(held_row["mae_improvement_pct_vs_current_weather_policy"])
        rows.append(
            {
                "held_out_issue_date": held_out,
                "n_training_issue_dates": len(dates) - 1,
                "selected_policy": policy,
                "train_mean_improvement_pct": float(selected["train_mean_improvement_pct"]),
                "train_median_improvement_pct": float(selected["train_median_improvement_pct"]),
                "train_issue_date_win_rate": float(selected["train_issue_date_win_rate"]),
                "train_mean_vs_current_pct": float(selected["train_mean_vs_current_pct"]),
                "held_out_improvement_pct": held_improvement,
                "held_out_vs_current_pct": held_vs_current,
                "held_out_beats_baseline": held_improvement > 0,
                "held_out_beats_current": held_vs_current > 0,
            }
        )

    return pd.DataFrame(rows)


def summarize_selection_stability(loo: pd.DataFrame) -> pd.DataFrame:
    if loo.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "n_leave_one_date_out_folds": int(len(loo)),
                "n_distinct_selected_policies": int(loo["selected_policy"].nunique()),
                "most_common_selected_policy": str(loo["selected_policy"].mode().iloc[0]),
                "selected_policy_consistency_rate": float(
                    loo["selected_policy"].value_counts(normalize=True).max()
                ),
                "held_out_baseline_win_rate": float(loo["held_out_beats_baseline"].astype(bool).mean()),
                "held_out_current_win_rate": float(loo["held_out_beats_current"].astype(bool).mean()),
                "mean_held_out_improvement_pct": float(loo["held_out_improvement_pct"].mean()),
                "median_held_out_improvement_pct": float(loo["held_out_improvement_pct"].median()),
                "worst_held_out_improvement_pct": float(loo["held_out_improvement_pct"].min()),
                "selection_stable_across_dates": bool(
                    loo["selected_policy"].nunique() == 1
                    and loo["held_out_beats_baseline"].astype(bool).all()
                ),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_grid_by_issue_date_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.policy_grid_by_issue_date_csv)
    loo = summarize_leave_one_date_out(frame)
    if loo.empty:
        print("Need at least two issue dates for leave-one-date-out policy stability.")
        return

    loo.to_csv(args.output_dir / "weather-policy-grid-leave-one-date-out.csv", index=False)
    stability = summarize_selection_stability(loo)
    stability.to_csv(args.output_dir / "weather-policy-grid-leave-one-date-out-summary.csv", index=False)

    print("Leave-one-issue-date-out weather policy selection:")
    print(loo.to_string(index=False))
    print("\nSelection stability summary:")
    print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
