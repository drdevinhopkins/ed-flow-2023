#!/usr/bin/env python3
"""Leave-one-complete-issue-date-out stability check for candidate selective policies.

Diagnostic only. This consumes the candidate selective-policy per-issue-date output and,
for each target/horizon band, selects the best policy on all but one complete issue date
then evaluates that selected policy on the held-out complete date. Partial dates are
excluded before any model/policy selection step.

This does not alter candidate routing, promotion thresholds, production workflows, or
production forecast artifacts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_COMPLETE_ISSUE_DATES_FOR_LOO_CLAIM = 28


def summarize_leave_one_date_out(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_name",
        "horizon_band",
        "forecast_issue_date",
        "issue_date_complete",
        "policy",
        "mean_paired_mae_delta",
        "mae_improvement_pct",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required candidate policy columns: {sorted(missing)}")

    complete = frame.loc[frame["issue_date_complete"].astype(str).str.lower().isin({"1", "true", "yes"})].copy()
    if complete.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (target, band), group in complete.groupby(["target_name", "horizon_band"], sort=True):
        dates = sorted(group["forecast_issue_date"].astype(str).dropna().unique().tolist())
        if len(dates) < 2:
            continue
        for held_out in dates:
            train = group.loc[group["forecast_issue_date"].astype(str).ne(held_out)].copy()
            test = group.loc[group["forecast_issue_date"].astype(str).eq(held_out)].copy()
            train_summary = (
                train.groupby("policy", as_index=False)
                .agg(
                    train_mean_mae_delta=("mean_paired_mae_delta", "mean"),
                    train_median_mae_delta=("mean_paired_mae_delta", "median"),
                    train_mean_improvement_pct=("mae_improvement_pct", "mean"),
                    train_issue_date_win_rate=(
                        "mean_paired_mae_delta",
                        lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean()),
                    ),
                )
                .sort_values(
                    ["train_mean_mae_delta", "train_median_mae_delta", "policy"],
                    ascending=[False, False, True],
                    ignore_index=True,
                )
            )
            if train_summary.empty:
                continue
            selected = train_summary.iloc[0]
            policy = str(selected["policy"])
            held = test.loc[test["policy"].astype(str).eq(policy)]
            if len(held) != 1:
                raise ValueError(
                    f"Expected one held-out row for target={target} band={band} "
                    f"policy={policy} date={held_out}; got {len(held)}"
                )
            held_row = held.iloc[0]
            held_delta = float(held_row["mean_paired_mae_delta"])
            held_improvement = float(held_row["mae_improvement_pct"])
            rows.append(
                {
                    "target_name": target,
                    "horizon_band": band,
                    "held_out_issue_date": held_out,
                    "n_training_complete_issue_dates": len(dates) - 1,
                    "selected_policy": policy,
                    "train_mean_mae_delta": float(selected["train_mean_mae_delta"]),
                    "train_median_mae_delta": float(selected["train_median_mae_delta"]),
                    "train_mean_improvement_pct": float(selected["train_mean_improvement_pct"]),
                    "train_issue_date_win_rate": float(selected["train_issue_date_win_rate"]),
                    "held_out_mae_delta": held_delta,
                    "held_out_improvement_pct": held_improvement,
                    "held_out_beats_baseline": held_delta > 0,
                }
            )
    return pd.DataFrame(rows)


def summarize_selection_stability(loo: pd.DataFrame) -> pd.DataFrame:
    if loo.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (target, band), group in loo.groupby(["target_name", "horizon_band"], sort=True):
        n_folds = int(len(group))
        exploratory_stable = bool(
            group["selected_policy"].nunique() == 1
            and group["held_out_beats_baseline"].astype(bool).all()
        )
        evidence_ready = n_folds >= MIN_COMPLETE_ISSUE_DATES_FOR_LOO_CLAIM
        status = (
            "insufficient_complete_issue_dates"
            if not evidence_ready
            else "supports_stability" if exploratory_stable else "does_not_support_stability"
        )
        rows.append(
            {
                "target_name": target,
                "horizon_band": band,
                "n_leave_one_complete_date_out_folds": n_folds,
                "n_distinct_selected_policies": int(group["selected_policy"].nunique()),
                "most_common_selected_policy": str(group["selected_policy"].mode().iloc[0]),
                "selected_policy_consistency_rate": float(group["selected_policy"].value_counts(normalize=True).max()),
                "held_out_baseline_win_rate": float(group["held_out_beats_baseline"].astype(bool).mean()),
                "mean_held_out_improvement_pct": float(group["held_out_improvement_pct"].mean()),
                "median_held_out_improvement_pct": float(group["held_out_improvement_pct"].median()),
                "worst_held_out_improvement_pct": float(group["held_out_improvement_pct"].min()),
                "exploratory_selection_stable_across_complete_dates": exploratory_stable,
                "min_complete_issue_dates_for_loo_claim": MIN_COMPLETE_ISSUE_DATES_FOR_LOO_CLAIM,
                "loo_evidence_ready": evidence_ready,
                "loo_status": status,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selective_policy_by_issue_date_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.selective_policy_by_issue_date_csv)
    loo = summarize_leave_one_date_out(frame)
    if loo.empty:
        print("Need at least two complete candidate issue dates within a target/horizon band for leave-one-date-out stability.")
        return

    loo.to_csv(args.output_dir / "candidate-policy-leave-one-complete-date-out.csv", index=False)
    summary = summarize_selection_stability(loo)
    summary.to_csv(args.output_dir / "candidate-policy-leave-one-complete-date-out-summary.csv", index=False)
    print("Candidate leave-one-complete-date-out policy selection:")
    print(loo.to_string(index=False))
    print("\nCandidate policy selection stability:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
