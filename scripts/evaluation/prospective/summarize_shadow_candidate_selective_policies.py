#!/usr/bin/env python3
"""Compare selective candidate-route policies without changing forecast routing.

For each target/horizon band in matured prospective candidate detail, this diagnostic
compares baseline everywhere, the current candidate route at every active horizon hour,
and every non-empty subset of those horizon hours. A selective policy uses the candidate
prediction only for its selected lead hours and the paired baseline prediction elsewhere.

This is exploratory only. It does not modify the pre-registered prospective collector,
production routing, or promotion status.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def _col(frame: pd.DataFrame, canonical: str, *aliases: str) -> str:
    for name in (canonical, *aliases):
        if name in frame.columns:
            return name
    raise ValueError(f"Missing required column {canonical}; tried {(canonical, *aliases)}")


def _policy_name(hours: tuple[int, ...]) -> str:
    if not hours:
        return "baseline"
    return "candidate_hours_" + "_".join(str(hour) for hour in hours)


def _summarize_policy(frame: pd.DataFrame, hours: tuple[int, ...]) -> dict[str, object]:
    baseline_pred = frame[_col(frame, "baseline_prediction")].astype(float)
    candidate_pred = frame[_col(frame, "candidate_prediction")].astype(float)
    actual = frame["actual"].astype(float)
    use_candidate = frame["horizon_hour"].astype(int).isin(hours)
    prediction = baseline_pred.where(~use_candidate, candidate_pred)
    abs_error = (prediction - actual).abs()
    baseline_abs = (baseline_pred - actual).abs()
    paired_delta = baseline_abs - abs_error
    baseline_mae = float(baseline_abs.mean())
    policy_mae = float(abs_error.mean())
    issue_dates = pd.to_datetime(frame["forecast_issued_at"], utc=True).dt.tz_convert(
        "America/Toronto"
    ).dt.date
    return {
        "policy": _policy_name(hours),
        "candidate_hours": ",".join(str(hour) for hour in hours),
        "n": int(len(frame)),
        "n_runs": int(frame["forecast_run_id"].nunique()),
        "n_issue_dates": int(issue_dates.nunique()),
        "n_unique_target_hours": int(pd.to_datetime(frame["target_ds"]).nunique()),
        "baseline_mae": baseline_mae,
        "policy_mae": policy_mae,
        "mean_paired_mae_delta": float(paired_delta.mean()),
        "median_paired_mae_delta": float(paired_delta.median()),
        "policy_win_rate": float((paired_delta > 0).mean()),
        "mae_improvement_pct": (
            (baseline_mae - policy_mae) / baseline_mae * 100 if baseline_mae else np.nan
        ),
    }


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "target_name", "horizon_band", "horizon_hour", "actual",
        "forecast_run_id", "forecast_issued_at", "target_ds",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate detail missing columns: {sorted(missing)}")
    _col(frame, "baseline_prediction")
    _col(frame, "candidate_prediction")

    # Only rows where the collector actually used a non-baseline candidate route provide
    # a meaningful counterfactual candidate prediction.
    if "candidate_route_active" in frame.columns:
        active = frame["candidate_route_active"].astype(str).str.lower().isin(
            {"1", "true", "yes"}
        )
    elif "candidate_scenario" in frame.columns:
        active = ~frame["candidate_scenario"].astype(str).eq("baseline")
    else:
        raise ValueError("Candidate detail lacks route-active/scenario information")
    frame = frame.loc[active].copy()
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()

    pooled_rows: list[dict[str, object]] = []
    date_rows: list[dict[str, object]] = []
    local_dates = pd.to_datetime(frame["forecast_issued_at"], utc=True).dt.tz_convert(
        "America/Toronto"
    ).dt.date
    frame["forecast_issue_date"] = local_dates.astype(str)

    for (target, band), group in frame.groupby(["target_name", "horizon_band"], sort=True):
        hours = tuple(sorted(group["horizon_hour"].astype(int).unique()))
        policies = [tuple()]
        for size in range(1, len(hours) + 1):
            policies.extend(itertools.combinations(hours, size))
        for policy_hours in policies:
            row = _summarize_policy(group, policy_hours)
            row.update({"target_name": target, "horizon_band": band})
            pooled_rows.append(row)
            for issue_date, date_group in group.groupby("forecast_issue_date", sort=True):
                date_row = _summarize_policy(date_group, policy_hours)
                date_row.update(
                    {
                        "target_name": target,
                        "horizon_band": band,
                        "forecast_issue_date": issue_date,
                    }
                )
                date_rows.append(date_row)

    pooled = pd.DataFrame(pooled_rows)
    by_date = pd.DataFrame(date_rows)
    if not pooled.empty:
        pooled = pooled.sort_values(
            ["target_name", "horizon_band", "policy_mae", "policy"], ignore_index=True
        )
        pooled["pooled_rank"] = pooled.groupby(
            ["target_name", "horizon_band"]
        )["policy_mae"].rank(method="dense")
    return pooled, by_date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.detail)
    pooled, by_date = summarize(frame)
    pooled.to_csv(args.output_dir / "selective-policy-grid.csv", index=False)
    by_date.to_csv(args.output_dir / "selective-policy-by-issue-date.csv", index=False)
    print(f"Candidate selective-policy diagnostic: {len(pooled)} pooled policy row(s)")
    if not pooled.empty:
        winners = pooled.loc[pooled["pooled_rank"].eq(1)]
        print(winners[["target_name", "horizon_band", "policy", "mae_improvement_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
