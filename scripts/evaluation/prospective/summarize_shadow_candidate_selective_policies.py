#!/usr/bin/env python3
"""Compare selective candidate-route policies without changing forecast routing.

For each target/horizon band in matured prospective candidate detail, this diagnostic
compares baseline everywhere, the current candidate route at every active horizon hour,
and every non-empty subset of those horizon hours. A selective policy uses the candidate
prediction only for its selected lead hours and the paired baseline prediction elsewhere.

Pooled and per-issue-date results are exploratory. Formal date-balanced policy stability
uses only complete issue dates: the full local calendar day plus the horizon band's maximum
lead must have matured. This prevents a partially observed recent day from selecting a
policy simply because only its early/easy horizons are available.

This is diagnostic only. It does not modify the pre-registered prospective collector,
production routing, or promotion status.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_shadow_candidate_complete_issue_dates import (
    LOCAL_TZ,
    _as_of_local,
    completion_time,
)


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
        LOCAL_TZ
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
        LOCAL_TZ
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


def complete_date_stability(
    by_date: pd.DataFrame,
    as_of_local: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate policy/date rows and summarize stability using complete dates only."""
    if by_date.empty:
        return by_date.copy(), pd.DataFrame()

    annotated = by_date.copy()
    annotated["issue_date_complete_at"] = [
        completion_time(date, band).isoformat()
        for date, band in zip(annotated["forecast_issue_date"], annotated["horizon_band"])
    ]
    annotated["issue_date_complete"] = [
        as_of_local >= completion_time(date, band)
        for date, band in zip(annotated["forecast_issue_date"], annotated["horizon_band"])
    ]
    annotated["policy_better_on_issue_date"] = annotated["mean_paired_mae_delta"] > 0

    complete = annotated.loc[annotated["issue_date_complete"].astype(bool)].copy()
    if complete.empty:
        return annotated, pd.DataFrame()

    group = ["target_name", "horizon_band", "policy", "candidate_hours"]
    stability = complete.groupby(group, as_index=False).agg(
        n_complete_issue_dates=("forecast_issue_date", "nunique"),
        n_unique_target_hours=("n_unique_target_hours", "sum"),
        mean_issue_date_mae_delta=("mean_paired_mae_delta", "mean"),
        median_issue_date_mae_delta=("mean_paired_mae_delta", "median"),
        mean_issue_date_mae_improvement_pct=("mae_improvement_pct", "mean"),
        median_issue_date_mae_improvement_pct=("mae_improvement_pct", "median"),
        issue_date_win_rate=("policy_better_on_issue_date", "mean"),
        harmful_issue_date_rate=(
            "policy_better_on_issue_date", lambda x: float((~x.astype(bool)).mean())
        ),
        worst_issue_date_mae_improvement_pct=("mae_improvement_pct", "min"),
    )
    stability["complete_date_rank"] = stability.groupby(
        ["target_name", "horizon_band"]
    )["mean_issue_date_mae_delta"].rank(method="dense", ascending=False)
    return (
        annotated.sort_values(
            ["target_name", "horizon_band", "policy", "forecast_issue_date"],
            ignore_index=True,
        ),
        stability.sort_values(
            ["target_name", "horizon_band", "complete_date_rank", "policy"],
            ignore_index=True,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.detail)
    pooled, by_date = summarize(frame)
    pooled.to_csv(args.output_dir / "selective-policy-grid.csv", index=False)

    as_of_local = _as_of_local(args.as_of)
    annotated_dates, complete_stability = complete_date_stability(by_date, as_of_local)
    annotated_dates.to_csv(
        args.output_dir / "selective-policy-by-issue-date.csv", index=False
    )
    complete_stability.to_csv(
        args.output_dir / "selective-policy-complete-date-stability.csv", index=False
    )

    print(f"Candidate selective-policy diagnostic: {len(pooled)} pooled policy row(s)")
    if not pooled.empty:
        winners = pooled.loc[pooled["pooled_rank"].eq(1)]
        print(winners[["target_name", "horizon_band", "policy", "mae_improvement_pct"]].to_string(index=False))
    n_visible = annotated_dates["forecast_issue_date"].nunique() if not annotated_dates.empty else 0
    n_complete = (
        annotated_dates.loc[
            annotated_dates["issue_date_complete"].astype(bool), "forecast_issue_date"
        ].nunique()
        if not annotated_dates.empty
        else 0
    )
    print(
        f"Selective-policy formal stability: {n_visible} visible issue date(s), "
        f"{n_complete} complete; partial dates excluded from formal ranking."
    )


if __name__ == "__main__":
    main()
