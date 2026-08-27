#!/usr/bin/env python3
"""Compare all short-horizon weather-routing subsets for one target/band.

Diagnostic only. For each non-empty subset of candidate lead hours, this script builds a
counterfactual that uses the weather prediction on those hours and the paired baseline
prediction on all other hours. It also includes all-baseline and the current weather policy.

In addition to pooled performance, it emits an issue-date-balanced stability table so a
policy cannot look attractive merely because one date contributes many repeated intraday
forecasts. Policy-selection labels remain explicitly non-actionable until the same
prospective evidence thresholds used elsewhere are met.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ISSUE_DATES = 28
MIN_PROSPECTIVE_SPAN_DAYS = 56.0
MIN_UNIQUE_TARGET_HOURS = 100
MIN_ISSUE_DATE_WIN_RATE = 0.55


def metrics(frame: pd.DataFrame, prediction: pd.Series, label: str, hours: tuple[int, ...] | None) -> dict[str, object]:
    error = prediction - frame["actual"]
    abs_error = error.abs()
    baseline_abs = (frame["baseline_prediction"] - frame["actual"]).abs()
    delta = baseline_abs - abs_error
    baseline_mae = float(baseline_abs.mean())
    mae = float(abs_error.mean())
    return {
        "policy": label,
        "weather_hours": "" if hours is None else ",".join(map(str, hours)),
        "n": int(len(frame)),
        "n_runs": int(frame["forecast_run_id"].nunique()),
        "n_issue_dates": int(frame["forecast_issue_date"].nunique()),
        "n_unique_target_hours": int(frame["ds"].nunique()),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement_pct_vs_baseline": ((baseline_mae - mae) / baseline_mae * 100.0) if baseline_mae else np.nan,
        "mean_paired_mae_delta_vs_baseline": float(delta.mean()),
        "median_paired_mae_delta_vs_baseline": float(delta.median()),
        "win_rate_vs_baseline": float((delta > 0).mean()),
        "bias": float(error.mean()),
    }


def summarize(frame: pd.DataFrame, candidate_hours: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(metrics(frame, frame["baseline_prediction"], "all_baseline", tuple()))
    rows.append(metrics(frame, frame["weather_prediction"], "current_weather_policy", None))
    for size in range(1, len(candidate_hours) + 1):
        for hours in combinations(candidate_hours, size):
            use_weather = frame["horizon_hour"].isin(hours)
            prediction = pd.Series(
                np.where(use_weather, frame["weather_prediction"], frame["baseline_prediction"]),
                index=frame.index,
            )
            rows.append(metrics(frame, prediction, "weather_hours_" + "_".join(map(str, hours)), hours))
    result = pd.DataFrame(rows)
    current_mae = float(result.loc[result["policy"].eq("current_weather_policy"), "mae"].iloc[0])
    result["mae_improvement_pct_vs_current_weather_policy"] = (
        (current_mae - result["mae"]) / current_mae * 100.0 if current_mae else np.nan
    )
    return result.sort_values(["mae", "policy"], ignore_index=True)


def summarize_issue_date_stability(
    by_date: pd.DataFrame,
    *,
    n_unique_target_hours: int,
    prospective_span_days: float,
) -> pd.DataFrame:
    """Give each issue date one vote and pre-register when policy selection is evaluable."""
    rows: list[dict[str, object]] = []
    for (policy, weather_hours), group in by_date.groupby(["policy", "weather_hours"], dropna=False):
        improvement = pd.to_numeric(group["mae_improvement_pct_vs_baseline"], errors="coerce").dropna()
        delta = pd.to_numeric(group["mean_paired_mae_delta_vs_baseline"], errors="coerce").dropna()
        vs_current = pd.to_numeric(group["mae_improvement_pct_vs_current_weather_policy"], errors="coerce").dropna()
        if improvement.empty:
            continue
        n_issue_dates = int(group["forecast_issue_date"].nunique())
        evidence_ready = (
            n_issue_dates >= MIN_ISSUE_DATES
            and n_unique_target_hours >= MIN_UNIQUE_TARGET_HOURS
            and prospective_span_days >= MIN_PROSPECTIVE_SPAN_DAYS
        )
        mean_improvement = float(improvement.mean())
        median_improvement = float(improvement.median())
        issue_date_win_rate = float((improvement > 0).mean())
        mean_vs_current = float(vs_current.mean()) if not vs_current.empty else np.nan
        is_baseline = policy == "all_baseline"
        is_current = policy == "current_weather_policy"
        directional_candidate = (
            (not is_baseline)
            and mean_improvement > 0
            and median_improvement > 0
            and issue_date_win_rate >= MIN_ISSUE_DATE_WIN_RATE
            and (is_current or (pd.notna(mean_vs_current) and mean_vs_current > 0))
        )
        rows.append({
            "policy": policy,
            "weather_hours": "" if pd.isna(weather_hours) else str(weather_hours),
            "n_issue_dates": n_issue_dates,
            "n_unique_target_hours": n_unique_target_hours,
            "prospective_span_days": prospective_span_days,
            "issue_date_mean_mae_improvement_pct": mean_improvement,
            "issue_date_median_mae_improvement_pct": median_improvement,
            "issue_date_mean_paired_mae_delta": float(delta.mean()) if not delta.empty else np.nan,
            "issue_date_win_rate": issue_date_win_rate,
            "harmful_issue_date_rate": float((improvement < 0).mean()),
            "worst_issue_date_mae_improvement_pct": float(improvement.min()),
            "p10_issue_date_mae_improvement_pct": float(improvement.quantile(0.10)),
            "issue_date_mean_improvement_pct_vs_current_weather_policy": mean_vs_current,
            "selection_evidence_ready": evidence_ready,
            "directional_policy_candidate": directional_candidate,
            "policy_selection_status": (
                "evaluable_candidate" if evidence_ready and directional_candidate
                else "evaluable_not_candidate" if evidence_ready
                else "collecting"
            ),
            "issue_dates_remaining": max(0, MIN_ISSUE_DATES - n_issue_dates),
            "unique_target_hours_remaining": max(0, MIN_UNIQUE_TARGET_HOURS - n_unique_target_hours),
            "span_days_remaining": max(0.0, MIN_PROSPECTIVE_SPAN_DAYS - prospective_span_days),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        [
            "directional_policy_candidate",
            "issue_date_mean_mae_improvement_pct",
            "issue_date_win_rate",
            "worst_issue_date_mae_improvement_pct",
            "policy",
        ],
        ascending=[False, False, False, False, True],
        ignore_index=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="Total_TBS")
    parser.add_argument("--horizon-band", default="h01_04")
    parser.add_argument("--candidate-hours", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.detail_csv)
    required = {
        "forecast_run_id", "forecast_issued_at", "ds", "target_name", "horizon_band",
        "horizon_hour", "actual", "baseline_prediction", "weather_prediction",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required scorer columns: {sorted(missing)}")

    frame["forecast_issued_at"] = pd.to_datetime(frame["forecast_issued_at"], utc=True, errors="coerce")
    frame["forecast_issue_date"] = frame["forecast_issued_at"].dt.date.astype("string")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame["horizon_hour"] = pd.to_numeric(frame["horizon_hour"], errors="coerce").astype("Int64")
    frame = frame.loc[
        frame["target_name"].eq(args.target)
        & frame["horizon_band"].eq(args.horizon_band)
    ].dropna(subset=["forecast_issued_at", "ds", "horizon_hour", "actual"]).copy()
    if frame.empty:
        print("No matured rows for requested policy-grid diagnostic.")
        return

    candidate_hours = sorted(set(args.candidate_hours))
    summary = summarize(frame, candidate_hours)
    summary["target_name"] = args.target
    summary["horizon_band"] = args.horizon_band
    summary.to_csv(args.output_dir / "weather-policy-grid.csv", index=False)

    date_frames = []
    for issue_date, group in frame.groupby("forecast_issue_date"):
        date_summary = summarize(group, candidate_hours)
        date_summary["forecast_issue_date"] = issue_date
        date_frames.append(date_summary)
    by_date = pd.concat(date_frames, ignore_index=True)
    by_date.to_csv(args.output_dir / "weather-policy-grid-by-issue-date.csv", index=False)

    first_issued = frame["forecast_issued_at"].min()
    last_issued = frame["forecast_issued_at"].max()
    span_days = float((last_issued - first_issued).total_seconds() / 86400.0)
    stability = summarize_issue_date_stability(
        by_date,
        n_unique_target_hours=int(frame["ds"].nunique()),
        prospective_span_days=span_days,
    )
    if not stability.empty:
        stability["target_name"] = args.target
        stability["horizon_band"] = args.horizon_band
        stability.to_csv(args.output_dir / "weather-policy-grid-issue-date-stability.csv", index=False)

    print("Short-horizon weather policy grid (best pooled MAE first):")
    print(summary.to_string(index=False))
    print("\nTop policy by issue date:")
    winners = by_date.sort_values(["forecast_issue_date", "mae", "policy"]).groupby("forecast_issue_date", as_index=False).first()
    print(winners[["forecast_issue_date", "policy", "weather_hours", "mae", "mae_improvement_pct_vs_baseline"]].to_string(index=False))
    if not stability.empty:
        print("\nIssue-date-balanced policy stability (selection remains collecting until pre-registered evidence thresholds are met):")
        print(stability.to_string(index=False))


if __name__ == "__main__":
    main()
