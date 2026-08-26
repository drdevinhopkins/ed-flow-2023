#!/usr/bin/env python3
"""Estimate uncertainty for prospective weather-route MAE improvement.

Rows from the same forecast issue are correlated across horizons, so confidence intervals
are produced by resampling issue dates rather than individual forecast rows. This output
is diagnostic only and does not alter promotion guardrails or routing.

Bootstrap intervals are computed as soon as at least two issue dates exist, but they are
explicitly exploratory until the prospective validation plan's 28-issue-date evidence
threshold is reached. This prevents a very small number of clusters from being described
as confirmatory evidence merely because their bootstrap interval excludes zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOTSTRAP = 5000
SEED = 20260826
MIN_CONFIDENCE_ISSUE_DATES = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _bootstrap_group(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    issue_dates = pd.Index(frame["forecast_issue_date"].dropna().unique())
    observed_delta = float(frame["paired_absolute_error_delta"].mean())
    baseline_mae = float(frame["baseline_absolute_error"].mean())
    observed_pct = observed_delta / baseline_mae * 100 if baseline_mae else np.nan

    result: dict[str, object] = {
        "n": len(frame),
        "n_issue_dates": len(issue_dates),
        "mean_paired_mae_delta": observed_delta,
        "mae_improvement_pct": observed_pct,
        "bootstrap_method": "issue-date cluster bootstrap",
        "bootstrap_replicates": N_BOOTSTRAP,
        "min_issue_dates_for_confidence_claim": MIN_CONFIDENCE_ISSUE_DATES,
    }
    if len(issue_dates) < 2:
        result.update({
            "paired_delta_ci95_lower": np.nan,
            "paired_delta_ci95_upper": np.nan,
            "mae_improvement_pct_ci95_lower": np.nan,
            "mae_improvement_pct_ci95_upper": np.nan,
            "probability_improvement_positive": np.nan,
            "exploratory_ci_direction": "not_estimable",
            "confidence_evidence_ready": False,
            "confidence_status": "insufficient_issue_dates",
        })
        return result

    by_date = {date: frame.loc[frame["forecast_issue_date"].eq(date)] for date in issue_dates}
    deltas = np.empty(N_BOOTSTRAP, dtype=float)
    pcts = np.empty(N_BOOTSTRAP, dtype=float)
    for i in range(N_BOOTSTRAP):
        sampled = rng.choice(issue_dates.to_numpy(), size=len(issue_dates), replace=True)
        sample = pd.concat([by_date[date] for date in sampled], ignore_index=True)
        delta = float(sample["paired_absolute_error_delta"].mean())
        base = float(sample["baseline_absolute_error"].mean())
        deltas[i] = delta
        pcts[i] = delta / base * 100 if base else np.nan

    finite_pct = pcts[np.isfinite(pcts)]
    delta_lower = float(np.quantile(deltas, 0.025))
    delta_upper = float(np.quantile(deltas, 0.975))
    if delta_lower > 0:
        exploratory_direction = "supports_improvement"
    elif delta_upper < 0:
        exploratory_direction = "supports_harm"
    else:
        exploratory_direction = "uncertain"

    evidence_ready = len(issue_dates) >= MIN_CONFIDENCE_ISSUE_DATES
    result.update({
        "paired_delta_ci95_lower": delta_lower,
        "paired_delta_ci95_upper": delta_upper,
        "mae_improvement_pct_ci95_lower": float(np.quantile(finite_pct, 0.025)) if len(finite_pct) else np.nan,
        "mae_improvement_pct_ci95_upper": float(np.quantile(finite_pct, 0.975)) if len(finite_pct) else np.nan,
        "probability_improvement_positive": float(np.mean(deltas > 0)),
        "exploratory_ci_direction": exploratory_direction,
        "confidence_evidence_ready": evidence_ready,
        "confidence_status": exploratory_direction if evidence_ready else "insufficient_issue_dates",
    })
    return result


def _issue_date_summary(active: pd.DataFrame) -> pd.DataFrame:
    """Summarize active weather-route performance at the independent issue-date level."""
    grouped = active.groupby(
        ["target_name", "horizon_band", "forecast_issue_date"], as_index=False
    ).agg(
        n=("paired_absolute_error_delta", "size"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        weather_win_rate=("paired_absolute_error_delta", lambda values: float((values > 0).mean())),
    )
    grouped["mae_improvement_pct"] = np.where(
        grouped["baseline_mae"].ne(0),
        grouped["mean_paired_mae_delta"] / grouped["baseline_mae"] * 100,
        np.nan,
    )
    grouped["issue_date_direction"] = np.select(
        [
            grouped["mean_paired_mae_delta"].gt(0),
            grouped["mean_paired_mae_delta"].lt(0),
        ],
        ["weather_better", "weather_worse"],
        default="tie",
    )
    return grouped.sort_values(
        ["target_name", "horizon_band", "forecast_issue_date"], ignore_index=True
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    required = {
        "target_name", "horizon_band", "forecast_issue_date", "weather_route_active",
        "baseline_absolute_error", "weather_absolute_error", "paired_absolute_error_delta",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing columns: {sorted(missing)}")

    active = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if active.empty:
        print("No matured active weather-route rows; confidence summary skipped.")
        return

    issue_date_summary = _issue_date_summary(active)
    issue_date_summary.to_csv(
        args.output_dir / "weather-route-by-issue-date.csv", index=False
    )
    print(
        "Issue-date weather-route summary "
        f"({issue_date_summary['forecast_issue_date'].nunique()} distinct issue date(s)):"
    )
    print(issue_date_summary.to_string(index=False))

    rng = np.random.default_rng(SEED)
    rows = []
    for (target, band), frame in active.groupby(["target_name", "horizon_band"], sort=True):
        row = {"target_name": target, "horizon_band": band}
        row.update(_bootstrap_group(frame, rng))
        rows.append(row)

    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "weather-route-confidence.csv", index=False)
    print("Prospective weather-route clustered-bootstrap confidence summary:")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
