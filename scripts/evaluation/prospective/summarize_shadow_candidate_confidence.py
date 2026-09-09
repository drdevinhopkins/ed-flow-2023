#!/usr/bin/env python3
"""Estimate uncertainty for prospective candidate-route MAE improvement.

Forecast rows issued on the same local date are correlated, so bootstrap resampling is
performed at the complete issue-date level rather than over individual rows. Intervals
are exploratory until at least 28 complete issue dates exist. This diagnostic is
shadow-only and does not alter routing or promotion decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TZ = "America/Toronto"
BAND_MAX_HORIZON = {"h01_04": 4, "h05_08": 8, "h09_12": 12, "h13_24": 24}
N_BOOTSTRAP = 5000
SEED = 20260828
MIN_CONFIDENCE_ISSUE_DATES = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", type=str, default=None)
    return parser.parse_args()


def _as_of_local(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=LOCAL_TZ)
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return stamp.tz_convert(LOCAL_TZ)


def _completion_time(issue_date: object, horizon_band: str) -> pd.Timestamp:
    if horizon_band not in BAND_MAX_HORIZON:
        raise ValueError(f"Unknown horizon band: {horizon_band}")
    day = pd.Timestamp(issue_date)
    if day.tzinfo is None:
        day = day.tz_localize(LOCAL_TZ)
    else:
        day = day.tz_convert(LOCAL_TZ)
    return day.normalize() + pd.Timedelta(days=1, hours=BAND_MAX_HORIZON[horizon_band])


def _bootstrap(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    dates = pd.Index(frame["forecast_issue_date"].dropna().unique())
    observed_delta = float(frame["paired_absolute_error_delta"].mean())
    baseline_mae = float(frame["baseline_absolute_error"].mean())
    result: dict[str, object] = {
        "n": len(frame),
        "n_complete_issue_dates": len(dates),
        "mean_paired_mae_delta": observed_delta,
        "mae_improvement_pct": observed_delta / baseline_mae * 100 if baseline_mae else np.nan,
        "bootstrap_method": "complete issue-date cluster bootstrap",
        "bootstrap_replicates": N_BOOTSTRAP,
        "min_issue_dates_for_confidence_claim": MIN_CONFIDENCE_ISSUE_DATES,
    }
    if len(dates) < 2:
        result.update({
            "paired_delta_ci95_lower": np.nan,
            "paired_delta_ci95_upper": np.nan,
            "mae_improvement_pct_ci95_lower": np.nan,
            "mae_improvement_pct_ci95_upper": np.nan,
            "probability_improvement_positive": np.nan,
            "exploratory_ci_direction": "not_estimable",
            "confidence_evidence_ready": False,
            "confidence_status": "insufficient_complete_issue_dates",
        })
        return result

    by_date = {date: frame.loc[frame["forecast_issue_date"].eq(date)] for date in dates}
    deltas = np.empty(N_BOOTSTRAP)
    pcts = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        sampled = rng.choice(dates.to_numpy(), size=len(dates), replace=True)
        sample = pd.concat([by_date[date] for date in sampled], ignore_index=True)
        delta = float(sample["paired_absolute_error_delta"].mean())
        base = float(sample["baseline_absolute_error"].mean())
        deltas[i] = delta
        pcts[i] = delta / base * 100 if base else np.nan

    lower, upper = np.quantile(deltas, [0.025, 0.975])
    direction = "supports_improvement" if lower > 0 else "supports_harm" if upper < 0 else "uncertain"
    ready = len(dates) >= MIN_CONFIDENCE_ISSUE_DATES
    finite_pct = pcts[np.isfinite(pcts)]
    result.update({
        "paired_delta_ci95_lower": float(lower),
        "paired_delta_ci95_upper": float(upper),
        "mae_improvement_pct_ci95_lower": float(np.quantile(finite_pct, 0.025)) if len(finite_pct) else np.nan,
        "mae_improvement_pct_ci95_upper": float(np.quantile(finite_pct, 0.975)) if len(finite_pct) else np.nan,
        "probability_improvement_positive": float(np.mean(deltas > 0)),
        "exploratory_ci_direction": direction,
        "confidence_evidence_ready": ready,
        "confidence_status": direction if ready else "insufficient_complete_issue_dates",
    })
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    required = {
        "target_name", "horizon_band", "candidate_scenario", "forecast_issue_date",
        "candidate_route_active", "baseline_absolute_error", "paired_absolute_error_delta",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Candidate detail missing columns: {sorted(missing)}")

    active = detail.loc[detail["candidate_route_active"].astype(bool)].copy()
    if active.empty:
        print("No matured active candidate-route rows; confidence summary skipped.")
        return

    as_of = _as_of_local(args.as_of)
    active["issue_date_complete"] = [
        as_of >= _completion_time(date, band)
        for date, band in zip(active["forecast_issue_date"], active["horizon_band"])
    ]
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    group_cols = ["target_name", "horizon_band", "candidate_scenario"]
    for key, all_frame in active.groupby(group_cols, sort=True):
        complete = all_frame.loc[all_frame["issue_date_complete"]].copy()
        row = dict(zip(group_cols, key))
        row["n_all_matured_active_rows"] = len(all_frame)
        row["n_all_issue_dates"] = int(all_frame["forecast_issue_date"].nunique())
        if complete.empty:
            row.update({
                "n": 0, "n_complete_issue_dates": 0,
                "mean_paired_mae_delta": np.nan, "mae_improvement_pct": np.nan,
                "bootstrap_method": "complete issue-date cluster bootstrap",
                "bootstrap_replicates": N_BOOTSTRAP,
                "min_issue_dates_for_confidence_claim": MIN_CONFIDENCE_ISSUE_DATES,
                "paired_delta_ci95_lower": np.nan, "paired_delta_ci95_upper": np.nan,
                "mae_improvement_pct_ci95_lower": np.nan, "mae_improvement_pct_ci95_upper": np.nan,
                "probability_improvement_positive": np.nan,
                "exploratory_ci_direction": "not_estimable",
                "confidence_evidence_ready": False,
                "confidence_status": "insufficient_complete_issue_dates",
            })
        else:
            row.update(_bootstrap(complete, rng))
        rows.append(row)

    output = pd.DataFrame(rows).sort_values(group_cols, ignore_index=True)
    output.to_csv(args.output_dir / "candidate-route-confidence.csv", index=False)
    print("Prospective candidate-route clustered-bootstrap confidence summary (complete dates only):")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
