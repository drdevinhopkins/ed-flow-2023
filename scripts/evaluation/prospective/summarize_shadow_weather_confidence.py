#!/usr/bin/env python3
"""Estimate uncertainty for prospective weather-route MAE improvement.

Rows from the same forecast issue are correlated across horizons, so confidence intervals
are produced by resampling issue dates rather than individual forecast rows. This output
is diagnostic only and does not alter promotion guardrails or routing.

Bootstrap intervals are computed as soon as at least two *complete* issue dates exist,
but they are explicitly exploratory until the prospective validation plan's 28-issue-date
evidence threshold is reached. An issue date is complete only after the full local calendar
day plus the horizon band's maximum lead has had time to mature. This prevents a partially
matured current/recent day from receiving the same weight as a completed issue date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOTSTRAP = 5000
SEED = 20260826
MIN_CONFIDENCE_ISSUE_DATES = 28
LOCAL_TZ = "America/Toronto"
BAND_MAX_HORIZON = {
    "h01_04": 4,
    "h05_08": 8,
    "h09_12": 12,
    "h13_24": 24,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="Optional timezone-aware timestamp used for deterministic completeness tests.",
    )
    return parser.parse_args()


def _as_of_local(value: str | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=LOCAL_TZ)
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError("--as-of must include a timezone offset")
    return stamp.tz_convert(LOCAL_TZ)


def _issue_date_completion_time(issue_date: object, horizon_band: str) -> pd.Timestamp:
    if horizon_band not in BAND_MAX_HORIZON:
        raise ValueError(f"Unknown horizon band for completeness: {horizon_band}")
    day = pd.Timestamp(issue_date)
    if day.tzinfo is None:
        day = day.tz_localize(LOCAL_TZ)
    else:
        day = day.tz_convert(LOCAL_TZ)
    next_midnight = day.normalize() + pd.Timedelta(days=1)
    return next_midnight + pd.Timedelta(hours=BAND_MAX_HORIZON[horizon_band])


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
        "bootstrap_method": "complete issue-date cluster bootstrap",
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
            "confidence_status": "insufficient_complete_issue_dates",
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
        "confidence_status": exploratory_direction if evidence_ready else "insufficient_complete_issue_dates",
    })
    return result


def _issue_date_summary(active: pd.DataFrame, *, as_of_local: pd.Timestamp) -> pd.DataFrame:
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
    grouped["issue_date_complete_at"] = [
        _issue_date_completion_time(date, band).isoformat()
        for date, band in zip(grouped["forecast_issue_date"], grouped["horizon_band"])
    ]
    grouped["issue_date_complete"] = [
        as_of_local >= _issue_date_completion_time(date, band)
        for date, band in zip(grouped["forecast_issue_date"], grouped["horizon_band"])
    ]
    return grouped.sort_values(
        ["target_name", "horizon_band", "forecast_issue_date"], ignore_index=True
    )


def _q10(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.quantile(0.10)) if len(finite) else np.nan


def _issue_date_balanced_summary(by_date: pd.DataFrame) -> pd.DataFrame:
    """Give each complete issue date equal weight and expose downside/tail behavior."""
    complete = by_date.loc[by_date["issue_date_complete"].astype(bool)].copy()
    if complete.empty:
        return pd.DataFrame()
    grouped = complete.groupby(["target_name", "horizon_band"], as_index=False).agg(
        n_issue_dates=("forecast_issue_date", "nunique"),
        issue_date_mean_paired_mae_delta=("mean_paired_mae_delta", "mean"),
        issue_date_median_paired_mae_delta=("mean_paired_mae_delta", "median"),
        issue_date_mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
        issue_date_median_mae_improvement_pct=("mae_improvement_pct", "median"),
        issue_date_win_rate=("mean_paired_mae_delta", lambda values: float((values > 0).mean())),
        harmful_issue_date_rate=("mean_paired_mae_delta", lambda values: float((values < 0).mean())),
        worst_issue_date_mae_improvement_pct=("mae_improvement_pct", "min"),
        p10_issue_date_mae_improvement_pct=("mae_improvement_pct", _q10),
    )
    grouped["issue_date_direction"] = np.select(
        [
            grouped["issue_date_mean_paired_mae_delta"].gt(0),
            grouped["issue_date_mean_paired_mae_delta"].lt(0),
        ],
        ["weather_better", "weather_worse"],
        default="tie",
    )
    return grouped.sort_values(["target_name", "horizon_band"], ignore_index=True)


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

    as_of_local = _as_of_local(args.as_of)
    issue_date_summary = _issue_date_summary(active, as_of_local=as_of_local)
    issue_date_summary.to_csv(
        args.output_dir / "weather-route-by-issue-date.csv", index=False
    )
    print(
        "Issue-date weather-route summary "
        f"({issue_date_summary['forecast_issue_date'].nunique()} distinct issue date(s); "
        "formal date-level evidence uses complete dates only):"
    )
    print(issue_date_summary.to_string(index=False))

    balanced = _issue_date_balanced_summary(issue_date_summary)
    balanced.to_csv(
        args.output_dir / "weather-route-issue-date-balanced.csv", index=False
    )
    print("Issue-date-balanced weather-route summary (complete dates only):")
    print(balanced.to_string(index=False) if not balanced.empty else "No complete issue dates yet.")

    complete_dates = issue_date_summary.loc[
        issue_date_summary["issue_date_complete"].astype(bool),
        ["target_name", "horizon_band", "forecast_issue_date"],
    ]
    active_complete = active.merge(
        complete_dates,
        on=["target_name", "horizon_band", "forecast_issue_date"],
        how="inner",
        validate="many_to_one",
    )

    rng = np.random.default_rng(SEED)
    rows = []
    for (target, band), all_frame in active.groupby(["target_name", "horizon_band"], sort=True):
        frame = active_complete.loc[
            active_complete["target_name"].eq(target)
            & active_complete["horizon_band"].eq(band)
        ]
        row = {"target_name": target, "horizon_band": band}
        row["n_all_matured_active_rows"] = len(all_frame)
        row["n_all_issue_dates"] = int(all_frame["forecast_issue_date"].nunique())
        if frame.empty:
            row.update({
                "n": 0,
                "n_issue_dates": 0,
                "mean_paired_mae_delta": np.nan,
                "mae_improvement_pct": np.nan,
                "bootstrap_method": "complete issue-date cluster bootstrap",
                "bootstrap_replicates": N_BOOTSTRAP,
                "min_issue_dates_for_confidence_claim": MIN_CONFIDENCE_ISSUE_DATES,
                "paired_delta_ci95_lower": np.nan,
                "paired_delta_ci95_upper": np.nan,
                "mae_improvement_pct_ci95_lower": np.nan,
                "mae_improvement_pct_ci95_upper": np.nan,
                "probability_improvement_positive": np.nan,
                "exploratory_ci_direction": "not_estimable",
                "confidence_evidence_ready": False,
                "confidence_status": "insufficient_complete_issue_dates",
            })
        else:
            row.update(_bootstrap_group(frame, rng))
        rows.append(row)

    output = pd.DataFrame(rows)
    output.to_csv(args.output_dir / "weather-route-confidence.csv", index=False)
    print("Prospective weather-route clustered-bootstrap confidence summary (complete dates only):")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
