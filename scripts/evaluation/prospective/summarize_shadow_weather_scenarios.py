#!/usr/bin/env python3
"""Summarize prospective weather evidence separately for each exact weather scenario.

Target/horizon summaries can otherwise pool multiple weather feature routes. This diagnostic
keeps route evidence separate so a strong scenario cannot mask a weak one, and reports the
same minimum evidence dimensions and directional guardrails used by prospective promotion.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PROSPECTIVE_SPAN_DAYS = 56
MIN_ISSUE_DATES = 28
MIN_ACTIVE_ROWS = 100
MAX_INTERVAL_COVERAGE_DROP = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/prospective-weather/latest-score"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail = pd.read_csv(args.detail_csv)
    required = {
        "target_name",
        "horizon_band",
        "weather_scenario",
        "weather_route_active",
        "forecast_run_id",
        "forecast_issued_at",
        "forecast_issue_date",
        "baseline_absolute_error",
        "weather_absolute_error",
        "paired_absolute_error_delta",
        "weather_wins",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    active = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if active.empty:
        print("No matured weather-active rows; scenario summary skipped.")
        return

    active["forecast_issued_at"] = pd.to_datetime(
        active["forecast_issued_at"], utc=True, errors="coerce"
    )
    if active["forecast_issued_at"].isna().any():
        raise ValueError("Invalid forecast_issued_at in matured weather-active rows")

    group = ["target_name", "horizon_band", "weather_scenario"]
    aggregations: dict[str, tuple[str, str]] = {
        "n": ("paired_absolute_error_delta", "size"),
        "n_runs": ("forecast_run_id", "nunique"),
        "n_issue_dates": ("forecast_issue_date", "nunique"),
        "first_issued_at": ("forecast_issued_at", "min"),
        "last_issued_at": ("forecast_issued_at", "max"),
        "baseline_mae": ("baseline_absolute_error", "mean"),
        "weather_mae": ("weather_absolute_error", "mean"),
        "mean_paired_mae_delta": ("paired_absolute_error_delta", "mean"),
        "median_paired_mae_delta": ("paired_absolute_error_delta", "median"),
        "weather_win_rate": ("weather_wins", "mean"),
    }
    if {"baseline_interval_covered", "weather_interval_covered"}.issubset(active.columns):
        aggregations.update(
            {
                "baseline_interval_coverage": ("baseline_interval_covered", "mean"),
                "weather_interval_coverage": ("weather_interval_covered", "mean"),
            }
        )

    summary = active.groupby(group, as_index=False).agg(**aggregations)
    summary["mae_improvement_pct"] = (
        (summary["baseline_mae"] - summary["weather_mae"])
        / summary["baseline_mae"].replace(0, np.nan)
        * 100
    )
    summary["prospective_span_days"] = (
        summary["last_issued_at"] - summary["first_issued_at"]
    ).dt.total_seconds() / 86400.0

    if {"baseline_interval_coverage", "weather_interval_coverage"}.issubset(summary.columns):
        summary["interval_coverage_delta"] = (
            summary["weather_interval_coverage"] - summary["baseline_interval_coverage"]
        )
        summary["interval_coverage_ok"] = (
            summary["interval_coverage_delta"] >= -MAX_INTERVAL_COVERAGE_DROP
        )
    else:
        summary["baseline_interval_coverage"] = np.nan
        summary["weather_interval_coverage"] = np.nan
        summary["interval_coverage_delta"] = np.nan
        summary["interval_coverage_ok"] = True

    summary["rows_needed"] = (MIN_ACTIVE_ROWS - summary["n"]).clip(lower=0)
    summary["issue_dates_needed"] = (MIN_ISSUE_DATES - summary["n_issue_dates"]).clip(lower=0)
    summary["span_days_needed"] = (
        MIN_PROSPECTIVE_SPAN_DAYS - summary["prospective_span_days"]
    ).clip(lower=0)
    summary["scenario_evidence_ready"] = (
        (summary["n"] >= MIN_ACTIVE_ROWS)
        & (summary["n_issue_dates"] >= MIN_ISSUE_DATES)
        & (summary["prospective_span_days"] >= MIN_PROSPECTIVE_SPAN_DAYS)
    )
    summary["scenario_directional_criteria_met"] = (
        (summary["weather_mae"] < summary["baseline_mae"])
        & (summary["mean_paired_mae_delta"] > 0)
        & (summary["median_paired_mae_delta"] > 0)
        & (summary["weather_win_rate"] >= 0.55)
        & summary["interval_coverage_ok"]
    )
    summary["scenario_status"] = np.select(
        [
            ~summary["scenario_evidence_ready"],
            summary["scenario_directional_criteria_met"],
        ],
        ["collecting", "evaluable_pass"],
        default="evaluable_fail",
    )
    summary["scenario_direction"] = np.select(
        [summary["mae_improvement_pct"] > 0, summary["mae_improvement_pct"] < 0],
        ["weather_better", "weather_worse"],
        default="tie",
    )

    out = args.output_dir / "weather-route-by-scenario.csv"
    summary.sort_values(group).to_csv(out, index=False)
    print("Scenario-specific prospective weather evidence:")
    print(summary.sort_values(group).to_string(index=False))


if __name__ == "__main__":
    main()
