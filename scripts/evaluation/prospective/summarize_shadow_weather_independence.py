#!/usr/bin/env python3
"""Quantify how much repeated intraday shadow forecasting inflates raw evidence counts.

This diagnostic is intentionally non-production. It reports overlap between matured
weather-active forecast rows, forecast runs, issue dates, and unique realized target
hours. Promotion readiness continues to be controlled by the stricter scorer guardrails;
this file only makes dependence/overlap visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/prospective-weather/latest-score"),
    )
    return parser.parse_args()


def summarize_independence(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_run_id",
        "forecast_issued_at",
        "forecast_issue_date",
        "ds",
        "target_name",
        "horizon_band",
        "weather_route_active",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    frame = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["forecast_issued_at"] = pd.to_datetime(
        frame["forecast_issued_at"], utc=True, errors="coerce"
    )
    frame["target_ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    if frame[["forecast_issued_at", "target_ds"]].isna().any().any():
        raise ValueError("Invalid forecast_issued_at or ds in prospective detail")

    frame["forecast_issue_hour"] = frame["forecast_issued_at"].dt.floor("h")
    group_cols = ["target_name", "horizon_band"]

    summary = frame.groupby(group_cols, as_index=False).agg(
        n_rows=("ds", "size"),
        n_runs=("forecast_run_id", "nunique"),
        n_issue_dates=("forecast_issue_date", "nunique"),
        n_issue_hours=("forecast_issue_hour", "nunique"),
        n_unique_target_hours=("target_ds", "nunique"),
        first_issued_at=("forecast_issued_at", "min"),
        last_issued_at=("forecast_issued_at", "max"),
    )

    per_target_hour = (
        frame.groupby([*group_cols, "target_ds"], as_index=False)
        .size()
        .rename(columns={"size": "forecasts_per_target_hour"})
    )
    target_hour_stats = per_target_hour.groupby(group_cols, as_index=False).agg(
        mean_forecasts_per_target_hour=("forecasts_per_target_hour", "mean"),
        median_forecasts_per_target_hour=("forecasts_per_target_hour", "median"),
        max_forecasts_per_target_hour=("forecasts_per_target_hour", "max"),
    )

    per_issue_date = (
        frame[[*group_cols, "forecast_issue_date", "forecast_run_id"]]
        .drop_duplicates()
        .groupby([*group_cols, "forecast_issue_date"], as_index=False)
        .size()
        .rename(columns={"size": "runs_per_issue_date"})
    )
    issue_date_stats = per_issue_date.groupby(group_cols, as_index=False).agg(
        mean_runs_per_issue_date=("runs_per_issue_date", "mean"),
        max_runs_per_issue_date=("runs_per_issue_date", "max"),
    )

    summary = summary.merge(target_hour_stats, on=group_cols, how="left")
    summary = summary.merge(issue_date_stats, on=group_cols, how="left")
    summary["prospective_span_days"] = (
        summary["last_issued_at"] - summary["first_issued_at"]
    ).dt.total_seconds() / 86400.0
    summary["row_inflation_vs_unique_target_hours"] = (
        summary["n_rows"] / summary["n_unique_target_hours"].replace(0, np.nan)
    )
    summary["run_inflation_vs_issue_dates"] = (
        summary["n_runs"] / summary["n_issue_dates"].replace(0, np.nan)
    )
    summary["independence_note"] = np.where(
        summary["row_inflation_vs_unique_target_hours"] > 2,
        "substantial_overlap",
        "limited_overlap",
    )
    return summary.sort_values(group_cols)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    summary = summarize_independence(detail)
    output = args.output_dir / "weather-route-independence.csv"
    summary.to_csv(output, index=False)
    if summary.empty:
        print("No matured weather-active rows yet; independence diagnostic is empty.")
    else:
        print("Prospective weather independence/overlap diagnostic:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
