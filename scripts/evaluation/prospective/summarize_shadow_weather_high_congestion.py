#!/usr/bin/env python3
"""Summarize prospective weather-route accuracy during high-congestion actuals.

This is a diagnostic only. Within each target × horizon band, high congestion is defined
from matured weather-active rows as actual values at or above that group's 75th percentile.
The output compares paired baseline/weather error specifically in that operational tail so
an aggregate MAE improvement cannot hide worse performance when the ED is busiest.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_name", "horizon_band", "actual", "weather_route_active",
        "baseline_absolute_error", "weather_absolute_error",
        "paired_absolute_error_delta", "weather_wins", "forecast_run_id",
        "forecast_issue_date",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    active = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if active.empty:
        return pd.DataFrame()

    group = ["target_name", "horizon_band"]
    thresholds = (
        active.groupby(group, as_index=False)["actual"]
        .quantile(0.75)
        .rename(columns={"actual": "high_congestion_threshold_q75"})
    )
    active = active.merge(thresholds, on=group, how="left", validate="many_to_one")
    tail = active.loc[active["actual"] >= active["high_congestion_threshold_q75"]].copy()

    summary = tail.groupby(group, as_index=False).agg(
        n_high_congestion=("actual", "size"),
        n_runs=("forecast_run_id", "nunique"),
        n_issue_dates=("forecast_issue_date", "nunique"),
        high_congestion_threshold_q75=("high_congestion_threshold_q75", "first"),
        mean_actual=("actual", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        weather_win_rate=("weather_wins", "mean"),
    )
    summary["mae_improvement_pct"] = (
        (summary["baseline_mae"] - summary["weather_mae"])
        / summary["baseline_mae"].replace(0, np.nan)
        * 100
    )
    summary["high_congestion_direction"] = np.where(
        summary["weather_mae"] < summary["baseline_mae"],
        "weather_better",
        np.where(summary["weather_mae"] > summary["baseline_mae"], "weather_worse", "tie"),
    )
    return summary.sort_values(group)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    result = summarize(detail)
    path = args.output_dir / "weather-route-high-congestion.csv"
    result.to_csv(path, index=False)
    if result.empty:
        print("No matured weather-active rows available for high-congestion diagnostics.")
    else:
        print("High-congestion prospective weather diagnostics (top quartile of actuals):")
        print(result.to_string(index=False))


if __name__ == "__main__":
    main()
