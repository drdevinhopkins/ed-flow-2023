#!/usr/bin/env python3
"""Sensitivity analysis using one forecast per realized target hour.

Repeated intraday shadow runs can target the same realized ED hour many times. This
non-production diagnostic removes that duplication in two complementary ways: it keeps
the earliest available weather-active forecast and the latest available weather-active
forecast for each target × horizon band × realized target hour, then compares paired
baseline-vs-weather performance. Agreement between those views is more reassuring than
an aggregate result driven by many overlapping forecasts.
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


def _summarize(frame: pd.DataFrame, selection: str) -> pd.DataFrame:
    group_cols = ["target_name", "horizon_band"]
    out = frame.groupby(group_cols, as_index=False).agg(
        n_unique_target_hours=("target_ds", "nunique"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        mean_paired_improvement=("paired_absolute_error_delta", "mean"),
        median_paired_improvement=("paired_absolute_error_delta", "median"),
        weather_win_rate=("candidate_wins", "mean"),
        baseline_bias=("baseline_signed_error", "mean"),
        weather_bias=("weather_signed_error", "mean"),
    )
    out["mae_improvement_pct"] = (
        out["baseline_mae"] - out["weather_mae"]
    ) / out["baseline_mae"].replace(0, np.nan) * 100.0
    out.insert(2, "selection", selection)
    return out


def summarize_unique_hour_sensitivity(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_issued_at",
        "ds",
        "target_name",
        "horizon_band",
        "weather_route_active",
        "baseline_absolute_error",
        "weather_absolute_error",
        "paired_absolute_error_delta",
        "candidate_wins",
        "baseline_signed_error",
        "weather_signed_error",
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

    key = ["target_name", "horizon_band", "target_ds"]
    ordered = frame.sort_values([*key, "forecast_issued_at"])
    earliest = ordered.drop_duplicates(key, keep="first")
    latest = ordered.drop_duplicates(key, keep="last")

    result = pd.concat(
        [
            _summarize(earliest, "earliest_forecast_per_realized_hour"),
            _summarize(latest, "latest_forecast_per_realized_hour"),
        ],
        ignore_index=True,
    )

    pivot = result.pivot_table(
        index=["target_name", "horizon_band"],
        columns="selection",
        values="mae_improvement_pct",
        aggfunc="first",
    ).reset_index()
    early_col = "earliest_forecast_per_realized_hour"
    late_col = "latest_forecast_per_realized_hour"
    if early_col in pivot.columns and late_col in pivot.columns:
        pivot["selection_direction_agrees"] = (
            np.sign(pivot[early_col].fillna(0)) == np.sign(pivot[late_col].fillna(0))
        )
        pivot = pivot[["target_name", "horizon_band", "selection_direction_agrees"]]
        result = result.merge(pivot, on=["target_name", "horizon_band"], how="left")
    else:
        result["selection_direction_agrees"] = False

    return result.sort_values(["target_name", "horizon_band", "selection"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    summary = summarize_unique_hour_sensitivity(detail)
    output = args.output_dir / "weather-route-unique-hour-sensitivity.csv"
    summary.to_csv(output, index=False)
    if summary.empty:
        print("No matured weather-active rows yet; unique-hour sensitivity is empty.")
    else:
        print("Prospective weather unique-realized-hour sensitivity:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
