#!/usr/bin/env python3
"""Summarize matured active weather-route scores by weekday and target hour.

This diagnostic is deliberately non-production. It reads the scorer's audited detail.csv
and emits slice tables only for rows where weather routing actually changed the route.
Small samples remain descriptive; no promotion decision is made here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def summarize(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    out = frame.groupby(group, as_index=False).agg(
        n=("actual", "size"),
        n_runs=("forecast_run_id", "nunique"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        weather_win_rate=("weather_wins", "mean"),
        baseline_bias=("baseline_error", "mean"),
        weather_bias=("weather_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
    )
    out["mae_improvement_pct"] = (
        (out["baseline_mae"] - out["weather_mae"])
        / out["baseline_mae"].replace(0, pd.NA)
        * 100
    )
    return out.sort_values(group)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    required = {
        "forecast_run_id", "ds", "target_name", "horizon_band", "weather_route_active",
        "actual", "baseline_absolute_error", "weather_absolute_error", "weather_wins",
        "baseline_error", "weather_error", "paired_absolute_error_delta",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"detail.csv missing required columns: {sorted(missing)}")

    active = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if active.empty:
        print("No matured active weather-route rows yet; no diagnostic slices emitted.")
        return

    active["target_ds"] = pd.to_datetime(active["ds"], errors="coerce")
    if active["target_ds"].isna().any():
        raise ValueError("Invalid ds values in scorer detail.csv")
    active["target_weekday"] = active["target_ds"].dt.day_name()
    active["target_hour"] = active["target_ds"].dt.hour

    summarize(
        active,
        ["target_name", "horizon_band", "target_weekday"],
    ).to_csv(args.output_dir / "summary-weather-routes-by-weekday.csv", index=False)
    summarize(
        active,
        ["target_name", "horizon_band", "target_hour"],
    ).to_csv(args.output_dir / "summary-weather-routes-by-hour.csv", index=False)

    print(
        f"Wrote weather-route diagnostics for {len(active)} matured active rows across "
        f"{active['forecast_run_id'].nunique()} run(s)."
    )


if __name__ == "__main__":
    main()
