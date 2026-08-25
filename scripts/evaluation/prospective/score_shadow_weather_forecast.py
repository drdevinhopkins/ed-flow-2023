#!/usr/bin/env python3
"""Score matured paired shadow-weather forecasts against observed ED flow values."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import backtest_covariate_ablation as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/prospective-weather/latest-score"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(args.forecast_csv)
    forecast["ds"] = base.parse_ds(forecast["ds"])
    required = {
        "forecast_run_id",
        "ds",
        "target_name",
        "horizon_hour",
        "horizon_band",
        "baseline_prediction",
        "weather_prediction",
        "weather_snapshot_is_prospective",
    }
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"Forecast archive missing required columns: {sorted(missing)}")
    if not forecast["weather_snapshot_is_prospective"].astype(bool).all():
        raise ValueError("Refusing to score weather rows not marked as prospective snapshots")

    flow = base.load_flow()
    actual_long = flow[["ds", *base.FLOW_TARGETS]].melt(
        id_vars="ds", var_name="target_name", value_name="actual"
    )
    actual_long["actual"] = pd.to_numeric(actual_long["actual"], errors="coerce")

    detail = forecast.merge(
        actual_long,
        on=["ds", "target_name"],
        how="left",
        validate="many_to_one",
    )
    detail = detail.loc[detail["actual"].notna()].copy()
    if detail.empty:
        print("No forecast rows have matured yet; nothing to score.")
        return

    detail["baseline_error"] = detail["baseline_prediction"] - detail["actual"]
    detail["weather_error"] = detail["weather_prediction"] - detail["actual"]
    detail["baseline_absolute_error"] = detail["baseline_error"].abs()
    detail["weather_absolute_error"] = detail["weather_error"].abs()
    detail["paired_absolute_error_delta"] = (
        detail["baseline_absolute_error"] - detail["weather_absolute_error"]
    )
    detail["weather_wins"] = detail["paired_absolute_error_delta"] > 0
    detail["baseline_squared_error"] = detail["baseline_error"] ** 2
    detail["weather_squared_error"] = detail["weather_error"] ** 2

    group = ["target_name", "horizon_band"]
    summary = detail.groupby(group, as_index=False).agg(
        n=("actual", "size"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        weather_win_rate=("weather_wins", "mean"),
        baseline_bias=("baseline_error", "mean"),
        weather_bias=("weather_error", "mean"),
        baseline_mse=("baseline_squared_error", "mean"),
        weather_mse=("weather_squared_error", "mean"),
    )
    summary["baseline_rmse"] = np.sqrt(summary.pop("baseline_mse"))
    summary["weather_rmse"] = np.sqrt(summary.pop("weather_mse"))
    summary["mae_improvement_pct"] = (
        (summary["baseline_mae"] - summary["weather_mae"])
        / summary["baseline_mae"].replace(0, np.nan)
        * 100
    )
    summary["prospective_pass_directional"] = (
        (summary["weather_mae"] < summary["baseline_mae"])
        & (summary["mean_paired_mae_delta"] > 0)
        & (summary["median_paired_mae_delta"] > 0)
        & (summary["weather_win_rate"] >= 0.55)
    )

    detail_path = args.output_dir / "detail.csv"
    summary_path = args.output_dir / "summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.sort_values(group).to_csv(summary_path, index=False)

    print(f"Scored {len(detail)} matured rows from {detail['forecast_run_id'].nunique()} run(s)")
    print(summary.sort_values(group).to_string(index=False))


if __name__ == "__main__":
    main()
