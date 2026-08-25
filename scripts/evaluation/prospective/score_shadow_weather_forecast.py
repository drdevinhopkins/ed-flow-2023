#!/usr/bin/env python3
"""Score matured paired shadow-weather forecasts against observed ED flow values.

All matured paired rows are retained in ``detail.csv`` for audit. Promotion-oriented
summary metrics are calculated only for rows where the weather-enabled router actually
selected a weather scenario; unchanged non-weather rows would otherwise count as ties and
artificially depress weather win rate.

Directional accuracy criteria are reported immediately, but are deliberately separated
from evidence readiness. A weather route is not considered promotion-evaluable until its
prospective observations span at least 56 days and at least 28 distinct issue dates.
"""

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

MIN_PROSPECTIVE_SPAN_DAYS = 56
MIN_ISSUE_DATES = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/prospective-weather/latest-score"),
    )
    return parser.parse_args()


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group = ["target_name", "horizon_band"]
    summary = detail.groupby(group, as_index=False).agg(
        n=("actual", "size"),
        n_runs=("forecast_run_id", "nunique"),
        n_issue_dates=("forecast_issue_date", "nunique"),
        first_issued_at=("forecast_issued_at", "min"),
        last_issued_at=("forecast_issued_at", "max"),
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
    summary["prospective_span_days"] = (
        summary["last_issued_at"] - summary["first_issued_at"]
    ).dt.total_seconds() / 86400.0
    summary["directional_criteria_met"] = (
        (summary["weather_mae"] < summary["baseline_mae"])
        & (summary["mean_paired_mae_delta"] > 0)
        & (summary["median_paired_mae_delta"] > 0)
        & (summary["weather_win_rate"] >= 0.55)
    )
    summary["promotion_evidence_ready"] = (
        (summary["prospective_span_days"] >= MIN_PROSPECTIVE_SPAN_DAYS)
        & (summary["n_issue_dates"] >= MIN_ISSUE_DATES)
    )
    summary["promotion_status"] = np.select(
        [
            ~summary["promotion_evidence_ready"],
            summary["directional_criteria_met"],
        ],
        ["collecting", "evaluable_pass"],
        default="evaluable_fail",
    )
    return summary.sort_values(group)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forecast = pd.read_csv(args.forecast_csv)
    forecast["ds"] = base.parse_ds(forecast["ds"])
    required = {
        "forecast_run_id",
        "forecast_issued_at",
        "ds",
        "target_name",
        "horizon_hour",
        "horizon_band",
        "baseline_prediction",
        "weather_prediction",
        "weather_scenario",
        "weather_snapshot_is_prospective",
    }
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"Forecast archive missing required columns: {sorted(missing)}")
    if not forecast["weather_snapshot_is_prospective"].astype(bool).all():
        raise ValueError("Refusing to score weather rows not marked as prospective snapshots")

    forecast["forecast_issued_at"] = pd.to_datetime(
        forecast["forecast_issued_at"], utc=True, errors="coerce"
    )
    if forecast["forecast_issued_at"].isna().any():
        raise ValueError("Forecast archive contains invalid forecast_issued_at values")
    forecast["forecast_issue_date"] = forecast["forecast_issued_at"].dt.date
    forecast["weather_route_active"] = (
        forecast["weather_scenario"].astype(str).str.startswith("weather")
    )

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

    detail_path = args.output_dir / "detail.csv"
    all_summary_path = args.output_dir / "summary-all-pairs.csv"
    active_summary_path = args.output_dir / "summary-weather-routes.csv"
    detail.to_csv(detail_path, index=False)

    all_summary = _summarize(detail)
    all_summary.to_csv(all_summary_path, index=False)

    active = detail.loc[detail["weather_route_active"]].copy()
    if active.empty:
        print(
            f"Scored {len(detail)} matured paired rows, but no matured rows used an active weather route yet."
        )
        return

    active_summary = _summarize(active)
    active_summary.to_csv(active_summary_path, index=False)

    print(
        f"Scored {len(detail)} matured paired rows from "
        f"{detail['forecast_run_id'].nunique()} run(s); "
        f"{len(active)} rows actually used weather routing"
    )
    print("Promotion-oriented weather-route summary (status remains collecting until evidence guardrails are met):")
    print(active_summary.to_string(index=False))


if __name__ == "__main__":
    main()
