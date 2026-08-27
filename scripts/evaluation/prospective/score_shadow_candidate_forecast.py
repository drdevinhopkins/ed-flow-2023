#!/usr/bin/env python3
"""Score matured prospective candidate-metric shadow forecasts against observed ED flow.

The candidate archive remains non-production. Promotion-oriented summaries only use rows
where the pre-registered candidate route differs from history-only baseline. Evidence is
reported immediately but cannot become promotion-evaluable until it spans 56 days, 28
issue dates, 100 active routed rows, and 100 unique realized target hours.
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

from evaluation.backtests import backtest_candidate_metrics_cutoff as candidate_bt  # noqa: E402

MIN_PROSPECTIVE_SPAN_DAYS = 56
MIN_ISSUE_DATES = 28
MIN_ACTIVE_ROWS = 100
MIN_UNIQUE_TARGET_HOURS = 100
MAX_INTERVAL_COVERAGE_DROP = 0.05
MAX_ABS_BIAS_WORSENING_FRACTION_OF_BASELINE_MAE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/prospective-candidates/latest-score"),
    )
    return parser.parse_args()


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group = ["target_name", "horizon_band", "candidate_scenario"]
    summary = detail.groupby(group, as_index=False).agg(
        n=("actual", "size"),
        n_runs=("forecast_run_id", "nunique"),
        n_issue_dates=("forecast_issue_date", "nunique"),
        n_unique_target_hours=("target_ds", "nunique"),
        first_issued_at=("forecast_issued_at", "min"),
        last_issued_at=("forecast_issued_at", "max"),
        baseline_mae=("baseline_absolute_error", "mean"),
        candidate_mae=("candidate_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        candidate_win_rate=("candidate_wins", "mean"),
        baseline_bias=("baseline_error", "mean"),
        candidate_bias=("candidate_error", "mean"),
        baseline_mse=("baseline_squared_error", "mean"),
        candidate_mse=("candidate_squared_error", "mean"),
        baseline_interval_coverage=("baseline_interval_covered", "mean"),
        candidate_interval_coverage=("candidate_interval_covered", "mean"),
    )
    summary["baseline_rmse"] = np.sqrt(summary.pop("baseline_mse"))
    summary["candidate_rmse"] = np.sqrt(summary.pop("candidate_mse"))
    summary["mae_improvement_pct"] = (
        (summary["baseline_mae"] - summary["candidate_mae"])
        / summary["baseline_mae"].replace(0, np.nan)
        * 100
    )
    summary["prospective_span_days"] = (
        summary["last_issued_at"] - summary["first_issued_at"]
    ).dt.total_seconds() / 86400.0
    summary["interval_coverage_delta"] = (
        summary["candidate_interval_coverage"] - summary["baseline_interval_coverage"]
    )
    summary["interval_coverage_ok"] = summary["interval_coverage_delta"] >= -MAX_INTERVAL_COVERAGE_DROP
    summary["absolute_bias_worsening"] = (
        summary["candidate_bias"].abs() - summary["baseline_bias"].abs()
    )
    summary["bias_worsening_tolerance"] = (
        summary["baseline_mae"] * MAX_ABS_BIAS_WORSENING_FRACTION_OF_BASELINE_MAE
    )
    summary["bias_not_materially_worse"] = (
        summary["absolute_bias_worsening"] <= summary["bias_worsening_tolerance"]
    )
    summary["directional_criteria_met"] = (
        (summary["candidate_mae"] < summary["baseline_mae"])
        & (summary["mean_paired_mae_delta"] > 0)
        & (summary["median_paired_mae_delta"] > 0)
        & (summary["candidate_win_rate"] >= 0.55)
        & summary["interval_coverage_ok"]
        & summary["bias_not_materially_worse"]
    )
    summary["promotion_evidence_ready"] = (
        (summary["prospective_span_days"] >= MIN_PROSPECTIVE_SPAN_DAYS)
        & (summary["n_issue_dates"] >= MIN_ISSUE_DATES)
        & (summary["n"] >= MIN_ACTIVE_ROWS)
        & (summary["n_unique_target_hours"] >= MIN_UNIQUE_TARGET_HOURS)
    )
    summary["promotion_status"] = np.select(
        [~summary["promotion_evidence_ready"], summary["directional_criteria_met"]],
        ["collecting", "evaluable_pass"],
        default="evaluable_fail",
    )
    return summary.sort_values(group)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast = pd.read_csv(args.forecast_csv)
    required = {
        "forecast_run_id", "forecast_issued_at", "target_ds", "target_name",
        "horizon_hour", "horizon_band", "baseline_prediction",
        "candidate_prediction", "candidate_scenario",
    }
    missing = required - set(forecast.columns)
    if missing:
        raise ValueError(f"Candidate archive missing required columns: {sorted(missing)}")

    forecast["target_ds"] = pd.to_datetime(forecast["target_ds"], format="mixed", errors="coerce")
    if forecast["target_ds"].isna().any():
        raise ValueError("Candidate archive contains invalid target_ds values")
    forecast["forecast_issued_at"] = pd.to_datetime(
        forecast["forecast_issued_at"], utc=True, errors="coerce"
    )
    if forecast["forecast_issued_at"].isna().any():
        raise ValueError("Candidate archive contains invalid forecast_issued_at values")
    forecast["forecast_issue_date"] = forecast["forecast_issued_at"].dt.date
    forecast["candidate_route_active"] = forecast["candidate_scenario"].ne("baseline")

    flow = candidate_bt.load_flow()
    actual_long = flow[["ds", *candidate_bt.CANDIDATE_TARGETS]].melt(
        id_vars="ds", var_name="target_name", value_name="actual"
    )
    actual_long = actual_long.rename(columns={"ds": "target_ds"})
    actual_long["actual"] = pd.to_numeric(actual_long["actual"], errors="coerce")

    detail = forecast.merge(
        actual_long,
        on=["target_ds", "target_name"],
        how="left",
        validate="many_to_one",
    )
    detail = detail.loc[detail["actual"].notna()].copy()
    if detail.empty:
        print("No candidate forecast rows have matured yet; nothing to score.")
        return

    for prefix in ["baseline", "candidate"]:
        detail[f"{prefix}_error"] = detail[f"{prefix}_prediction"] - detail["actual"]
        detail[f"{prefix}_absolute_error"] = detail[f"{prefix}_error"].abs()
        detail[f"{prefix}_squared_error"] = detail[f"{prefix}_error"] ** 2
        valid = detail[[f"{prefix}_lower", f"{prefix}_upper"]].notna().all(axis=1)
        detail[f"{prefix}_interval_covered"] = np.where(
            valid,
            (detail["actual"] >= detail[f"{prefix}_lower"])
            & (detail["actual"] <= detail[f"{prefix}_upper"]),
            np.nan,
        )
    detail["paired_absolute_error_delta"] = (
        detail["baseline_absolute_error"] - detail["candidate_absolute_error"]
    )
    detail["candidate_wins"] = detail["paired_absolute_error_delta"] > 0

    detail.to_csv(args.output_dir / "detail.csv", index=False)
    _summarize(detail).to_csv(args.output_dir / "summary-all-pairs.csv", index=False)

    active = detail.loc[detail["candidate_route_active"]].copy()
    if active.empty:
        print(f"Scored {len(detail)} matured rows; no routed candidate rows have matured yet.")
        return
    summary = _summarize(active)
    summary.to_csv(args.output_dir / "summary-candidate-routes.csv", index=False)
    print(
        f"Scored {len(detail)} matured candidate rows from "
        f"{detail['forecast_run_id'].nunique()} run(s); {len(active)} used a candidate route"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
