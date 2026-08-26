#!/usr/bin/env python3
"""Summarize prospective weather-route accuracy during high-congestion actuals.

This is diagnostic only. High congestion is defined within each target × horizon band from
*unique realized target timestamps* so repeated shadow forecasts of the same ED hour cannot
move the congestion threshold. The script reports pooled tail performance, an equal-weight
unique-realized-hour view, and an issue-date-balanced view so repeated runs cannot dominate
the conclusion.
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


def _prepare_tail(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ds", "target_name", "horizon_band", "actual", "weather_route_active",
        "baseline_absolute_error", "weather_absolute_error",
        "paired_absolute_error_delta", "weather_wins", "forecast_run_id",
        "forecast_issue_date",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    active = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if active.empty:
        return active

    active["ds"] = pd.to_datetime(active["ds"], errors="coerce")
    active["actual"] = pd.to_numeric(active["actual"], errors="coerce")
    active = active.dropna(subset=["ds", "actual"])
    if active.empty:
        return active

    group = ["target_name", "horizon_band"]

    # Thresholds are based on unique realized target timestamps, not forecast rows.
    # The same actual ED hour can appear in several shadow runs and should count once.
    realized = active[[*group, "ds", "actual"]].drop_duplicates([*group, "ds"])
    thresholds = (
        realized.groupby(group, as_index=False)["actual"]
        .quantile(0.75)
        .rename(columns={"actual": "high_congestion_threshold_q75"})
    )
    active = active.merge(thresholds, on=group, how="left", validate="many_to_one")
    return active.loc[
        active["actual"] >= active["high_congestion_threshold_q75"]
    ].copy()


def _add_direction(frame: pd.DataFrame, *, output_column: str) -> pd.DataFrame:
    frame["mae_improvement_pct"] = (
        (frame["baseline_mae"] - frame["weather_mae"])
        / frame["baseline_mae"].replace(0, np.nan)
        * 100
    )
    frame[output_column] = np.where(
        frame["weather_mae"] < frame["baseline_mae"],
        "weather_better",
        np.where(frame["weather_mae"] > frame["baseline_mae"], "weather_worse", "tie"),
    )
    return frame


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    tail = _prepare_tail(detail)
    if tail.empty:
        return pd.DataFrame()

    group = ["target_name", "horizon_band"]
    summary = tail.groupby(group, as_index=False).agg(
        n_high_congestion=("actual", "size"),
        n_unique_target_hours=("ds", "nunique"),
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
    return _add_direction(summary, output_column="high_congestion_direction").sort_values(group)


def summarize_unique_realized_hours(detail: pd.DataFrame) -> pd.DataFrame:
    """Give every realized high-congestion ED hour equal weight.

    Multiple forecast runs may target the same realized timestamp. First average the paired
    forecast errors for each target × band × realized hour, then aggregate those hour-level
    values. This prevents frequent shadow runs from making one realized crowded hour count
    many times in the tail diagnostic.
    """
    tail = _prepare_tail(detail)
    if tail.empty:
        return pd.DataFrame()

    hour_group = ["target_name", "horizon_band", "ds"]
    by_hour = tail.groupby(hour_group, as_index=False).agg(
        actual=("actual", "first"),
        baseline_absolute_error=("baseline_absolute_error", "mean"),
        weather_absolute_error=("weather_absolute_error", "mean"),
        paired_absolute_error_delta=("paired_absolute_error_delta", "mean"),
        weather_win_rate=("weather_wins", "mean"),
        n_forecasts_for_realized_hour=("forecast_run_id", "nunique"),
        high_congestion_threshold_q75=("high_congestion_threshold_q75", "first"),
    )

    group = ["target_name", "horizon_band"]
    summary = by_hour.groupby(group, as_index=False).agg(
        n_unique_target_hours=("ds", "nunique"),
        high_congestion_threshold_q75=("high_congestion_threshold_q75", "first"),
        mean_actual=("actual", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        realized_hour_weather_win_rate=(
            "paired_absolute_error_delta", lambda x: (x > 0).mean()
        ),
        max_forecasts_per_realized_hour=("n_forecasts_for_realized_hour", "max"),
    )
    return _add_direction(
        summary, output_column="unique_hour_high_congestion_direction"
    ).sort_values(group)


def summarize_by_issue_date(detail: pd.DataFrame) -> pd.DataFrame:
    tail = _prepare_tail(detail)
    if tail.empty:
        return pd.DataFrame()

    group = ["target_name", "horizon_band", "forecast_issue_date"]
    result = tail.groupby(group, as_index=False).agg(
        n_high_congestion=("actual", "size"),
        n_unique_target_hours=("ds", "nunique"),
        baseline_mae=("baseline_absolute_error", "mean"),
        weather_mae=("weather_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        weather_win_rate=("weather_wins", "mean"),
        high_congestion_threshold_q75=("high_congestion_threshold_q75", "first"),
    )
    result["mae_improvement_pct"] = (
        (result["baseline_mae"] - result["weather_mae"])
        / result["baseline_mae"].replace(0, np.nan)
        * 100
    )
    result["issue_date_weather_better"] = result["weather_mae"] < result["baseline_mae"]
    return result.sort_values(group)


def summarize_issue_date_balanced(detail: pd.DataFrame) -> pd.DataFrame:
    by_date = summarize_by_issue_date(detail)
    if by_date.empty:
        return pd.DataFrame()

    group = ["target_name", "horizon_band"]
    balanced = by_date.groupby(group, as_index=False).agg(
        n_issue_dates=("forecast_issue_date", "nunique"),
        mean_issue_date_mae_improvement_pct=("mae_improvement_pct", "mean"),
        median_issue_date_mae_improvement_pct=("mae_improvement_pct", "median"),
        issue_date_win_rate=("issue_date_weather_better", "mean"),
        worst_issue_date_mae_improvement_pct=("mae_improvement_pct", "min"),
        p10_issue_date_mae_improvement_pct=("mae_improvement_pct", lambda x: x.quantile(0.10)),
    )
    balanced["harmful_issue_date_rate"] = 1.0 - balanced["issue_date_win_rate"]
    balanced["balanced_high_congestion_direction"] = np.where(
        balanced["mean_issue_date_mae_improvement_pct"] > 0,
        "weather_better",
        np.where(
            balanced["mean_issue_date_mae_improvement_pct"] < 0,
            "weather_worse",
            "tie",
        ),
    )
    return balanced.sort_values(group)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)

    pooled = summarize(detail)
    unique_hours = summarize_unique_realized_hours(detail)
    by_date = summarize_by_issue_date(detail)
    balanced = summarize_issue_date_balanced(detail)

    pooled.to_csv(args.output_dir / "weather-route-high-congestion.csv", index=False)
    unique_hours.to_csv(
        args.output_dir / "weather-route-high-congestion-unique-hours.csv", index=False
    )
    by_date.to_csv(
        args.output_dir / "weather-route-high-congestion-by-issue-date.csv", index=False
    )
    balanced.to_csv(
        args.output_dir / "weather-route-high-congestion-balanced.csv", index=False
    )

    if pooled.empty:
        print("No matured weather-active rows available for high-congestion diagnostics.")
    else:
        print("High-congestion prospective weather diagnostics (top quartile of unique realized target hours):")
        print(pooled.to_string(index=False))
        print("\nEqual-weight unique-realized-hour high-congestion diagnostics:")
        print(unique_hours.to_string(index=False))
        print("\nIssue-date-balanced high-congestion diagnostics:")
        print(balanced.to_string(index=False))


if __name__ == "__main__":
    main()
