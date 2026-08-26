#!/usr/bin/env python3
"""Summarize prospective weather performance by exact forecast lead hour.

This diagnostic is intentionally non-production. It uses only matured rows where the
weather-enabled route actually differed from the safe route, then reports paired
weather-vs-baseline performance for each exact horizon hour. It also produces an
issue-date-balanced view so repeated intraday forecasts cannot dominate the result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    aliases = {
        "baseline_abs_error": ["baseline_absolute_error"],
        "weather_abs_error": ["weather_absolute_error"],
        "baseline_error": ["baseline_signed_error"],
        "weather_error": ["weather_signed_error"],
        "weather_wins": ["candidate_wins"],
    }
    for canonical, candidates in aliases.items():
        if canonical not in out.columns:
            for candidate in candidates:
                if candidate in out.columns:
                    out[canonical] = out[candidate]
                    break

    required = {
        "target_name", "horizon_band", "horizon_hour", "forecast_issued_at",
        "baseline_abs_error", "weather_abs_error", "baseline_error", "weather_error",
    }
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing required scorer columns: {sorted(missing)}")

    out["forecast_issued_at"] = pd.to_datetime(out["forecast_issued_at"], utc=True, errors="coerce")
    out["forecast_issue_date"] = out["forecast_issued_at"].dt.date.astype("string")
    out["horizon_hour"] = pd.to_numeric(out["horizon_hour"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["forecast_issued_at", "horizon_hour"])

    if "weather_route_active" in out.columns:
        active = out["weather_route_active"].astype(str).str.lower().isin({"1", "true", "yes"})
    elif {"baseline_scenario", "weather_scenario"}.issubset(out.columns):
        active = out["baseline_scenario"].astype(str).ne(out["weather_scenario"].astype(str))
    else:
        active = pd.Series(True, index=out.index)
    return out.loc[active].copy()


def _summarize(group: pd.DataFrame) -> pd.Series:
    delta = group["baseline_abs_error"] - group["weather_abs_error"]
    baseline_mae = float(group["baseline_abs_error"].mean())
    weather_mae = float(group["weather_abs_error"].mean())
    return pd.Series({
        "n": len(group),
        "n_runs": group["forecast_run_id"].nunique() if "forecast_run_id" in group.columns else np.nan,
        "n_issue_dates": group["forecast_issue_date"].nunique(),
        "baseline_mae": baseline_mae,
        "weather_mae": weather_mae,
        "mean_paired_mae_delta": float(delta.mean()),
        "median_paired_mae_delta": float(delta.median()),
        "weather_win_rate": float((delta > 0).mean()),
        "baseline_bias": float(group["baseline_error"].mean()),
        "weather_bias": float(group["weather_error"].mean()),
        "mae_improvement_pct": ((baseline_mae - weather_mae) / baseline_mae * 100.0) if baseline_mae else np.nan,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail = _normalize(pd.read_csv(args.detail_csv))
    if detail.empty:
        print("No matured weather-active rows; exact-horizon summary skipped.")
        return

    by_hour = (
        detail.groupby(["target_name", "horizon_band", "horizon_hour"], dropna=False)
        .apply(_summarize, include_groups=False)
        .reset_index()
        .sort_values(["target_name", "horizon_hour"])
    )
    by_hour.to_csv(args.output_dir / "weather-route-by-horizon-hour.csv", index=False)

    by_date_hour = (
        detail.groupby(["target_name", "horizon_band", "horizon_hour", "forecast_issue_date"], dropna=False)
        .apply(_summarize, include_groups=False)
        .reset_index()
    )
    balanced = (
        by_date_hour.groupby(["target_name", "horizon_band", "horizon_hour"], as_index=False)
        .agg(
            n_issue_dates=("forecast_issue_date", "nunique"),
            issue_date_mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
            issue_date_median_mae_improvement_pct=("mae_improvement_pct", "median"),
            issue_date_win_rate=("mae_improvement_pct", lambda s: float((s > 0).mean())),
            worst_issue_date_mae_improvement_pct=("mae_improvement_pct", "min"),
        )
        .sort_values(["target_name", "horizon_hour"])
    )
    balanced.to_csv(args.output_dir / "weather-route-by-horizon-hour-balanced.csv", index=False)

    print("Prospective weather performance by exact lead hour:")
    print(by_hour.to_string(index=False))
    print("\nIssue-date-balanced exact-lead-hour summary:")
    print(balanced.to_string(index=False))


if __name__ == "__main__":
    main()
