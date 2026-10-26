#!/usr/bin/env python3
"""Score a selective exact-hour weather-routing counterfactual.

Diagnostic only: this script never changes routing. It compares three paired policies on
matured rows for one target/horizon band: all-baseline, the current weather-enabled shadow
policy, and a selective policy that uses weather only at chosen exact lead hours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _metrics(frame: pd.DataFrame, prediction_col: str, label: str) -> dict[str, float | int | str]:
    error = frame[prediction_col] - frame["actual"]
    abs_error = error.abs()
    baseline_abs = (frame["baseline_prediction"] - frame["actual"]).abs()
    delta = baseline_abs - abs_error
    baseline_mae = float(baseline_abs.mean())
    mae = float(abs_error.mean())
    return {
        "policy": label,
        "n": int(len(frame)),
        "n_runs": int(frame["forecast_run_id"].nunique()),
        "n_issue_dates": int(frame["forecast_issue_date"].nunique()),
        "n_unique_target_hours": int(frame["ds"].nunique()),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement_pct_vs_baseline": ((baseline_mae - mae) / baseline_mae * 100.0) if baseline_mae else np.nan,
        "mean_paired_mae_delta_vs_baseline": float(delta.mean()),
        "median_paired_mae_delta_vs_baseline": float(delta.median()),
        "win_rate_vs_baseline": float((delta > 0).mean()),
        "bias": float(error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", default="Total_TBS")
    parser.add_argument("--horizon-band", default="h01_04")
    parser.add_argument("--weather-hours", type=int, nargs="+", default=[3, 4])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.detail_csv)
    required = {
        "forecast_run_id", "forecast_issued_at", "ds", "target_name", "horizon_band",
        "horizon_hour", "actual", "baseline_prediction", "weather_prediction"
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required scorer columns: {sorted(missing)}")

    frame["forecast_issued_at"] = pd.to_datetime(frame["forecast_issued_at"], utc=True, errors="coerce")
    frame["forecast_issue_date"] = frame["forecast_issued_at"].dt.date.astype("string")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame["horizon_hour"] = pd.to_numeric(frame["horizon_hour"], errors="coerce").astype("Int64")
    frame = frame.loc[
        frame["target_name"].eq(args.target)
        & frame["horizon_band"].eq(args.horizon_band)
    ].dropna(subset=["forecast_issued_at", "ds", "horizon_hour", "actual"]).copy()
    if frame.empty:
        print("No matured rows for requested selective-policy diagnostic.")
        return

    selected = frame["horizon_hour"].isin(args.weather_hours)
    frame["selective_prediction"] = np.where(
        selected, frame["weather_prediction"], frame["baseline_prediction"]
    )
    frame["selective_uses_weather"] = selected

    rows = [
        _metrics(frame, "baseline_prediction", "all_baseline"),
        _metrics(frame, "weather_prediction", "current_weather_policy"),
        _metrics(frame, "selective_prediction", "selective_weather_hours_" + "_".join(map(str, args.weather_hours))),
    ]
    summary = pd.DataFrame(rows)
    current_mae = float(summary.loc[summary["policy"].eq("current_weather_policy"), "mae"].iloc[0])
    summary["mae_improvement_pct_vs_current_weather_policy"] = (
        (current_mae - summary["mae"]) / current_mae * 100.0 if current_mae else np.nan
    )
    summary["target_name"] = args.target
    summary["horizon_band"] = args.horizon_band
    summary["weather_hours"] = ",".join(map(str, args.weather_hours))
    summary.to_csv(args.output_dir / "weather-selective-policy.csv", index=False)

    by_date_rows = []
    for issue_date, group in frame.groupby("forecast_issue_date"):
        for prediction_col, label in [
            ("baseline_prediction", "all_baseline"),
            ("weather_prediction", "current_weather_policy"),
            ("selective_prediction", "selective_weather_hours_" + "_".join(map(str, args.weather_hours))),
        ]:
            row = _metrics(group, prediction_col, label)
            row["forecast_issue_date"] = issue_date
            by_date_rows.append(row)
    by_date = pd.DataFrame(by_date_rows)
    by_date.to_csv(args.output_dir / "weather-selective-policy-by-issue-date.csv", index=False)

    print("Selective weather-routing counterfactual:")
    print(summary.to_string(index=False))
    print("\nBy issue date:")
    print(by_date.to_string(index=False))


if __name__ == "__main__":
    main()
