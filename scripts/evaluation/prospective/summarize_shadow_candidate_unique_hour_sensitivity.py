#!/usr/bin/env python3
"""Compare candidate routes using one forecast per realized target hour.

Repeated intraday shadow runs often forecast the same realized ED hour. This diagnostic
keeps either the earliest or latest available routed forecast for each realized target
hour, then recomputes paired candidate-vs-baseline performance. It is diagnostic only and
does not alter the pre-registered route map or promotion criteria.
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
        "--output",
        type=Path,
        default=Path(
            "validation/prospective-candidates/latest-score/"
            "candidate-route-unique-hour-sensitivity.csv"
        ),
    )
    return parser.parse_args()


def _summarize(frame: pd.DataFrame, selection: str) -> pd.DataFrame:
    group = ["target_name", "horizon_band", "candidate_scenario"]
    out = frame.groupby(group, as_index=False).agg(
        n_unique_target_hours=("target_ds", "nunique"),
        baseline_mae=("baseline_absolute_error", "mean"),
        candidate_mae=("candidate_absolute_error", "mean"),
        mean_paired_mae_delta=("paired_absolute_error_delta", "mean"),
        median_paired_mae_delta=("paired_absolute_error_delta", "median"),
        candidate_win_rate=("candidate_wins", "mean"),
        baseline_bias=("baseline_error", "mean"),
        candidate_bias=("candidate_error", "mean"),
    )
    out["mae_improvement_pct"] = (
        (out["baseline_mae"] - out["candidate_mae"])
        / out["baseline_mae"].replace(0, np.nan)
        * 100.0
    )
    out.insert(3, "selection", selection)
    return out


def summarize_unique_hour_sensitivity(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "forecast_issued_at",
        "target_ds",
        "target_name",
        "horizon_band",
        "candidate_scenario",
        "candidate_route_active",
        "baseline_absolute_error",
        "candidate_absolute_error",
        "paired_absolute_error_delta",
        "candidate_wins",
        "baseline_error",
        "candidate_error",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    frame = detail.loc[detail["candidate_route_active"].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame()

    frame["forecast_issued_at"] = pd.to_datetime(
        frame["forecast_issued_at"], utc=True, errors="coerce"
    )
    frame["target_ds"] = pd.to_datetime(frame["target_ds"], errors="coerce")
    if frame[["forecast_issued_at", "target_ds"]].isna().any().any():
        raise ValueError("Invalid forecast_issued_at or target_ds in candidate detail")

    key = ["target_name", "horizon_band", "candidate_scenario", "target_ds"]
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
        index=["target_name", "horizon_band", "candidate_scenario"],
        columns="selection",
        values="mae_improvement_pct",
        aggfunc="first",
    ).reset_index()
    early = "earliest_forecast_per_realized_hour"
    late = "latest_forecast_per_realized_hour"
    if early in pivot.columns and late in pivot.columns:
        pivot["selection_direction_agrees"] = (
            np.sign(pivot[early].fillna(0)) == np.sign(pivot[late].fillna(0))
        )
        result = result.merge(
            pivot[[
                "target_name", "horizon_band", "candidate_scenario",
                "selection_direction_agrees",
            ]],
            on=["target_name", "horizon_band", "candidate_scenario"],
            how="left",
        )
    else:
        result["selection_direction_agrees"] = False

    return result.sort_values(
        ["target_name", "horizon_band", "candidate_scenario", "selection"]
    )


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    summary = summarize_unique_hour_sensitivity(detail)
    summary.to_csv(args.output, index=False)
    if summary.empty:
        print("No matured candidate-route rows yet; unique-hour sensitivity is empty.")
    else:
        print("Prospective candidate unique-realized-hour sensitivity:")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
