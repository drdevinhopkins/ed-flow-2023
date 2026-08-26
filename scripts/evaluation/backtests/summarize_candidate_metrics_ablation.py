#!/usr/bin/env python3
"""Aggregate parallel candidate-metric feature-ablation detail files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_flow_metrics import CANDIDATE_TARGETS

EXPECTED_BANDS = ("h01_04", "h05_08", "h09_12", "h13_24")
WEATHER_SCENARIOS = {"weather_raw", "weather_raw_plus_snow"}


def metrics(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = detail.groupby(group_cols, as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)
    return table


def add_baseline_comparison(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    baseline = table.loc[table["scenario"].eq("baseline"), [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    out = table.merge(baseline, on=keys, how="left")
    out["mae_improvement"] = out["baseline_mae"] - out["mae"]
    out["mae_improvement_pct"] = (
        out["mae_improvement"] / out["baseline_mae"].replace(0, np.nan) * 100
    )
    out["beats_baseline"] = out["mae_improvement"].gt(0)
    return out


def winners(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = table.groupby(keys, observed=True)["mae"].idxmin()
    return table.loc[idx].sort_values(keys).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cutoffs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.rglob("detail-*.csv"))
    if len(paths) != args.expected_cutoffs:
        raise RuntimeError(f"Expected {args.expected_cutoffs} cutoff detail files, found {len(paths)}")

    detail = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if set(detail["target_name"]) != set(CANDIDATE_TARGETS):
        raise RuntimeError(f"Unexpected targets: {sorted(detail['target_name'].unique())}")
    if detail["cutoff"].nunique() != args.expected_cutoffs:
        raise RuntimeError("Unexpected number of unique cutoffs")
    if set(detail["horizon_band"]) != set(EXPECTED_BANDS):
        raise RuntimeError(f"Unexpected horizon bands: {sorted(detail['horizon_band'].unique())}")

    overall = add_baseline_comparison(
        metrics(detail, ["target_name", "scenario", "family"]), ["target_name"]
    ).sort_values(["target_name", "mae"])
    by_band = add_baseline_comparison(
        metrics(detail, ["target_name", "horizon_band", "scenario", "family"]),
        ["target_name", "horizon_band"],
    ).sort_values(["target_name", "horizon_band", "mae"])
    by_hour = add_baseline_comparison(
        metrics(detail, ["target_name", "horizon_hour", "scenario", "family"]),
        ["target_name", "horizon_hour"],
    ).sort_values(["target_name", "horizon_hour", "mae"])

    overall_winners = winners(overall, ["target_name"])
    band_winners = winners(by_band, ["target_name", "horizon_band"])

    safe_candidates = by_band.loc[
        ~by_band["scenario"].isin({"baseline", *WEATHER_SCENARIOS})
        & by_band["beats_baseline"]
    ].copy()
    safe_winners = (
        winners(safe_candidates, ["target_name", "horizon_band"])
        if not safe_candidates.empty
        else safe_candidates
    )

    # Cutoff-level paired improvements make it easy to judge whether aggregate winners are
    # robust or are being driven by a small number of weeks.
    cutoff_metrics = metrics(
        detail, ["cutoff", "target_name", "horizon_band", "scenario", "family"]
    )
    baseline = cutoff_metrics.loc[
        cutoff_metrics["scenario"].eq("baseline"),
        ["cutoff", "target_name", "horizon_band", "mae"],
    ].rename(columns={"mae": "baseline_mae"})
    paired = cutoff_metrics.merge(
        baseline, on=["cutoff", "target_name", "horizon_band"], how="left"
    )
    paired["paired_mae_improvement"] = paired["baseline_mae"] - paired["mae"]
    robustness = paired.groupby(
        ["target_name", "horizon_band", "scenario", "family"], as_index=False
    ).agg(
        cutoff_count=("cutoff", "nunique"),
        cutoff_wins=("paired_mae_improvement", lambda s: int((s > 0).sum())),
        mean_paired_improvement=("paired_mae_improvement", "mean"),
        median_paired_improvement=("paired_mae_improvement", "median"),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    overall.to_csv(args.output_dir / "summary.csv", index=False)
    by_band.to_csv(args.output_dir / "by_horizon_band.csv", index=False)
    by_hour.to_csv(args.output_dir / "by_horizon_hour.csv", index=False)
    overall_winners.to_csv(args.output_dir / "winners_by_target.csv", index=False)
    band_winners.to_csv(args.output_dir / "winners_by_target_horizon_band.csv", index=False)
    safe_winners.to_csv(args.output_dir / "safe_nonweather_winners_by_target_horizon_band.csv", index=False)
    robustness.to_csv(args.output_dir / "cutoff_robustness.csv", index=False)

    quality_paths = sorted(args.input_dir.rglob("candidate-quality.csv"))
    if quality_paths:
        pd.read_csv(quality_paths[0]).to_csv(args.output_dir / "candidate-quality.csv", index=False)

    print("=== Candidate winners by horizon band ===")
    print(
        band_winners[
            ["target_name", "horizon_band", "family", "scenario", "mae", "baseline_mae", "mae_improvement_pct", "beats_baseline"]
        ].to_string(index=False)
    )
    print("\n=== Production-safe non-weather candidates that beat baseline ===")
    if safe_winners.empty:
        print("None")
    else:
        print(
            safe_winners[
                ["target_name", "horizon_band", "family", "scenario", "mae", "baseline_mae", "mae_improvement_pct"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
