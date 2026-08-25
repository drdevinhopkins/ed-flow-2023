#!/usr/bin/env python3
"""Aggregate parallel triage-hallway feature-ablation detail files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRIAGE_TARGETS = ("TRG_HALLWAY1", "TRG_HALLWAY_TBS")
EXPECTED_BANDS = ("h01_04", "h05_08", "h09_12", "h13_24")


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
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(
        0, np.nan
    )
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
        raise RuntimeError(
            f"Expected {args.expected_cutoffs} cutoff detail files, found {len(paths)}"
        )

    detail = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    if set(detail["target_name"]) != set(TRIAGE_TARGETS):
        raise RuntimeError(f"Unexpected targets: {sorted(detail['target_name'].unique())}")
    cutoff_count = detail["cutoff"].nunique()
    if cutoff_count != args.expected_cutoffs:
        raise RuntimeError(f"Expected {args.expected_cutoffs} unique cutoffs, got {cutoff_count}")
    if set(detail["horizon_band"]) != set(EXPECTED_BANDS):
        raise RuntimeError(
            f"Unexpected horizon bands: {sorted(detail['horizon_band'].unique())}"
        )

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
    nonbaseline = overall.loc[~overall["scenario"].eq("baseline")].copy()
    best_nonbaseline = winners(nonbaseline, ["target_name"])

    safe_candidates = by_band.loc[
        ~by_band["scenario"].isin({"baseline", "weather_raw", "weather_raw_plus_snow"})
        & by_band["beats_baseline"]
    ].copy()
    if safe_candidates.empty:
        safe_winners = safe_candidates
    else:
        safe_winners = winners(safe_candidates, ["target_name", "horizon_band"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    overall.to_csv(args.output_dir / "summary.csv", index=False)
    by_band.to_csv(args.output_dir / "by_horizon_band.csv", index=False)
    by_hour.to_csv(args.output_dir / "by_horizon_hour.csv", index=False)
    overall_winners.to_csv(args.output_dir / "winners_by_target.csv", index=False)
    band_winners.to_csv(
        args.output_dir / "winners_by_target_horizon_band.csv", index=False
    )
    best_nonbaseline.to_csv(
        args.output_dir / "best_feature_family_by_target.csv", index=False
    )
    safe_winners.to_csv(
        args.output_dir / "safe_nonweather_winners_by_target_horizon_band.csv",
        index=False,
    )

    print("=== Triage hallway winners by horizon band ===")
    print(
        band_winners[
            [
                "target_name",
                "horizon_band",
                "family",
                "scenario",
                "mae",
                "baseline_mae",
                "mae_improvement_pct",
                "beats_baseline",
            ]
        ].to_string(index=False)
    )
    print("\n=== Best production-safe non-weather candidates that beat baseline ===")
    if safe_winners.empty:
        print("None")
    else:
        print(
            safe_winners[
                [
                    "target_name",
                    "horizon_band",
                    "family",
                    "scenario",
                    "mae",
                    "baseline_mae",
                    "mae_improvement_pct",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
