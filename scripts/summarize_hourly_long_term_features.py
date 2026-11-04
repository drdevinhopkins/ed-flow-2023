#!/usr/bin/env python3
"""Aggregate sharded hourly annual-memory/growth backtest detail files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REFERENCE_WINDOW = 8192
REFERENCE_SCENARIO = "baseline"


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


def add_reference(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ref = table.loc[
        table["history_hours"].eq(REFERENCE_WINDOW)
        & table["scenario"].eq(REFERENCE_SCENARIO),
        [*keys, "mae"],
    ].rename(columns={"mae": "max_context_baseline_mae"})
    if ref.empty:
        raise ValueError("Missing max-context baseline reference")
    out = table.merge(ref, on=keys, how="left")
    out["mae_improvement_vs_max_baseline"] = out["max_context_baseline_mae"] - out["mae"]
    out["mae_improvement_vs_max_baseline_pct"] = (
        out["mae_improvement_vs_max_baseline"]
        / out["max_context_baseline_mae"].replace(0, np.nan)
        * 100
    )
    return out


def add_within_window_baseline(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    ref = table.loc[
        table["scenario"].eq("baseline"),
        [*keys, "history_hours", "mae"],
    ].rename(columns={"mae": "same_window_baseline_mae"})
    out = table.merge(ref, on=[*keys, "history_hours"], how="left")
    out["mae_improvement_vs_same_window"] = out["same_window_baseline_mae"] - out["mae"]
    out["mae_improvement_vs_same_window_pct"] = (
        out["mae_improvement_vs_same_window"]
        / out["same_window_baseline_mae"].replace(0, np.nan)
        * 100
    )
    return out


def winners(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = table.groupby(keys, observed=True)["mae"].idxmin()
    return table.loc[idx].sort_values(keys).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.input_root.glob("**/detail.csv"))
    if not files:
        raise FileNotFoundError(f"No detail.csv files below {args.input_root}")
    detail = pd.concat([pd.read_csv(path, parse_dates=["ds", "cutoff"]) for path in files], ignore_index=True)
    detail.to_csv(args.output_dir / "detail.csv", index=False)

    overall = metrics(detail, ["target_name", "history_label", "history_hours", "scenario"])
    overall = add_reference(overall, ["target_name"])
    overall = add_within_window_baseline(overall, ["target_name"])
    overall = overall.sort_values(["target_name", "mae"])
    overall.to_csv(args.output_dir / "summary.csv", index=False)

    by_band = metrics(
        detail,
        ["target_name", "horizon_band", "history_label", "history_hours", "scenario"],
    )
    by_band = add_reference(by_band, ["target_name", "horizon_band"])
    by_band = add_within_window_baseline(by_band, ["target_name", "horizon_band"])
    by_band = by_band.sort_values(["target_name", "horizon_band", "mae"])
    by_band.to_csv(args.output_dir / "by_horizon_band.csv", index=False)

    by_cutoff = metrics(
        detail,
        ["target_name", "cutoff", "history_label", "history_hours", "scenario"],
    )
    by_cutoff = add_reference(by_cutoff, ["target_name", "cutoff"])
    by_cutoff = add_within_window_baseline(by_cutoff, ["target_name", "cutoff"])
    by_cutoff = by_cutoff.sort_values(["cutoff", "target_name", "mae"])
    by_cutoff.to_csv(args.output_dir / "by_cutoff.csv", index=False)

    target_winners = winners(overall, ["target_name"])
    target_winners.to_csv(args.output_dir / "winners_by_target.csv", index=False)
    band_winners = winners(by_band, ["target_name", "horizon_band"])
    band_winners.to_csv(args.output_dir / "winners_by_target_horizon_band.csv", index=False)

    scenario_rank = overall.copy()
    scenario_rank["relative_mae"] = scenario_rank["mae"] / scenario_rank["max_context_baseline_mae"].replace(0, np.nan)
    global_rank = (
        scenario_rank.groupby(["history_label", "history_hours", "scenario"], as_index=False)
        .agg(
            mean_relative_mae=("relative_mae", "mean"),
            median_relative_mae=("relative_mae", "median"),
            mean_improvement_vs_max_baseline_pct=("mae_improvement_vs_max_baseline_pct", "mean"),
            mean_improvement_vs_same_window_pct=("mae_improvement_vs_same_window_pct", "mean"),
        )
        .sort_values(["mean_relative_mae", "history_hours"])
    )
    global_rank.to_csv(args.output_dir / "global_ranking.csv", index=False)

    print("=== Global ranking ===")
    print(global_rank.head(20).to_string(index=False))
    print("\n=== Best configuration per target ===")
    print(
        target_winners[
            [
                "target_name",
                "history_label",
                "scenario",
                "mae",
                "mae_improvement_vs_max_baseline_pct",
                "mae_improvement_vs_same_window_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
