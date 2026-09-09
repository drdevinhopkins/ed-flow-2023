#!/usr/bin/env python3
"""Backtest useful Chronos-2 hourly history/context windows.

Chronos-2 has a maximum inference context of 8,192 time steps. For hourly JGH
flow data this is about 341.3 days, so fixed training starts older than that are
silently equivalent at inference: Chronos keeps only the most recent 8,192
observations. This experiment therefore compares shorter context windows against
the model maximum rather than moving an arbitrary multi-year start date.

The experiment intentionally uses the target-only baseline to isolate history
length from calendar/weather/staffing feature-family choices. If a materially
better window emerges, it can then be re-tested with the target x horizon routed
production feature set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base

FLOW_TARGETS = base.FLOW_TARGETS
MODEL_ID = base.MODEL_ID
MODEL_MAX_CONTEXT = 8192
DEFAULT_WINDOWS = [720, 1440, 2160, 4320, 6480, 8192]


def window_label(hours: int) -> str:
    if hours == MODEL_MAX_CONTEXT:
        return "max_8192h"
    if hours % 24 == 0:
        return f"{hours // 24}d"
    return f"{hours}h"


def select_cutoffs(
    flow: pd.DataFrame,
    *,
    horizon: int,
    num_cutoffs: int,
    spacing_hours: int,
    max_history_hours: int,
) -> list[pd.Timestamp]:
    earliest = flow["ds"].min() + pd.Timedelta(hours=max_history_hours - 1)
    latest = flow["ds"].max() - pd.Timedelta(hours=horizon)
    if latest < earliest:
        raise ValueError(f"No eligible backtest range: {earliest} to {latest}")

    indexed = flow.set_index("ds")
    cutoffs: list[pd.Timestamp] = []
    current = latest.floor("h")
    while current >= earliest and len(cutoffs) < num_cutoffs:
        future_hours = pd.date_range(
            current + pd.Timedelta(hours=1), periods=horizon, freq="h"
        )
        actual = indexed.reindex(future_hours)[FLOW_TARGETS]
        if len(actual) == horizon and not actual.isna().any().any():
            cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)

    if len(cutoffs) < num_cutoffs:
        print(
            f"WARNING: requested {num_cutoffs} cutoffs but only found {len(cutoffs)} "
            "with the required maximum history."
        )
    if not cutoffs:
        raise ValueError("No eligible historical cutoffs found")
    return sorted(cutoffs)


def build_history(
    flow: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    history_hours: int,
) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(hours=history_hours - 1)
    history = flow.loc[
        flow["ds"].between(start, cutoff), ["ds", *FLOW_TARGETS]
    ].copy()
    if len(history) != history_hours:
        raise ValueError(
            f"Expected {history_hours} history rows at {cutoff}, got {len(history)}"
        )
    if history[FLOW_TARGETS].isna().any().any():
        bad = history[FLOW_TARGETS].columns[
            history[FLOW_TARGETS].isna().any()
        ].tolist()
        raise ValueError(f"Missing target history at {cutoff}: {bad}")
    history["id"] = "jgh"
    return history[["id", "ds", *FLOW_TARGETS]]


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    *,
    horizon: int,
    history_hours: int,
) -> pd.DataFrame:
    result = pipeline.predict_df(
        history,
        prediction_length=horizon,
        id_column="id",
        timestamp_column="ds",
        target=FLOW_TARGETS,
        quantile_levels=[0.5],
        context_length=history_hours,
    )
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].rename(
        columns={"predictions": "prediction"}
    )


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["abs_actual"] = out["actual"].abs()
    return out


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
    table["wape"] = table.pop("abs_error_sum") / table.pop(
        "abs_actual_sum"
    ).replace(0, np.nan)
    return table


def add_max_context_comparison(
    table: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    reference = table.loc[
        table["history_hours"].eq(MODEL_MAX_CONTEXT), [*keys, "mae"]
    ].rename(columns={"mae": "max_context_mae"})
    if reference.empty:
        raise ValueError("The 8192-hour reference window is required")
    out = table.merge(reference, on=keys, how="left")
    out["mae_improvement_vs_max"] = out["max_context_mae"] - out["mae"]
    out["mae_improvement_vs_max_pct"] = (
        out["mae_improvement_vs_max"]
        / out["max_context_mae"].replace(0, np.nan)
        * 100
    )
    out["beats_max_context"] = out["mae_improvement_vs_max"].gt(0)
    return out


def global_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    scored["relative_mae_vs_max"] = scored["mae"] / scored[
        "max_context_mae"
    ].replace(0, np.nan)
    ranking = scored.groupby(
        ["history_label", "history_hours", "history_days_equivalent"],
        as_index=False,
    ).agg(
        mean_relative_mae=("relative_mae_vs_max", "mean"),
        median_relative_mae=("relative_mae_vs_max", "median"),
        targets_beating_max=("beats_max_context", "sum"),
    )
    ranking["mean_improvement_vs_max_pct"] = (
        1 - ranking["mean_relative_mae"]
    ) * 100
    return ranking.sort_values(
        ["mean_relative_mae", "history_hours"], ignore_index=True
    )


def winners(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = table.groupby(keys, observed=True)["mae"].idxmin()
    return table.loc[idx].sort_values(keys).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=1008)
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS,
        help="History/context windows in hours (must include 8192).",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation-output-hourly-history-windows"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = sorted(set(args.windows))
    if min(args.horizon, args.num_cutoffs, args.spacing_hours, *windows) < 1:
        raise ValueError("Backtest sizes and history windows must be positive")
    if MODEL_MAX_CONTEXT not in windows:
        raise ValueError(f"--windows must include the {MODEL_MAX_CONTEXT}-hour reference")
    if max(windows) > MODEL_MAX_CONTEXT:
        raise ValueError(
            f"Chronos-2 context cannot exceed {MODEL_MAX_CONTEXT}; got {max(windows)}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    flow = base.load_flow()
    cutoffs = select_cutoffs(
        flow,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        max_history_hours=max(windows),
    )
    pd.DataFrame({"cutoff": cutoffs}).to_csv(
        args.output_dir / "cutoffs.csv", index=False
    )

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(
        "Windows: "
        + ", ".join(f"{window_label(hours)}={hours}h" for hours in windows)
    )
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print(
        "Reference: Chronos-2 maximum context = 8192 hourly observations "
        f"({MODEL_MAX_CONTEXT / 24:.1f} days)."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = base.actuals_long(flow, cutoff, args.horizon)
        for history_hours in windows:
            label = window_label(history_hours)
            print(
                f"Forecasting cutoff={cutoff} history={label} ({history_hours}h)"
            )
            history = build_history(
                flow, cutoff=cutoff, history_hours=history_hours
            )
            forecast = run_forecast(
                pipeline,
                history,
                horizon=args.horizon,
                history_hours=history_hours,
            )
            joined = forecast.merge(
                actual, on=["ds", "target_name"], how="inner"
            )
            joined["cutoff"] = cutoff
            joined["history_label"] = label
            joined["history_hours"] = history_hours
            joined["history_days_equivalent"] = history_hours / 24.0
            joined["horizon_hour"] = (
                (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
            ).astype(int)
            frames.append(add_errors(joined))

    if not frames:
        raise RuntimeError("Backtest produced no rows")
    detail = pd.concat(frames, ignore_index=True).dropna(
        subset=["prediction", "actual"]
    )
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["h01_04", "h05_08", "h09_12", "h13_24"],
        include_lowest=True,
    ).astype(str)
    detail.to_csv(args.output_dir / "detail.csv", index=False)

    overall = add_max_context_comparison(
        metrics(
            detail,
            [
                "target_name",
                "history_label",
                "history_hours",
                "history_days_equivalent",
            ],
        ),
        ["target_name"],
    ).sort_values(["target_name", "mae"])
    overall.to_csv(args.output_dir / "summary.csv", index=False)

    by_band = add_max_context_comparison(
        metrics(
            detail,
            [
                "target_name",
                "horizon_band",
                "history_label",
                "history_hours",
                "history_days_equivalent",
            ],
        ),
        ["target_name", "horizon_band"],
    ).sort_values(["target_name", "horizon_band", "mae"])
    by_band.to_csv(args.output_dir / "by_horizon_band.csv", index=False)

    by_cutoff = add_max_context_comparison(
        metrics(
            detail,
            [
                "target_name",
                "cutoff",
                "history_label",
                "history_hours",
                "history_days_equivalent",
            ],
        ),
        ["target_name", "cutoff"],
    ).sort_values(["cutoff", "target_name", "mae"])
    by_cutoff.to_csv(args.output_dir / "by_cutoff.csv", index=False)

    ranking = global_ranking(overall)
    ranking.to_csv(args.output_dir / "global_ranking.csv", index=False)

    target_winners = winners(overall, ["target_name"])
    target_winners.to_csv(args.output_dir / "winners_by_target.csv", index=False)
    band_winners = winners(by_band, ["target_name", "horizon_band"])
    band_winners.to_csv(
        args.output_dir / "winners_by_target_horizon_band.csv", index=False
    )

    print("\n=== Global ranking (normalized across targets; lower is better) ===")
    print(
        ranking[
            [
                "history_label",
                "history_hours",
                "mean_relative_mae",
                "mean_improvement_vs_max_pct",
                "targets_beating_max",
            ]
        ].to_string(index=False)
    )
    print("\n=== Best history window per target ===")
    print(
        target_winners[
            [
                "target_name",
                "history_label",
                "history_hours",
                "mae",
                "mae_improvement_vs_max_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
