#!/usr/bin/env python3
"""Production-routed Chronos-2 backtest across hourly history windows.

This follow-up to ``backtest_hourly_history_windows.py`` tests the history window
inside the current forecast-v2 modeling path rather than target-only baseline
forecasts. For every cutoff/window it generates the same safe production
scenarios (weather routing disabled), then selects the validated scenario for
each target and forecast-horizon hour using ``hourly_feature_routing``.

The current production setting is 365 days. Chronos-2 itself truncates model
context to its 8,192-step maximum (~341.3 days hourly), while upstream staffing
effect estimation can still use the full 365-day frame. Thus the 365-day arm is
the correct production-pipeline reference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_final_features as final_bt
import backtest_hourly_history_windows as history_bt
import backtest_staffing_features as staffing_bt
from hourly_feature_routing import (
    FLOW_TARGETS,
    horizon_band,
    scenario_for,
    scenarios_needed,
)
from staffing_features import build_schedule_feature_frames

MODEL_ID = base.MODEL_ID
REFERENCE_DAYS = 365
DEFAULT_WINDOWS_DAYS = [30, 60, 90, 180, 270, 365]
EFFECT_MIN_HOURS = 24
EFFECT_SHRINKAGE_HOURS = 72.0


def window_label(days: int) -> str:
    return "production_365d" if days == REFERENCE_DAYS else f"{days}d"


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


def add_reference_comparison(
    table: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    reference = table.loc[
        table["history_days"].eq(REFERENCE_DAYS), [*keys, "mae"]
    ].rename(columns={"mae": "production_365d_mae"})
    if reference.empty:
        raise ValueError("The 365-day production reference is required")
    out = table.merge(reference, on=keys, how="left")
    out["mae_improvement_vs_production"] = (
        out["production_365d_mae"] - out["mae"]
    )
    out["mae_improvement_vs_production_pct"] = (
        out["mae_improvement_vs_production"]
        / out["production_365d_mae"].replace(0, np.nan)
        * 100
    )
    out["beats_production"] = out[
        "mae_improvement_vs_production"
    ].gt(0)
    return out


def global_ranking(summary: pd.DataFrame) -> pd.DataFrame:
    scored = summary.copy()
    scored["relative_mae_vs_production"] = (
        scored["mae"] / scored["production_365d_mae"].replace(0, np.nan)
    )
    ranking = scored.groupby(
        ["history_label", "history_days"], as_index=False
    ).agg(
        mean_relative_mae=("relative_mae_vs_production", "mean"),
        median_relative_mae=("relative_mae_vs_production", "median"),
        targets_beating_production=("beats_production", "sum"),
    )
    ranking["mean_improvement_vs_production_pct"] = (
        1 - ranking["mean_relative_mae"]
    ) * 100
    return ranking.sort_values(
        ["mean_relative_mae", "history_days"], ignore_index=True
    )


def winners(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    idx = table.groupby(keys, observed=True)["mae"].idxmin()
    return table.loc[idx].sort_values(keys).reset_index(drop=True)


def build_routed_forecast(
    pipeline: Chronos2Pipeline,
    *,
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    shifts: pd.DataFrame,
    schedule_frames,
    calendar: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    history_days: int,
) -> pd.DataFrame:
    required_scenarios = sorted(scenarios_needed(allow_weather=False))
    outputs: list[pd.DataFrame] = []

    # Weather is not used by any safe route; keep a shape-compatible placeholder
    # because final_bt.scenario_frames has one common signature for all families.
    weather_placeholder = pd.DataFrame({"ds": pd.Series(dtype="datetime64[ns]")})

    for scenario in required_scenarios:
        print(
            f"  scenario={scenario} history={window_label(history_days)}"
        )
        history, future = final_bt.scenario_frames(
            scenario,
            flow=flow,
            staffing=staffing,
            weather=weather_placeholder,
            shifts=shifts,
            schedule_frames=schedule_frames,
            calendar=calendar,
            cutoff=cutoff,
            horizon=horizon,
            max_history_days=history_days,
            effect_min_hours=EFFECT_MIN_HOURS,
            effect_shrinkage_hours=EFFECT_SHRINKAGE_HOURS,
        )
        forecast = final_bt.run_forecast(
            pipeline, history, future, horizon=horizon
        ).copy()
        forecast["scenario"] = scenario
        outputs.append(forecast)

    combined = pd.concat(outputs, ignore_index=True)
    combined["horizon_hour"] = (
        (combined["ds"] - cutoff) / pd.Timedelta(hours=1)
    ).astype(int)
    combined["selected_scenario"] = [
        scenario_for(target, hour, allow_weather=False)
        for target, hour in zip(
            combined["target_name"], combined["horizon_hour"]
        )
    ]
    routed = combined.loc[
        combined["scenario"].eq(combined["selected_scenario"])
    ].copy()

    expected_rows = len(FLOW_TARGETS) * horizon
    if len(routed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} routed rows at {cutoff}; "
            f"got {len(routed)}"
        )
    if routed[["ds", "target_name"]].duplicated().any():
        raise RuntimeError(f"Duplicate routed forecasts at {cutoff}")
    return routed.drop(columns=["selected_scenario"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=1008)
    parser.add_argument(
        "--windows-days",
        type=int,
        nargs="+",
        default=DEFAULT_WINDOWS_DAYS,
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation-output-hourly-history-routed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = sorted(set(args.windows_days))
    if min(args.horizon, args.num_cutoffs, args.spacing_hours, *windows) < 1:
        raise ValueError("Backtest sizes and history windows must be positive")
    if REFERENCE_DAYS not in windows:
        raise ValueError(
            f"--windows-days must include production reference {REFERENCE_DAYS}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    flow = staffing_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()

    # Use the exact cutoff-selection logic from the target-only experiment so
    # the follow-up is directly comparable and each origin has 8,192 hours.
    cutoffs = history_bt.select_cutoffs(
        flow,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        max_history_hours=history_bt.MODEL_MAX_CONTEXT,
    )
    calendar = final_bt.build_calendar_frame(flow, cutoffs, args.horizon)
    pd.DataFrame({"cutoff": cutoffs}).to_csv(
        args.output_dir / "cutoffs.csv", index=False
    )

    required_scenarios = sorted(scenarios_needed(allow_weather=False))
    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Safe scenarios: {', '.join(required_scenarios)}")
    print(
        "Windows: "
        + ", ".join(
            f"{window_label(days)}={days}d" for days in windows
        )
    )
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print(
        "Reference is current production MAX_HISTORY_DAYS=365; "
        "Chronos model context within that arm is capped at 8,192 hours."
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = base.actuals_long(flow, cutoff, args.horizon)
        for history_days in windows:
            label = window_label(history_days)
            print(
                f"Forecasting cutoff={cutoff} history={label} "
                f"({history_days}d)"
            )
            routed = build_routed_forecast(
                pipeline,
                flow=flow,
                staffing=staffing,
                shifts=shifts,
                schedule_frames=schedule_frames,
                calendar=calendar,
                cutoff=cutoff,
                horizon=args.horizon,
                history_days=history_days,
            )
            joined = routed.merge(
                actual, on=["ds", "target_name"], how="inner"
            )
            joined["cutoff"] = cutoff
            joined["history_label"] = label
            joined["history_days"] = history_days
            joined["horizon_band"] = joined["horizon_hour"].map(horizon_band)
            frames.append(add_errors(joined))

    if not frames:
        raise RuntimeError("Backtest produced no rows")
    detail = pd.concat(frames, ignore_index=True).dropna(
        subset=["prediction", "actual"]
    )
    detail.to_csv(args.output_dir / "detail.csv", index=False)

    overall = add_reference_comparison(
        metrics(
            detail,
            ["target_name", "history_label", "history_days"],
        ),
        ["target_name"],
    ).sort_values(["target_name", "mae"])
    overall.to_csv(args.output_dir / "summary.csv", index=False)

    by_band = add_reference_comparison(
        metrics(
            detail,
            [
                "target_name",
                "horizon_band",
                "history_label",
                "history_days",
            ],
        ),
        ["target_name", "horizon_band"],
    ).sort_values(["target_name", "horizon_band", "mae"])
    by_band.to_csv(
        args.output_dir / "by_horizon_band.csv", index=False
    )

    by_cutoff = add_reference_comparison(
        metrics(
            detail,
            [
                "target_name",
                "cutoff",
                "history_label",
                "history_days",
            ],
        ),
        ["target_name", "cutoff"],
    ).sort_values(["cutoff", "target_name", "mae"])
    by_cutoff.to_csv(args.output_dir / "by_cutoff.csv", index=False)

    ranking = global_ranking(overall)
    ranking.to_csv(args.output_dir / "global_ranking.csv", index=False)

    target_winners = winners(overall, ["target_name"])
    target_winners.to_csv(
        args.output_dir / "winners_by_target.csv", index=False
    )
    band_winners = winners(
        by_band, ["target_name", "horizon_band"]
    )
    band_winners.to_csv(
        args.output_dir / "winners_by_target_horizon_band.csv",
        index=False,
    )

    route_mix = (
        detail.groupby(
            ["target_name", "horizon_band", "scenario"], as_index=False
        )
        .size()
        .rename(columns={"size": "rows"})
    )
    route_mix.to_csv(args.output_dir / "route_mix.csv", index=False)

    print(
        "\n=== Global routed ranking "
        "(normalized across targets; lower is better) ==="
    )
    print(
        ranking[
            [
                "history_label",
                "history_days",
                "mean_relative_mae",
                "mean_improvement_vs_production_pct",
                "targets_beating_production",
            ]
        ].to_string(index=False)
    )
    print("\n=== Best routed history window per target ===")
    print(
        target_winners[
            [
                "target_name",
                "history_label",
                "history_days",
                "mae",
                "mae_improvement_vs_production_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
