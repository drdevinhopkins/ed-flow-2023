#!/usr/bin/env python3
"""Run one common cutoff of the hourly feature-family ablation for triage targets.

This helper parallelizes the expensive apples-to-apples Chronos-2 validation by cutoff.
Each job still forecasts the complete production target set and all feature scenarios;
only the two triage-hallway target rows are retained for aggregation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_final_features as final_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from staffing_features import build_schedule_feature_frames

TRIAGE_TARGETS = ("TRG_HALLWAY1", "TRG_HALLWAY_TBS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-index", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--min-history-hours", type=int, default=672)
    parser.add_argument("--effect-min-hours", type=int, default=24)
    parser.add_argument("--effect-shrinkage-hours", type=float, default=72.0)
    parser.add_argument("--model-id", default=final_bt.MODEL_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.cutoff_index < args.num_cutoffs:
        raise ValueError(
            f"cutoff-index must be in 0..{args.num_cutoffs - 1}, got {args.cutoff_index}"
        )

    flow = staffing_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)

    cutoffs = final_bt.select_common_cutoffs(
        flow,
        staffing,
        weather,
        schedule_frames,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )
    if len(cutoffs) != args.num_cutoffs:
        raise ValueError(f"Expected {args.num_cutoffs} cutoffs, got {len(cutoffs)}")
    cutoff = cutoffs[args.cutoff_index]
    calendar = final_bt.build_calendar_frame(flow, [cutoff], args.horizon)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cutoff index={args.cutoff_index}; cutoff={cutoff}; device={device}")
    print(f"Targets retained: {', '.join(TRIAGE_TARGETS)}")
    print(f"Scenarios: {', '.join(final_bt.SCENARIOS)}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    actual = base.actuals_long(flow, cutoff, args.horizon)
    frames: list[pd.DataFrame] = []
    for scenario in final_bt.SCENARIOS:
        print(f"Forecasting cutoff={cutoff} scenario={scenario}")
        history, future = final_bt.scenario_frames(
            scenario,
            flow=flow,
            staffing=staffing,
            weather=weather,
            shifts=shifts,
            schedule_frames=schedule_frames,
            calendar=calendar,
            cutoff=cutoff,
            horizon=args.horizon,
            max_history_days=args.max_history_days,
            effect_min_hours=args.effect_min_hours,
            effect_shrinkage_hours=args.effect_shrinkage_hours,
        )
        forecast = final_bt.run_forecast(
            pipeline, history, future, horizon=args.horizon
        )
        joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
        joined["cutoff"] = cutoff
        joined["scenario"] = scenario
        joined["family"] = final_bt.FAMILY[scenario]
        joined["horizon_hour"] = (
            (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
        ).astype(int)
        frames.append(final_bt.add_errors(joined))

    detail = pd.concat(frames, ignore_index=True)
    detail = detail.loc[detail["target_name"].isin(TRIAGE_TARGETS)].copy()
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["h01_04", "h05_08", "h09_12", "h13_24"],
        include_lowest=True,
    ).astype(str)

    expected = len(TRIAGE_TARGETS) * len(final_bt.SCENARIOS) * args.horizon
    if len(detail) != expected:
        raise RuntimeError(f"Expected {expected} retained rows, got {len(detail)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / f"detail-{args.cutoff_index}.csv", index=False)
    pd.DataFrame(
        {"cutoff_index": [args.cutoff_index], "cutoff": [cutoff]}
    ).to_csv(args.output_dir / f"cutoff-{args.cutoff_index}.csv", index=False)
    print(f"Saved {len(detail)} rows for cutoff index {args.cutoff_index}")


if __name__ == "__main__":
    main()
