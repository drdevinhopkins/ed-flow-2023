#!/usr/bin/env python3
"""Run one common-cutoff Chronos-2 feature ablation for candidate operational targets.

The experiment keeps the existing eight production targets in the Chronos target bundle and
adds five candidate operational targets. Only the candidate rows are retained for scoring.
No production routing is changed by this script.

Candidate definitions:
* Hourly_Inflow_Total: raw hourly Inflow_Total
* Hourly_Inflow_Stretcher: raw hourly INFLOW_STRETCHER
* Hourly_Inflow_Ambulances: raw hourly INFLOW_AMBULANCES
* AdmissionRequests_New: reset-aware hourly increment of CUM_ADMREQ
* Workup_Delay_Burden: sum of delayed consult/imaging counters across POD + RAZ;
  this is a burden index, not a unique-patient count, because components can overlap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_calendar_features as calendar_bt
import backtest_hourly_final_features as final_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from forecast_oncall_impact import derive_flow_metrics
from staffing_features import build_schedule_feature_frames, parse_hour

PRODUCTION_TARGETS = tuple(base.FLOW_TARGETS)
CANDIDATE_TARGETS = (
    "Hourly_Inflow_Total",
    "Hourly_Inflow_Stretcher",
    "Hourly_Inflow_Ambulances",
    "AdmissionRequests_New",
    "Workup_Delay_Burden",
)
ALL_TARGETS = (*PRODUCTION_TARGETS, *CANDIDATE_TARGETS)

DELAY_COMPONENTS = (
    "POD_CONS_MORE2H",
    "POD_IMCONS_MORE4H",
    "POD_XRAY_MORE2H",
    "POD_CT_MORE2H",
    "RAZ_CONS_MORE2H",
    "RAZ_IMCONS_MORE4H",
    "RAZ_XRAY_MORE2H",
    "RAZ_CT_MORE2H1",
)


def _patch_target_lists() -> None:
    """Point the isolated backtest modules at the 13-target experimental bundle."""
    targets = list(ALL_TARGETS)
    base.FLOW_TARGETS = targets
    staffing_bt.FLOW_TARGETS = targets
    calendar_bt.FLOW_TARGETS = targets
    weather_bt.FLOW_TARGETS = targets
    final_bt.FLOW_TARGETS = targets


def _reset_aware_increment(cumulative: pd.Series) -> pd.Series:
    """Convert a cumulative counter that may reset to an hourly non-negative increment."""
    current = pd.to_numeric(cumulative, errors="coerce")
    delta = current.diff()
    out = delta.where(delta.ge(0), current)
    out = out.where(delta.notna(), current)
    return out.clip(lower=0)


def load_candidate_flow(url: str = staffing_bt.FLOW_URL) -> pd.DataFrame:
    raw = pd.read_csv(url)
    raw["ds"] = parse_hour(raw["ds"])
    raw = raw.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    derived, _ = derive_flow_metrics(raw)
    source_map = {
        "Total_TBS": "total_tbs",
        "POD_TBS": "pod_tbs",
        "Vertical_TBS": "vertical_tbs",
        "Overflow": "overflow",
    }
    for target, source in source_map.items():
        if target not in derived.columns:
            if source not in derived.columns:
                raise ValueError(f"Could not derive required production target {target}")
            derived[target] = pd.to_numeric(derived[source], errors="coerce")

    required_raw = [
        "Inflow_Total",
        "INFLOW_STRETCHER",
        "INFLOW_AMBULANCES",
        "CUM_ADMREQ",
        *DELAY_COMPONENTS,
    ]
    missing_raw = [column for column in required_raw if column not in derived.columns]
    if missing_raw:
        raise ValueError(f"Missing candidate source column(s): {', '.join(missing_raw)}")

    derived["Hourly_Inflow_Total"] = pd.to_numeric(derived["Inflow_Total"], errors="coerce")
    derived["Hourly_Inflow_Stretcher"] = pd.to_numeric(
        derived["INFLOW_STRETCHER"], errors="coerce"
    )
    derived["Hourly_Inflow_Ambulances"] = pd.to_numeric(
        derived["INFLOW_AMBULANCES"], errors="coerce"
    )
    derived["AdmissionRequests_New"] = _reset_aware_increment(derived["CUM_ADMREQ"])

    delay = derived[list(DELAY_COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    derived["Workup_Delay_Burden"] = delay.sum(axis=1, min_count=1)

    missing_targets = [target for target in ALL_TARGETS if target not in derived.columns]
    if missing_targets:
        raise ValueError(f"Missing experimental target(s): {', '.join(missing_targets)}")

    flow = derived[["ds", *ALL_TARGETS]].copy()
    for target in ALL_TARGETS:
        flow[target] = pd.to_numeric(flow[target], errors="coerce")

    # Match the existing hourly pipeline's causal regularization: construct an hourly grid
    # and use past-only carry-forward for sparse missing snapshots. This is acceptable for
    # feature-family comparison; prospective scoring should later preserve raw event-rate
    # observations separately so missing snapshots are not mistaken for true zero arrivals.
    index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    flow = flow.set_index("ds").reindex(index).reset_index()
    for target in ALL_TARGETS:
        flow[target] = flow[target].ffill()
        if flow[target].notna().sum() == 0:
            raise ValueError(f"Target contains no numeric observations: {target}")

    return flow


def candidate_profile(flow: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in CANDIDATE_TARGETS:
        series = pd.to_numeric(flow[target], errors="coerce")
        rows.append(
            {
                "target_name": target,
                "n": int(series.notna().sum()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "zero_fraction": float(series.eq(0).mean()),
            }
        )
    return pd.DataFrame(rows)


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

    _patch_target_lists()
    flow = load_candidate_flow()
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
    print(f"Experimental bundle: {len(ALL_TARGETS)} targets")
    print(f"Candidates retained: {', '.join(CANDIDATE_TARGETS)}")
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
        forecast = final_bt.run_forecast(pipeline, history, future, horizon=args.horizon)
        joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
        joined["cutoff"] = cutoff
        joined["scenario"] = scenario
        joined["family"] = final_bt.FAMILY[scenario]
        joined["horizon_hour"] = (
            (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
        ).astype(int)
        frames.append(final_bt.add_errors(joined))

    detail = pd.concat(frames, ignore_index=True)
    detail = detail.loc[detail["target_name"].isin(CANDIDATE_TARGETS)].copy()
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["h01_04", "h05_08", "h09_12", "h13_24"],
        include_lowest=True,
    ).astype(str)

    expected = len(CANDIDATE_TARGETS) * len(final_bt.SCENARIOS) * args.horizon
    if len(detail) != expected:
        raise RuntimeError(f"Expected {expected} retained rows, got {len(detail)}")
    if not np.isfinite(detail["prediction"]).all():
        raise RuntimeError("Non-finite predictions found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / f"detail-{args.cutoff_index}.csv", index=False)
    pd.DataFrame({"cutoff_index": [args.cutoff_index], "cutoff": [cutoff]}).to_csv(
        args.output_dir / f"cutoff-{args.cutoff_index}.csv", index=False
    )
    candidate_profile(flow).to_csv(
        args.output_dir / f"target-profile-{args.cutoff_index}.csv", index=False
    )
    print(f"Saved {len(detail)} candidate scoring rows for cutoff index {args.cutoff_index}")


if __name__ == "__main__":
    main()
