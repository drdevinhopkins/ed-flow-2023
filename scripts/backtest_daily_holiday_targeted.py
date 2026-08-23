#!/usr/bin/env python3
"""Second-pass native Chronos-2 holiday ablation for JGH daily ED visits.

The first holiday-focused pass showed that generic holiday/proximity feature soup was
not uniformly helpful, while health-system closure/rebound structure improved daily
visit accuracy.  This script tests smaller, hypothesis-driven feature groups:

* calendar_closure: relevant calendars + seasonal clusters + closure/rebound structure,
  omitting broad any-holiday, generic before/after, proximity, and Israeli civic flags.
* targeted_edges: calendar_closure plus exact long-weekend shoulder positions such as
  Friday before a Monday holiday and Tuesday after a Monday holiday.
* targeted_block: targeted_edges plus position within a multi-day closure block and the
  number of holiday systems that overlap on the date.

All forecasts use Chronos2Pipeline.predict_df directly; AutoGluon is not involved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from backtest_holiday_features import (
    FLOW_URL,
    MAX_CONTEXT_DAYS,
    MODEL_ID,
    SERIES_ID,
    TARGET,
    contiguous_history,
    load_daily_visits,
    select_holiday_cutoffs,
)
from holiday_features import add_holiday_features

SCENARIOS = [
    "baseline",
    "calendars",
    "closures",
    "calendar_closure",
    "targeted_edges",
    "targeted_block",
]

CALENDAR_CLOSURE_COLUMNS = [
    "is_qc_holiday",
    "is_federal_holiday",
    "is_ramq_holiday",
    "is_major_jewish_holiday",
    "is_major_jewish_holiday_eve",
    "is_christmas_newyear_period",
    "is_quebec_canada_day_period",
    "is_system_closed_day",
    "closed_days_immediately_before",
    "closed_days_immediately_ahead",
    "closed_days_previous_7d",
    "closed_days_next_7d",
    "is_first_business_day_after_closure",
    "is_rebound_after_long_closure",
    "is_last_business_day_before_closure",
    "is_pre_long_closure",
]

EDGE_COLUMNS = [
    "is_friday_before_monday_holiday",
    "is_tuesday_after_monday_holiday",
    "is_thursday_before_friday_holiday",
    "is_monday_after_friday_holiday",
]

BLOCK_COLUMNS = [
    "holiday_system_count",
    "multiple_holiday_systems",
    "is_monday_holiday",
    "is_friday_holiday",
    "closure_block_length",
    "is_first_day_long_closure",
    "is_last_day_long_closure",
]

EVENT_COLUMNS = [
    "is_any_holiday",
    "is_qc_holiday",
    "is_ramq_holiday",
    "is_major_jewish_holiday",
    "is_major_jewish_holiday_eve",
    "is_friday_before_monday_holiday",
    "is_tuesday_after_monday_holiday",
    "is_thursday_before_friday_holiday",
    "is_monday_after_friday_holiday",
    "is_first_business_day_after_closure",
    "is_rebound_after_long_closure",
    "is_last_business_day_before_closure",
    "is_pre_long_closure",
    "is_christmas_newyear_period",
    "is_quebec_canada_day_period",
    "is_first_day_long_closure",
    "is_last_day_long_closure",
]


def engineer_targeted_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the full deterministic feature pool used by the targeted scenarios."""
    out = add_holiday_features(frame, feature_set="closures")
    weekday = pd.to_datetime(out["ds"]).dt.weekday

    calendar_flags = [
        "is_qc_holiday",
        "is_federal_holiday",
        "is_ramq_holiday",
        "is_major_jewish_holiday",
    ]
    out["holiday_system_count"] = out[calendar_flags].sum(axis=1).astype(np.int8)
    out["multiple_holiday_systems"] = (out["holiday_system_count"] >= 2).astype(np.int8)
    out["is_monday_holiday"] = (
        out["is_any_holiday"].astype(bool) & weekday.eq(0)
    ).astype(np.int8)
    out["is_friday_holiday"] = (
        out["is_any_holiday"].astype(bool) & weekday.eq(4)
    ).astype(np.int8)

    closed = out["is_system_closed_day"].astype(bool)
    before = out["closed_days_immediately_before"].astype(int)
    ahead = out["closed_days_immediately_ahead"].astype(int)
    out["closure_block_length"] = np.where(closed, before + 1 + ahead, 0).astype(np.int8)
    out["is_first_day_long_closure"] = (
        closed & before.eq(0) & ahead.ge(2)
    ).astype(np.int8)
    out["is_last_day_long_closure"] = (
        closed & ahead.eq(0) & before.ge(2)
    ).astype(np.int8)
    return out


def scenario_features(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Return frame plus only the known-future covariates requested by scenario."""
    if scenario == "baseline":
        return frame.copy()
    if scenario in {"calendars", "closures"}:
        return add_holiday_features(frame, feature_set=scenario)

    featured = engineer_targeted_features(frame)
    if scenario == "calendar_closure":
        selected = CALENDAR_CLOSURE_COLUMNS
    elif scenario == "targeted_edges":
        selected = [*CALENDAR_CLOSURE_COLUMNS, *EDGE_COLUMNS]
    elif scenario == "targeted_block":
        selected = [*CALENDAR_CLOSURE_COLUMNS, *EDGE_COLUMNS, *BLOCK_COLUMNS]
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    keep = list(frame.columns)
    for column in selected:
        if column not in keep:
            keep.append(column)
    return featured[keep]


def build_frames(
    daily: pd.DataFrame,
    *,
    scenario: str,
    cutoff: pd.Timestamp,
    horizon_days: int,
    context_days: int,
    min_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    history = contiguous_history(
        daily,
        cutoff,
        context_days=context_days,
        min_history_days=min_history_days,
    )
    future = pd.DataFrame(
        {"ds": pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")}
    )

    if scenario != "baseline":
        history = scenario_features(history, scenario)
        future = scenario_features(future, scenario)

    history["id"] = SERIES_ID
    if scenario == "baseline":
        return history[["id", "ds", TARGET]], None

    future["id"] = SERIES_ID
    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    for column in covariates:
        history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
        future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")

    return (
        history[["id", "ds", TARGET, *covariates]],
        future[["id", "ds", *covariates]],
    )


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    horizon_days: int,
    context_days: int,
) -> pd.DataFrame:
    kwargs = {
        "prediction_length": horizon_days,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": [TARGET],
        "quantile_levels": [0.5],
        "context_length": min(context_days, MAX_CONTEXT_DAYS, len(history)),
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    return result[["ds", "target_name", "predictions"]].rename(
        columns={"predictions": "prediction"}
    )


def actuals_with_labels(
    daily: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int
) -> pd.DataFrame:
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    actual = daily.loc[daily["ds"].isin(dates), ["ds", TARGET]].copy()
    if len(actual) != horizon_days or actual[TARGET].isna().any():
        raise ValueError(f"Incomplete actual horizon after cutoff {cutoff.date()}")
    actual = engineer_targeted_features(actual)
    actual["is_event_day"] = actual[EVENT_COLUMNS].max(axis=1).astype(np.int8)
    actual = actual.rename(columns={TARGET: "actual"})
    actual["target_name"] = TARGET
    return actual


def metric_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    table = frame.groupby(group_columns, as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table["abs_error_sum"] / table["abs_actual_sum"].replace(0, np.nan)
    return table.drop(columns=["abs_error_sum", "abs_actual_sum"])


def add_baseline_improvement(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    baseline = table.loc[table["scenario"] == "baseline", [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    table = table.merge(baseline, on=keys, how="left")
    table["mae_improvement"] = table["baseline_mae"] - table["mae"]
    table["mae_improvement_pct"] = (
        table["mae_improvement"] / table["baseline_mae"].replace(0, np.nan) * 100
    )
    return table


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for segment, mask in (
        ("all_days", pd.Series(True, index=detail.index)),
        ("holiday_window", detail["is_event_day"].astype(bool)),
        ("ordinary_days", ~detail["is_event_day"].astype(bool)),
    ):
        table = metric_table(detail.loc[mask], ["target_name", "scenario"])
        table.insert(0, "segment", segment)
        frames.append(table)
    summary = pd.concat(frames, ignore_index=True)
    return add_baseline_improvement(summary, ["segment", "target_name"]).sort_values(
        ["segment", "mae"]
    )


def summarize_by_event(detail: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for event in EVENT_COLUMNS:
        subset = detail.loc[detail[event].astype(bool)]
        if subset.empty:
            continue
        table = metric_table(subset, ["target_name", "scenario"])
        table.insert(0, "event", event.removeprefix("is_"))
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return add_baseline_improvement(result, ["event", "target_name"]).sort_values(
        ["event", "mae"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--num-cutoffs", type=int, default=24)
    parser.add_argument("--context-days", type=int, default=1095)
    parser.add_argument("--min-history-days", type=int, default=180)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--flow-url", default=FLOW_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = load_daily_visits(args.flow_url)
    cutoffs = select_holiday_cutoffs(
        daily,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
        num_cutoffs=args.num_cutoffs,
    )

    print(f"Scenarios: {', '.join(args.scenarios)}")
    print("Targeted daily cutoffs:")
    print(cutoffs[["cutoff", "ds", "event_type", "history_days"]].to_string(index=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff_row in cutoffs.itertuples(index=False):
        cutoff = pd.Timestamp(cutoff_row.cutoff)
        actual = actuals_with_labels(daily, cutoff, args.horizon_days)

        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff.date()} scenario={scenario}")
            history, future = build_frames(
                daily,
                scenario=scenario,
                cutoff=cutoff,
                horizon_days=args.horizon_days,
                context_days=args.context_days,
                min_history_days=args.min_history_days,
            )
            forecast = run_forecast(
                pipeline,
                history,
                future,
                horizon_days=args.horizon_days,
                context_days=args.context_days,
            )
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            if len(joined) != args.horizon_days:
                raise ValueError(
                    f"Expected {args.horizon_days} scored rows at {cutoff.date()}, got {len(joined)}"
                )
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_day"] = ((joined["ds"] - cutoff) / pd.Timedelta(days=1)).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)
    by_event = summarize_by_event(detail)
    winners = (
        summary.sort_values(["segment", "target_name", "mae"])
        .groupby(["segment", "target_name"], as_index=False)
        .first()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "daily_visit_holiday_targeted_detail.csv", index=False)
    summary.to_csv(args.output_dir / "daily_visit_holiday_targeted_summary.csv", index=False)
    by_event.to_csv(args.output_dir / "daily_visit_holiday_targeted_by_event.csv", index=False)
    winners.to_csv(args.output_dir / "daily_visit_holiday_targeted_winners.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nWinners:")
    print(
        winners[
            ["segment", "scenario", "mae", "baseline_mae", "mae_improvement_pct"]
        ].to_string(index=False)
    )
    print("\nBy event:")
    print(by_event.to_string(index=False))


if __name__ == "__main__":
    main()
