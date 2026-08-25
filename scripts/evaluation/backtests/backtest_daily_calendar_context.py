#!/usr/bin/env python3
"""Third-pass native Chronos-2 ablation for JGH daily ED arrivals.

This pass keeps the second-pass ``calendar_closure`` feature set as the main control and
asks whether social-calendar context adds signal beyond annual seasonality:

* Quebec construction vacation and its pre/post weeks.
* Compact aggregate school-break/back-to-school features.
* Separate French (CSSDM), English (EMSB), and Jewish-school proxy features.
* A combined social-calendar set with transition distances and break-type details.

The target remains total daily ED arrivals (sum of hourly Inflow_Total). AutoGluon is not
used; all forecasts call Chronos2Pipeline.predict_df directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from backtest_daily_holiday_targeted import (
    CALENDAR_CLOSURE_COLUMNS,
    add_baseline_improvement,
    engineer_targeted_features,
    metric_table,
    run_forecast,
)
from backtest_holiday_features import (
    FLOW_URL,
    MODEL_ID,
    SERIES_ID,
    TARGET,
    contiguous_history,
    load_daily_visits,
)
from calendar_context_features import add_calendar_context_features

SCENARIOS = [
    "baseline",
    "calendar_closure",
    "construction",
    "school_core",
    "school_systems",
    "social_calendar",
]

CONSTRUCTION_COLUMNS = [
    "is_construction_holiday",
    "is_construction_holiday_start",
    "is_construction_holiday_end",
    "is_week_before_construction_holiday",
    "is_week_after_construction_holiday",
    "construction_holiday_day",
]

SCHOOL_CORE_COLUMNS = [
    "is_any_school_break_proxy",
    "school_systems_closed_count",
    "is_any_back_to_school_window",
    "school_transition_intensity",
]

SCHOOL_SYSTEM_COLUMNS = [
    *SCHOOL_CORE_COLUMNS,
    "is_french_school_break_proxy",
    "is_english_school_break_proxy",
    "is_jewish_school_break_proxy",
    "is_jewish_school_religious_break_proxy",
    "is_french_school_start",
    "is_english_school_start",
    "is_jewish_school_start_proxy",
    "is_french_back_to_school_window",
    "is_english_back_to_school_window",
    "is_jewish_back_to_school_window",
]

SOCIAL_DETAIL_COLUMNS = [
    "is_french_summer_break_proxy",
    "is_english_summer_break_proxy",
    "is_jewish_summer_break_proxy",
    "is_french_winter_break_proxy",
    "is_english_winter_break_proxy",
    "is_jewish_winter_break_proxy",
    "is_french_spring_break_proxy",
    "is_english_spring_break_proxy",
    "is_jewish_spring_break_proxy",
    "days_since_french_school_start",
    "days_since_english_school_start",
    "days_since_jewish_school_start",
    "days_to_french_school_start",
    "days_to_english_school_start",
    "days_to_jewish_school_start",
    "is_split_school_transition",
]

EVENT_COLUMNS = [
    "is_construction_holiday",
    "is_week_before_construction_holiday",
    "is_week_after_construction_holiday",
    "is_french_school_start",
    "is_english_school_start",
    "is_jewish_school_start_proxy",
    "is_french_back_to_school_window",
    "is_english_back_to_school_window",
    "is_jewish_back_to_school_window",
    "is_french_spring_break_proxy",
    "is_english_spring_break_proxy",
    "is_jewish_spring_break_proxy",
    "is_jewish_school_religious_break_proxy",
]


def engineer_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    return add_calendar_context_features(engineer_targeted_features(frame))


def scenario_features(frame: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "baseline":
        return frame.copy()

    featured = engineer_context_features(frame)
    if scenario == "calendar_closure":
        selected = CALENDAR_CLOSURE_COLUMNS
    elif scenario == "construction":
        selected = [*CALENDAR_CLOSURE_COLUMNS, *CONSTRUCTION_COLUMNS]
    elif scenario == "school_core":
        selected = [*CALENDAR_CLOSURE_COLUMNS, *SCHOOL_CORE_COLUMNS]
    elif scenario == "school_systems":
        selected = [*CALENDAR_CLOSURE_COLUMNS, *SCHOOL_SYSTEM_COLUMNS]
    elif scenario == "social_calendar":
        selected = [
            *CALENDAR_CLOSURE_COLUMNS,
            *CONSTRUCTION_COLUMNS,
            *SCHOOL_SYSTEM_COLUMNS,
            *SOCIAL_DETAIL_COLUMNS,
        ]
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
    return history[["id", "ds", TARGET, *covariates]], future[["id", "ds", *covariates]]


def label_social_events(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_context_features(frame.copy())
    out["is_social_event_day"] = out[EVENT_COLUMNS].max(axis=1).astype(np.int8)
    return out


def _event_type(row: pd.Series) -> str:
    labels = [column.removeprefix("is_") for column in EVENT_COLUMNS if int(row.get(column, 0))]
    return "|".join(labels) if labels else "none"


def select_social_cutoffs(
    daily: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
    min_history_days: int,
    num_cutoffs: int,
) -> pd.DataFrame:
    """Select diverse onsets of construction/school events with clean scoring horizons."""
    labelled = label_social_events(daily[["ds"]].copy())
    active = labelled[EVENT_COLUMNS].astype(bool)
    onset = active & ~active.shift(1, fill_value=False)
    candidates = labelled.loc[onset.any(axis=1)].copy()
    candidates["event_type"] = candidates.apply(_event_type, axis=1)
    candidates["cutoff"] = candidates["ds"] - pd.Timedelta(days=1)

    eligible_rows: list[dict[str, object]] = []
    for row in candidates.itertuples(index=False):
        event_day = pd.Timestamp(row.ds)
        cutoff = pd.Timestamp(row.cutoff)
        future_dates = pd.date_range(event_day, periods=horizon_days, freq="D")
        actual = daily.loc[daily["ds"].isin(future_dates), ["ds", TARGET]]
        if len(actual) != horizon_days or actual[TARGET].isna().any():
            continue
        try:
            history = contiguous_history(
                daily,
                cutoff,
                context_days=context_days,
                min_history_days=min_history_days,
            )
        except ValueError:
            continue
        record = row._asdict()
        record["history_days"] = len(history)
        eligible_rows.append(record)

    if not eligible_rows:
        raise ValueError("No eligible construction/school event cutoffs")

    eligible = pd.DataFrame(eligible_rows).sort_values("cutoff").reset_index(drop=True)
    if len(eligible) > num_cutoffs:
        positions = np.linspace(0, len(eligible) - 1, num=num_cutoffs)
        indices = sorted(set(int(round(value)) for value in positions))
        eligible = eligible.iloc[indices].copy()
    return eligible[["cutoff", "ds", "event_type", "history_days"]].reset_index(drop=True)


def actuals_with_labels(
    daily: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int
) -> pd.DataFrame:
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    actual = daily.loc[daily["ds"].isin(dates), ["ds", TARGET]].copy()
    if len(actual) != horizon_days or actual[TARGET].isna().any():
        raise ValueError(f"Incomplete actual horizon after cutoff {cutoff.date()}")
    actual = label_social_events(actual)
    actual = actual.rename(columns={TARGET: "actual"})
    actual["target_name"] = TARGET
    return actual


def add_calendar_closure_improvement(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if table.empty:
        return table
    control = table.loc[
        table["scenario"] == "calendar_closure", [*keys, "mae"]
    ].rename(columns={"mae": "calendar_closure_mae"})
    table = table.merge(control, on=keys, how="left")
    table["mae_improvement_vs_calendar_closure"] = table["calendar_closure_mae"] - table["mae"]
    table["mae_improvement_vs_calendar_closure_pct"] = (
        table["mae_improvement_vs_calendar_closure"]
        / table["calendar_closure_mae"].replace(0, np.nan)
        * 100
    )
    return table


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for segment, mask in (
        ("all_days", pd.Series(True, index=detail.index)),
        ("social_window", detail["is_social_event_day"].astype(bool)),
        ("outside_social_window", ~detail["is_social_event_day"].astype(bool)),
    ):
        table = metric_table(detail.loc[mask], ["target_name", "scenario"])
        table.insert(0, "segment", segment)
        frames.append(table)
    result = pd.concat(frames, ignore_index=True)
    result = add_baseline_improvement(result, ["segment", "target_name"])
    result = add_calendar_closure_improvement(result, ["segment", "target_name"])
    return result.sort_values(["segment", "mae"]).reset_index(drop=True)


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
    result = add_baseline_improvement(result, ["event", "target_name"])
    result = add_calendar_closure_improvement(result, ["event", "target_name"])
    return result.sort_values(["event", "mae"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--num-cutoffs", type=int, default=24)
    parser.add_argument("--context-days", type=int, default=1095)
    parser.add_argument("--min-history-days", type=int, default=180)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-social"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--flow-url", default=FLOW_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = load_daily_visits(args.flow_url)
    cutoffs = select_social_cutoffs(
        daily,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
        num_cutoffs=args.num_cutoffs,
    )
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print("Social-calendar cutoffs:")
    print(cutoffs.to_string(index=False))

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
    cutoffs.to_csv(args.output_dir / "daily_visit_calendar_context_cutoffs.csv", index=False)
    detail.to_csv(args.output_dir / "daily_visit_calendar_context_detail.csv", index=False)
    summary.to_csv(args.output_dir / "daily_visit_calendar_context_summary.csv", index=False)
    by_event.to_csv(args.output_dir / "daily_visit_calendar_context_by_event.csv", index=False)
    winners.to_csv(args.output_dir / "daily_visit_calendar_context_winners.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nWinners:")
    print(
        winners[
            [
                "segment",
                "scenario",
                "mae",
                "baseline_mae",
                "calendar_closure_mae",
                "mae_improvement_vs_calendar_closure_pct",
            ]
        ].to_string(index=False)
    )
    print("\nBy event:")
    print(by_event.to_string(index=False))


if __name__ == "__main__":
    main()
