#!/usr/bin/env python3
"""Compare nominal RAMQ dates with JGH establishment 0011X dates for daily visits.

This is an apples-to-apples native Chronos-2 ablation. Both forecast scenarios use the
same ``calendar_closure`` covariates, cutoffs, target, context and model. The only change
is the RAMQ calendar used to build ``is_ramq_holiday`` and closure/rebound structure:

* ``calendar_closure_nominal_ramq``: generic 13-date RAMQ approximation.
* ``calendar_closure_jgh_ramq``: JGH / 0011X institution-specific dates.

The baseline is retained to make the magnitude interpretable. AutoGluon is not used.
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
    EVENT_COLUMNS,
    run_forecast,
)
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
    "calendar_closure_nominal_ramq",
    "calendar_closure_jgh_ramq",
]


def _calendar_closure(frame: pd.DataFrame, *, ramq_calendar: str) -> pd.DataFrame:
    featured = add_holiday_features(
        frame,
        feature_set="closures",
        ramq_calendar=ramq_calendar,
    )
    keep = list(frame.columns)
    keep.extend(column for column in CALENDAR_CLOSURE_COLUMNS if column not in keep)
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

    if scenario == "baseline":
        history["id"] = SERIES_ID
        return history[["id", "ds", TARGET]], None

    ramq_calendar = "nominal" if scenario.endswith("nominal_ramq") else "jgh"
    history = _calendar_closure(history, ramq_calendar=ramq_calendar)
    future = _calendar_closure(future, ramq_calendar=ramq_calendar)
    history["id"] = SERIES_ID
    future["id"] = SERIES_ID

    covariates = [
        column
        for column in CALENDAR_CLOSURE_COLUMNS
        if column in history.columns and column in future.columns
    ]
    for column in covariates:
        history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
        future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")

    return (
        history[["id", "ds", TARGET, *covariates]],
        future[["id", "ds", *covariates]],
    )


def actuals_with_labels(
    daily: pd.DataFrame, cutoff: pd.Timestamp, horizon_days: int
) -> pd.DataFrame:
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    actual = daily.loc[daily["ds"].isin(dates), ["ds", TARGET]].copy()
    if len(actual) != horizon_days or actual[TARGET].isna().any():
        raise ValueError(f"Incomplete actual horizon after cutoff {cutoff.date()}")

    labels = add_holiday_features(actual[["ds"]], feature_set="closures", ramq_calendar="jgh")
    # Two EVENT_COLUMNS (first/last day of a long closure) are targeted-script derived
    # features rather than base holiday_features outputs. They are irrelevant to this
    # RAMQ-only comparison, so initialize any unavailable event labels to zero.
    for column in EVENT_COLUMNS:
        actual[column] = labels[column].to_numpy() if column in labels.columns else 0
    actual["is_event_day"] = actual[EVENT_COLUMNS].max(axis=1).astype(np.int8)
    actual = actual.rename(columns={TARGET: "actual"})
    actual["target_name"] = TARGET
    return actual


def metric_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
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


def add_comparators(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    baseline = table.loc[table["scenario"] == "baseline", [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    nominal = table.loc[
        table["scenario"] == "calendar_closure_nominal_ramq", [*keys, "mae"]
    ].rename(columns={"mae": "nominal_ramq_mae"})
    table = table.merge(baseline, on=keys, how="left").merge(nominal, on=keys, how="left")
    table["mae_improvement_vs_baseline"] = table["baseline_mae"] - table["mae"]
    table["mae_improvement_vs_baseline_pct"] = (
        table["mae_improvement_vs_baseline"] / table["baseline_mae"].replace(0, np.nan) * 100
    )
    table["mae_improvement_vs_nominal_ramq"] = table["nominal_ramq_mae"] - table["mae"]
    table["mae_improvement_vs_nominal_ramq_pct"] = (
        table["mae_improvement_vs_nominal_ramq"]
        / table["nominal_ramq_mae"].replace(0, np.nan)
        * 100
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
    return add_comparators(
        pd.concat(frames, ignore_index=True), ["segment", "target_name"]
    ).sort_values(["segment", "mae"])


def summarize_ramq_days(detail: pd.DataFrame) -> pd.DataFrame:
    subset = detail.loc[detail["is_ramq_holiday"].astype(bool)]
    if subset.empty:
        return pd.DataFrame()
    table = metric_table(subset, ["target_name", "scenario"])
    table.insert(0, "segment", "jgh_ramq_days")
    return add_comparators(table, ["segment", "target_name"]).sort_values("mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--num-cutoffs", type=int, default=24)
    parser.add_argument("--context-days", type=int, default=1095)
    parser.add_argument("--min-history-days", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-ramq"))
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

    print("RAMQ calendar ablation scenarios:", ", ".join(SCENARIOS))
    print(cutoffs[["cutoff", "ds", "event_type", "history_days"]].to_string(index=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff_row in cutoffs.itertuples(index=False):
        cutoff = pd.Timestamp(cutoff_row.cutoff)
        actual = actuals_with_labels(daily, cutoff, args.horizon_days)
        for scenario in SCENARIOS:
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
                context_days=min(args.context_days, MAX_CONTEXT_DAYS),
            )
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            if len(joined) != args.horizon_days:
                raise ValueError(
                    f"Expected {args.horizon_days} scored rows at {cutoff.date()}, got {len(joined)}"
                )
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)
    ramq_days = summarize_ramq_days(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "daily_visit_jgh_ramq_ablation_detail.csv", index=False)
    summary.to_csv(args.output_dir / "daily_visit_jgh_ramq_ablation_summary.csv", index=False)
    ramq_days.to_csv(args.output_dir / "daily_visit_jgh_ramq_ablation_ramq_days.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nJGH RAMQ days:")
    print(ramq_days.to_string(index=False))


if __name__ == "__main__":
    main()
