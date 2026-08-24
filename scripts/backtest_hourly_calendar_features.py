#!/usr/bin/env python3
"""Event-focused native Chronos-2 ablation of calendar features for hourly JGH flow.

Targets are the six canonical operational flow metrics already used by the native
Chronos-2 workflow. The experiment deliberately isolates calendar signal first; staffing
and weather are not included here so a winning calendar representation can later be
layered onto the full production covariate stack without confounding this comparison.

Scenarios:
* baseline: timestamps + targets only.
* demand_calendar: Quebec/federal/nominal RAMQ/major-Jewish and closure/rebound features.
* demand_plus_jgh_flag: demand calendar + exact JGH RAMQ holiday flag.
* demand_plus_jgh_mismatch: demand calendar + JGH-only/nominal-only mismatch flags.
* demand_plus_jgh_interactions: mismatch representation + time-of-day interactions.

Cutoffs are intentionally enriched for holidays, RAMQ/JGH mismatches, closure edges and
matched ordinary days. Results therefore answer whether calendar features help around
calendar-sensitive periods; they are not an estimate of unconditional year-round error.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from backtest_covariate_ablation import FLOW_TARGETS, MODEL_ID, load_flow
from hourly_calendar_features import add_hourly_calendar_features, scenario_columns

SCENARIOS = [
    "baseline",
    "demand_calendar",
    "demand_plus_jgh_flag",
    "demand_plus_jgh_mismatch",
    "demand_plus_jgh_interactions",
]
SERIES_ID = "jgh"

EVENT_LABEL_COLUMNS = [
    "is_qc_holiday",
    "is_federal_holiday",
    "is_ramq_holiday",
    "is_major_jewish_holiday",
    "is_major_jewish_holiday_eve",
    "is_jgh_ramq_holiday",
    "is_jgh_only_ramq_holiday",
    "is_nominal_only_ramq_holiday",
    "is_ramq_calendar_mismatch",
    "is_rebound_after_long_closure",
    "is_pre_long_closure",
    "is_first_morning_after_jgh_holiday",
    "is_evening_before_jgh_holiday",
]


def _evenly_pick(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    ordered = frame.sort_values("date").drop_duplicates("date")
    if len(ordered) <= n:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, n).round().astype(int)
    return ordered.iloc[np.unique(positions)]


def _event_type(row: pd.Series) -> str:
    names = [column.removeprefix("is_") for column in EVENT_LABEL_COLUMNS if int(row.get(column, 0))]
    return "|".join(names) if names else "ordinary"


def select_event_cutoffs(
    flow: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    horizon: int,
    num_cutoffs: int,
    min_history_hours: int,
) -> pd.DataFrame:
    """Select reproducible event-enriched 23:00 cutoffs whose next 24h span a date."""
    labels = calendar[["ds", *EVENT_LABEL_COLUMNS]].copy()
    labels["date"] = labels["ds"].dt.normalize()
    daily = labels.groupby("date", as_index=False)[EVENT_LABEL_COLUMNS].max()
    daily["event_type"] = daily.apply(_event_type, axis=1)
    daily["cutoff"] = daily["date"] - pd.Timedelta(hours=1)

    earliest = flow["ds"].min() + pd.Timedelta(hours=min_history_hours)
    latest = flow["ds"].max() - pd.Timedelta(hours=horizon)
    daily = daily.loc[daily["cutoff"].between(earliest, latest)].copy()

    target_index = flow.set_index("ds")[FLOW_TARGETS]
    eligible = []
    for row in daily.itertuples(index=False):
        hours = pd.date_range(row.cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
        values = target_index.reindex(hours)
        eligible.append(len(values) == horizon and not values.isna().any().any())
    daily = daily.loc[eligible].reset_index(drop=True)
    if daily.empty:
        raise ValueError("No eligible event-focused hourly cutoffs found")

    selected_dates: set[pd.Timestamp] = set()
    selected_frames: list[pd.DataFrame] = []

    def add_group(mask: pd.Series, quota: int) -> None:
        nonlocal selected_dates
        candidates = daily.loc[mask & ~daily["date"].isin(selected_dates)]
        picked = _evenly_pick(candidates, min(quota, max(0, num_cutoffs - len(selected_dates))))
        if not picked.empty:
            selected_frames.append(picked)
            selected_dates.update(pd.Timestamp(value) for value in picked["date"])

    add_group(daily["is_jgh_only_ramq_holiday"].eq(1), 4)
    add_group(daily["is_nominal_only_ramq_holiday"].eq(1), 4)
    add_group(
        daily[["is_first_morning_after_jgh_holiday", "is_evening_before_jgh_holiday"]]
        .max(axis=1)
        .eq(1),
        3,
    )
    add_group(daily["is_rebound_after_long_closure"].eq(1), 3)
    add_group(daily["is_pre_long_closure"].eq(1), 3)
    add_group(
        daily[["is_major_jewish_holiday", "is_major_jewish_holiday_eve"]].max(axis=1).eq(1),
        2,
    )
    add_group(
        daily[["is_qc_holiday", "is_federal_holiday", "is_ramq_holiday"]].max(axis=1).eq(1),
        2,
    )
    any_event = daily[EVENT_LABEL_COLUMNS].max(axis=1).astype(bool)
    add_group(~any_event, 3)

    if len(selected_dates) < num_cutoffs:
        remaining = daily.loc[~daily["date"].isin(selected_dates)]
        picked = _evenly_pick(remaining, num_cutoffs - len(selected_dates))
        if not picked.empty:
            selected_frames.append(picked)
            selected_dates.update(pd.Timestamp(value) for value in picked["date"])

    selected = pd.concat(selected_frames, ignore_index=True) if selected_frames else daily.iloc[0:0]
    selected = selected.drop_duplicates("date").sort_values("cutoff").head(num_cutoffs).reset_index(drop=True)
    if selected.empty:
        raise ValueError("Cutoff selection produced no rows")
    return selected[["cutoff", "date", "event_type", *EVENT_LABEL_COLUMNS]]


def scenario_frames(
    flow: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    scenario: str,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    history = flow.loc[flow["ds"].between(start, cutoff)].copy()
    if history.empty or history[FLOW_TARGETS].isna().any().any():
        raise ValueError(f"Incomplete target history at cutoff {cutoff}")

    history["id"] = SERIES_ID
    columns = scenario_columns(scenario)
    if not columns:
        return history[["id", "ds", *FLOW_TARGETS]], None

    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    feature_index = calendar.set_index("ds")
    history_features = feature_index.reindex(history["ds"])[columns].reset_index(drop=True)
    future_features = feature_index.reindex(future_hours)[columns]
    if history_features.isna().any().any() or future_features.isna().any().any():
        raise ValueError(f"Missing deterministic calendar features at cutoff {cutoff}")

    for column in columns:
        history[column] = pd.to_numeric(history_features[column], errors="raise").astype("float64").to_numpy()

    future_features.index.name = "ds"
    future = future_features.reset_index()
    future["id"] = SERIES_ID
    for column in columns:
        future[column] = pd.to_numeric(future[column], errors="raise").astype("float64")

    return (
        history[["id", "ds", *FLOW_TARGETS, *columns]],
        future[["id", "ds", *columns]],
    )


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    horizon: int,
) -> pd.DataFrame:
    kwargs = {
        "prediction_length": horizon,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": FLOW_TARGETS,
        "quantile_levels": [0.5],
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    return result[["ds", "target_name", "predictions"]].rename(
        columns={"predictions": "prediction"}
    )


def actuals_with_labels(
    flow: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    high_thresholds: dict[str, float],
) -> pd.DataFrame:
    hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.set_index("ds").reindex(hours)[FLOW_TARGETS]
    actual.index.name = "ds"
    actual = actual.reset_index()
    labels = calendar.set_index("ds").reindex(hours)
    labels.index.name = "ds"
    labels = labels.reset_index()
    label_columns = [column for column in EVENT_LABEL_COLUMNS if column in labels.columns]
    actual = actual.merge(labels[["ds", *label_columns]], on="ds", how="left")
    actual["hour"] = actual["ds"].dt.hour
    actual["daypart"] = np.select(
        [actual["hour"].between(8, 15), actual["hour"].between(16, 23)],
        ["daytime", "evening"],
        default="overnight",
    )
    long = actual.melt(
        id_vars=["ds", "hour", "daypart", *label_columns],
        value_vars=FLOW_TARGETS,
        var_name="target_name",
        value_name="actual",
    )
    long["high_congestion"] = [
        int(value >= high_thresholds[target])
        for target, value in zip(long["target_name"], long["actual"])
    ]
    return long


def _metrics(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = frame.groupby(group_cols, as_index=False).agg(
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


def _add_comparators(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    baseline = table.loc[table["scenario"].eq("baseline"), [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    demand = table.loc[table["scenario"].eq("demand_calendar"), [*keys, "mae"]].rename(
        columns={"mae": "demand_calendar_mae"}
    )
    out = table.merge(baseline, on=keys, how="left").merge(demand, on=keys, how="left")
    out["mae_improvement_vs_baseline_pct"] = (
        (out["baseline_mae"] - out["mae"]) / out["baseline_mae"].replace(0, np.nan) * 100
    )
    out["mae_improvement_vs_demand_pct"] = (
        (out["demand_calendar_mae"] - out["mae"])
        / out["demand_calendar_mae"].replace(0, np.nan)
        * 100
    )
    return out


def summarize_segments(detail: pd.DataFrame) -> pd.DataFrame:
    segment_masks = {
        "all_hours": pd.Series(True, index=detail.index),
        "public_or_nominal_ramq": detail[["is_qc_holiday", "is_federal_holiday", "is_ramq_holiday"]]
        .max(axis=1)
        .astype(bool),
        "major_jewish": detail["is_major_jewish_holiday"].astype(bool),
        "jgh_ramq": detail["is_jgh_ramq_holiday"].astype(bool),
        "jgh_only_ramq": detail["is_jgh_only_ramq_holiday"].astype(bool),
        "nominal_only_ramq": detail["is_nominal_only_ramq_holiday"].astype(bool),
        "rebound_after_long_closure": detail["is_rebound_after_long_closure"].astype(bool),
        "pre_long_closure": detail["is_pre_long_closure"].astype(bool),
        "morning_after_jgh_holiday": detail["is_first_morning_after_jgh_holiday"].astype(bool),
        "evening_before_jgh_holiday": detail["is_evening_before_jgh_holiday"].astype(bool),
        "daytime": detail["daypart"].eq("daytime"),
        "evening": detail["daypart"].eq("evening"),
        "overnight": detail["daypart"].eq("overnight"),
        "high_congestion": detail["high_congestion"].astype(bool),
    }
    frames = []
    for segment, mask in segment_masks.items():
        subset = detail.loc[mask]
        if subset.empty:
            continue
        table = _metrics(subset, ["target_name", "scenario"])
        table.insert(0, "segment", segment)
        frames.append(table)
    combined = pd.concat(frames, ignore_index=True)
    return _add_comparators(combined, ["segment", "target_name"]).sort_values(
        ["segment", "target_name", "mae"]
    )


def summarize_horizon(detail: pd.DataFrame) -> pd.DataFrame:
    out = detail.copy()
    out["horizon_band"] = pd.cut(
        out["horizon_hour"],
        bins=[0, 6, 12, 18, 24],
        labels=["h01_06", "h07_12", "h13_18", "h19_24"],
        include_lowest=True,
    ).astype(str)
    table = _metrics(out, ["target_name", "horizon_band", "scenario"])
    return _add_comparators(table, ["target_name", "horizon_band"]).sort_values(
        ["target_name", "horizon_band", "mae"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=24)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--min-history-hours", type=int, default=24 * 30)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-hourly-calendar"))
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.horizon, args.num_cutoffs, args.max_history_days, args.min_history_hours) < 1:
        raise ValueError("Backtest arguments must be positive")
    if args.horizon != 24:
        print("Note: cutoff selection is date-centered; 24h is the production-comparable horizon.")

    flow = load_flow()
    calendar = add_hourly_calendar_features(flow[["ds"]])
    cutoffs = select_event_cutoffs(
        flow,
        calendar,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        min_history_hours=args.min_history_hours,
    )
    high_thresholds = {
        target: float(pd.to_numeric(flow[target], errors="coerce").quantile(0.90))
        for target in FLOW_TARGETS
    }

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Scenarios: {', '.join(SCENARIOS)}")
    print("Event-focused cutoffs:")
    print(cutoffs[["cutoff", "date", "event_type"]].to_string(index=False))
    print("90th percentile high-congestion thresholds:", high_thresholds)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames = []
    for cutoff_row in cutoffs.itertuples(index=False):
        cutoff = pd.Timestamp(cutoff_row.cutoff)
        actual = actuals_with_labels(
            flow,
            calendar,
            cutoff=cutoff,
            horizon=args.horizon,
            high_thresholds=high_thresholds,
        )
        for scenario in SCENARIOS:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future = scenario_frames(
                flow,
                calendar,
                scenario=scenario,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
            )
            forecast = run_forecast(pipeline, history, future, horizon=args.horizon)
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            joined["cutoff"] = cutoff
            joined["event_type"] = cutoff_row.event_type
            joined["scenario"] = scenario
            joined["horizon_hour"] = (
                (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
            ).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    summary = summarize_segments(detail)
    horizon_summary = summarize_horizon(detail)
    winners = (
        summary.sort_values("mae")
        .groupby(["segment", "target_name"], as_index=False)
        .first()[
            [
                "segment",
                "target_name",
                "scenario",
                "n",
                "mae",
                "mae_improvement_vs_baseline_pct",
                "mae_improvement_vs_demand_pct",
            ]
        ]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs.to_csv(args.output_dir / "hourly_calendar_cutoffs.csv", index=False)
    detail.to_csv(args.output_dir / "hourly_calendar_detail.csv", index=False)
    summary.to_csv(args.output_dir / "hourly_calendar_summary.csv", index=False)
    horizon_summary.to_csv(args.output_dir / "hourly_calendar_by_horizon.csv", index=False)
    winners.to_csv(args.output_dir / "hourly_calendar_winners.csv", index=False)

    print("\nAll-hours summary:")
    print(summary.loc[summary["segment"].eq("all_hours")].to_string(index=False))
    print("\nJGH-only summary:")
    print(summary.loc[summary["segment"].eq("jgh_only_ramq")].to_string(index=False))
    print("\nHigh-congestion summary:")
    print(summary.loc[summary["segment"].eq("high_congestion")].to_string(index=False))


if __name__ == "__main__":
    main()
