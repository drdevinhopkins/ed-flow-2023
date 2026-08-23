#!/usr/bin/env python3
"""Holiday-focused native Chronos-2 ablation for JGH daily ED visits.

The target is total ED arrivals per Montreal calendar day, calculated as the sum of
hourly ``Inflow_Total``. This replaces the legacy notebook's shifted end-of-day
cumulative-value construction with a direct daily total.

Incomplete source days are never imputed. They remain missing in the daily series and
act as hard boundaries: a backtest window may only use the contiguous run of complete
days after the most recent incomplete day. Forecast horizons containing an incomplete
day are not eligible for scoring.

The backtest deliberately samples cutoffs immediately before holiday or holiday-adjacent
"shoulder" days. A generic rolling backtest contains relatively few holidays and is
poorly powered to decide which holiday covariates help. Each scenario uses native
``Chronos2Pipeline.predict_df`` with deterministic known-future covariates; AutoGluon is
not involved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from holiday_features import add_holiday_features

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
MODEL_ID = "amazon/chronos-2"
TARGET = "daily_visits"
SERIES_ID = "jgh_daily"
SCENARIOS = ["baseline", "legacy", "calendars", "shoulders", "rich", "closures"]
MAX_CONTEXT_DAYS = 8192

EVENT_COLUMNS = [
    "is_any_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_long_weekend_edge",
    "is_major_jewish_holiday_eve",
    "is_christmas_newyear_period",
    "is_quebec_canada_day_period",
    "is_rebound_after_long_closure",
    "is_pre_long_closure",
]


def parse_ds(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("America/Montreal").dt.tz_localize(None)
    return parsed


def load_daily_visits(flow_url: str = FLOW_URL) -> pd.DataFrame:
    """Load hourly inflow and aggregate complete Montreal calendar days.

    A valid day has 23, 24, or 25 hourly rows to permit DST transitions, and every row
    must contain a numeric ``Inflow_Total``. Incomplete internal days are retained on the
    daily calendar with a missing target; they are never filled or interpolated. Leading
    and trailing partial days outside the first/last complete day are discarded.
    """
    raw = pd.read_csv(flow_url, usecols=["ds", "Inflow_Total"])

    # De-duplicate source timestamps before converting to local wall-clock. Offset-aware
    # timestamps around the autumn DST transition remain distinct at this stage.
    raw = raw.drop_duplicates("ds", keep="last")
    raw["ds"] = parse_ds(raw["ds"])
    raw["Inflow_Total"] = pd.to_numeric(raw["Inflow_Total"], errors="coerce")
    raw = raw.dropna(subset=["ds"]).sort_values("ds")
    raw["day"] = raw["ds"].dt.normalize()

    grouped = raw.groupby("day", as_index=False).agg(
        daily_visits=("Inflow_Total", lambda values: values.sum(min_count=1)),
        observed_rows=("ds", "size"),
        numeric_rows=("Inflow_Total", "count"),
    )
    grouped = grouped.rename(columns={"day": "ds"}).sort_values("ds").reset_index(drop=True)
    grouped["is_complete"] = (
        grouped["observed_rows"].between(23, 25)
        & grouped["numeric_rows"].eq(grouped["observed_rows"])
        & grouped[TARGET].notna()
    )

    complete = grouped.loc[grouped["is_complete"]]
    if complete.empty:
        raise ValueError("No complete daily inflow observations found")

    first_complete = complete["ds"].min()
    last_complete = complete["ds"].max()
    index = pd.date_range(first_complete, last_complete, freq="D", name="ds")
    daily = grouped.set_index("ds").reindex(index).reset_index()
    daily["observed_rows"] = daily["observed_rows"].fillna(0).astype(int)
    daily["numeric_rows"] = daily["numeric_rows"].fillna(0).astype(int)
    daily["is_complete"] = daily["is_complete"].fillna(False).astype(bool)
    daily[TARGET] = pd.to_numeric(daily[TARGET], errors="coerce").where(daily["is_complete"])
    daily[TARGET] = daily[TARGET].astype("float64")
    return daily[["ds", TARGET, "observed_rows", "numeric_rows", "is_complete"]]


def contiguous_history(
    daily: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    context_days: int,
    min_history_days: int,
) -> pd.DataFrame:
    """Return the complete daily run ending at cutoff, bounded by the latest data gap."""
    available = daily.loc[daily["ds"] <= cutoff, ["ds", TARGET]].copy()
    if available.empty or available["ds"].iloc[-1] != cutoff:
        raise ValueError(f"Cutoff {cutoff.date()} is not present in the daily calendar")
    if pd.isna(available[TARGET].iloc[-1]):
        raise ValueError(f"Cutoff {cutoff.date()} is incomplete")

    valid = available[TARGET].notna().to_numpy()
    invalid_positions = np.flatnonzero(~valid)
    start_position = int(invalid_positions[-1] + 1) if len(invalid_positions) else 0
    history = available.iloc[start_position:].tail(min(context_days, MAX_CONTEXT_DAYS)).copy()

    if len(history) < min_history_days:
        raise ValueError(
            f"Only {len(history)} contiguous complete days before cutoff {cutoff.date()}; "
            f"need {min_history_days}"
        )
    if history[TARGET].isna().any():
        raise ValueError(f"Unexpected missing target within history ending {cutoff.date()}")
    return history.reset_index(drop=True)


def _event_type(row: pd.Series) -> str:
    labels = [column.removeprefix("is_") for column in EVENT_COLUMNS if int(row.get(column, 0))]
    return "|".join(labels) if labels else "none"


def add_event_labels(frame: pd.DataFrame) -> pd.DataFrame:
    featured = add_holiday_features(frame, feature_set="closures")
    featured["is_event_day"] = featured[EVENT_COLUMNS].max(axis=1).astype(np.int8)
    featured["event_type"] = featured.apply(_event_type, axis=1)
    return featured


def select_holiday_cutoffs(
    daily: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
    min_history_days: int,
    num_cutoffs: int,
) -> pd.DataFrame:
    """Choose diverse holiday/shoulder events with clean history and actual horizons."""
    labelled = add_event_labels(daily[["ds"]].copy())
    candidates = labelled.loc[labelled["is_event_day"].astype(bool)].copy()
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
        raise ValueError(
            "No holiday-focused cutoffs have both a complete forecast horizon and the "
            f"required {min_history_days} contiguous history days"
        )

    eligible = pd.DataFrame(eligible_rows).sort_values("cutoff").reset_index(drop=True)
    if len(eligible) > num_cutoffs:
        # Spread selections across the full history rather than clustering in one holiday
        # season or year.
        positions = np.linspace(0, len(eligible) - 1, num=num_cutoffs)
        indices = sorted(set(int(round(value)) for value in positions))
        eligible = eligible.iloc[indices].copy()

    return eligible[["cutoff", "ds", "event_type", "history_days", *EVENT_COLUMNS]].reset_index(
        drop=True
    )


def _legacy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the old coarse Quebec/Jewish yes/no representation."""
    featured = add_holiday_features(frame, feature_set="calendars")
    featured["is_qc_holiday"] = featured["is_qc_holiday"].map({1: "yes", 0: "no"})
    featured["is_jewish_holiday"] = featured["is_jewish_holiday"].map({1: "yes", 0: "no"})
    keep = list(frame.columns) + ["is_qc_holiday", "is_jewish_holiday"]
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

    if scenario == "legacy":
        history = _legacy_features(history)
        future = _legacy_features(future)
    elif scenario != "baseline":
        history = add_holiday_features(history, feature_set=scenario)
        future = add_holiday_features(future, feature_set=scenario)

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
        if pd.api.types.is_numeric_dtype(history[column]) or pd.api.types.is_numeric_dtype(
            future[column]
        ):
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
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
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
    actual = add_event_labels(actual)
    actual = actual.rename(columns={TARGET: "actual"})
    actual["target_name"] = TARGET
    return actual


def _metric_table(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
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


def _add_baseline_improvement(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
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
    segments: list[pd.DataFrame] = []
    for name, mask in (
        ("all_days", pd.Series(True, index=detail.index)),
        ("holiday_window", detail["is_event_day"].astype(bool)),
        ("ordinary_days", ~detail["is_event_day"].astype(bool)),
    ):
        table = _metric_table(detail.loc[mask], ["target_name", "scenario"])
        if not table.empty:
            table.insert(0, "segment", name)
            segments.append(table)

    summary = pd.concat(segments, ignore_index=True)
    summary = _add_baseline_improvement(summary, ["segment", "target_name"])
    return summary.sort_values(["segment", "mae"]).reset_index(drop=True)


def summarize_by_event(detail: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for event_column in EVENT_COLUMNS:
        subset = detail.loc[detail[event_column].astype(bool)].copy()
        if subset.empty:
            continue
        table = _metric_table(subset, ["target_name", "scenario"])
        table.insert(0, "event", event_column.removeprefix("is_"))
        frames.append(table)

    if not frames:
        return pd.DataFrame()
    by_event = pd.concat(frames, ignore_index=True)
    by_event = _add_baseline_improvement(by_event, ["event", "target_name"])
    return by_event.sort_values(["event", "mae"]).reset_index(drop=True)


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
    if args.horizon_days < 1 or args.num_cutoffs < 1 or args.context_days < 28:
        raise ValueError("horizon/cutoffs must be positive and context-days must be >= 28")
    if args.min_history_days < 28 or args.min_history_days > args.context_days:
        raise ValueError("min-history-days must be >= 28 and <= context-days")

    daily = load_daily_visits(args.flow_url)
    incomplete = daily.loc[~daily["is_complete"]]
    print(
        f"Daily visits calendar: {daily['ds'].min().date()} to {daily['ds'].max().date()} "
        f"({len(daily)} days; {len(incomplete)} incomplete days excluded from modeling)"
    )
    if not incomplete.empty:
        print("Incomplete-day examples (not imputed):")
        print(incomplete[["ds", "observed_rows", "numeric_rows"]].head(15).to_string(index=False))

    cutoffs = select_holiday_cutoffs(
        daily,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
        num_cutoffs=args.num_cutoffs,
    )

    print(f"Scenarios: {', '.join(args.scenarios)}")
    print("Holiday-focused daily cutoffs:")
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
            joined["cutoff_event_type"] = cutoff_row.event_type
            joined["history_days"] = cutoff_row.history_days
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
    detail.to_csv(args.output_dir / "daily_visit_holiday_backtest_detail.csv", index=False)
    summary.to_csv(args.output_dir / "daily_visit_holiday_backtest_summary.csv", index=False)
    by_event.to_csv(args.output_dir / "daily_visit_holiday_backtest_by_event.csv", index=False)
    winners.to_csv(args.output_dir / "daily_visit_holiday_backtest_winners.csv", index=False)
    cutoffs.to_csv(args.output_dir / "daily_visit_holiday_backtest_cutoffs.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nWinners by evaluation segment:")
    print(
        winners[
            ["segment", "scenario", "mae", "baseline_mae", "mae_improvement_pct"]
        ].to_string(index=False)
    )
    if not by_event.empty:
        print("\nPerformance by holiday/shoulder type:")
        print(by_event.to_string(index=False))


if __name__ == "__main__":
    main()
