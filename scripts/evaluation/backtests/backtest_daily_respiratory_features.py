#!/usr/bin/env python3
"""Native Chronos-2 ablation of Montréal respiratory surveillance for daily ED visits.

All scenarios use the *current production daily base*: calendar/closure features plus the
validated raw-plus-snow weather representation.  Respiratory scenarios then add either
raw weekly activity or raw activity + trend features, so the experiment measures the
incremental value of surveillance rather than rediscovering calendar/weather signal.

Publication timing is enforced at every cutoff.  A historical forecast may use only
INSPQ reports whose ``available_date`` is on/before that cutoff.  The latest published
respiratory state is carried through D+1..D+7; reports that were published during the
historical forecast horizon are deliberately hidden.

Weather has the same retrospective caveat as the existing weather ablations: historical
future weather is realized/revised rather than an archive of exact forecast snapshots.
Because every scenario here receives the identical weather frame, the respiratory
comparison itself remains apples-to-apples within that limitation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_daily_weather_features as weather_bt
from backtest_daily_holiday_targeted import CALENDAR_CLOSURE_COLUMNS, engineer_targeted_features
from backtest_holiday_features import (
    FLOW_URL,
    MAX_CONTEXT_DAYS,
    MODEL_ID,
    SERIES_ID,
    TARGET,
    contiguous_history,
    load_daily_visits,
)
from daily_weather_feature_set import RAW_PLUS_SNOW_COLUMNS
from respiratory_surveillance import (
    RESPIRATORY_FEATURE_COLUMNS,
    RESPIRATORY_RAW_COLUMNS,
    expand_to_daily,
    load_surveillance_csv,
)

SCENARIOS = ["production_base", "respiratory_raw", "respiratory_trends"]


def _calendar(frame: pd.DataFrame) -> pd.DataFrame:
    featured = engineer_targeted_features(frame)
    keep = list(frame.columns)
    for column in CALENDAR_CLOSURE_COLUMNS:
        if column not in keep:
            keep.append(column)
    return featured[keep]


def select_cutoffs(
    daily: pd.DataFrame,
    daily_weather: pd.DataFrame,
    respiratory: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
    min_history_days: int,
    num_cutoffs: int,
) -> list[pd.Timestamp]:
    first_report = pd.Timestamp(respiratory["available_date"].min()).normalize()
    common_start = max(
        daily["ds"].min(),
        daily_weather["ds"].min(),
        first_report,
    ) + pd.Timedelta(days=min_history_days)
    common_end = min(daily["ds"].max(), daily_weather["ds"].max()) - pd.Timedelta(days=horizon_days)
    if common_end < common_start:
        raise ValueError(
            f"No respiratory backtest window with {min_history_days}d common history: "
            f"{common_start.date()}..{common_end.date()}"
        )

    candidates = pd.date_range(common_start, common_end, freq="D")
    if len(candidates) > num_cutoffs:
        positions = np.linspace(0, len(candidates) - 1, num=num_cutoffs)
        candidates = pd.DatetimeIndex([candidates[int(round(position))] for position in positions])

    cutoffs: list[pd.Timestamp] = []
    for cutoff in sorted(set(pd.Timestamp(value) for value in candidates)):
        try:
            history = contiguous_history(
                daily,
                cutoff,
                context_days=context_days,
                min_history_days=min_history_days,
            )
        except ValueError:
            continue
        history = history.loc[history["ds"] >= first_report]
        future_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        actual = daily.loc[daily["ds"].isin(future_dates), TARGET]
        weather_future = daily_weather.loc[daily_weather["ds"].isin(future_dates), "ds"]
        if len(history) < min_history_days or len(actual) != horizon_days or actual.isna().any():
            continue
        if len(weather_future) != horizon_days:
            continue
        if respiratory.loc[respiratory["available_date"].le(cutoff)].empty:
            continue
        cutoffs.append(cutoff)
    if not cutoffs:
        raise ValueError("No eligible respiratory-surveillance backtest cutoffs")
    return cutoffs


def _fill_numeric_covariates(
    history: pd.DataFrame,
    future: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.copy()
    future = future.copy()
    for column in columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
        future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")
        history[column] = history[column].ffill()
        fallback = history[column].median(skipna=True)
        if not np.isfinite(fallback):
            fallback = 0.0
        history[column] = history[column].fillna(float(fallback))
        if future[column].isna().any():
            future[column] = future[column].ffill().fillna(float(history[column].iloc[-1]))
        if future[column].isna().any():
            raise ValueError(f"Unable to fill future respiratory covariate {column}")
    return history, future


def build_frames(
    daily: pd.DataFrame,
    daily_weather: pd.DataFrame,
    respiratory: pd.DataFrame,
    *,
    scenario: str,
    cutoff: pd.Timestamp,
    horizon_days: int,
    context_days: int,
    min_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_report = pd.Timestamp(respiratory["available_date"].min()).normalize()
    history = contiguous_history(
        daily,
        cutoff,
        context_days=context_days,
        min_history_days=min_history_days,
    )
    # Equal history length for every scenario in this experiment.
    history = history.loc[history["ds"] >= first_report].copy()
    if len(history) < min_history_days:
        raise ValueError(f"Only {len(history)} common respiratory-era history days at {cutoff.date()}")
    future = pd.DataFrame(
        {"ds": pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")}
    )

    history = _calendar(history)
    future = _calendar(future)

    featured_weather = weather_bt.weather_at_cutoff(
        daily_weather,
        cutoff=cutoff,
        future_end=future["ds"].max(),
    )
    history = history.merge(featured_weather[["ds", *RAW_PLUS_SNOW_COLUMNS]], on="ds", how="left")
    future = future.merge(featured_weather[["ds", *RAW_PLUS_SNOW_COLUMNS]], on="ds", how="left")
    history, future = weather_bt._fill_covariates(history, future, RAW_PLUS_SNOW_COLUMNS)

    respiratory_columns: list[str] = []
    if scenario != "production_base":
        available = respiratory.loc[respiratory["available_date"].le(cutoff)].copy()
        # Filtering before expansion is the critical anti-leakage guard: a report that
        # happened to be published on D+3 of this historical horizon cannot enter D+3.
        respiratory_daily = expand_to_daily(
            available,
            start=history["ds"].min(),
            end=future["ds"].max(),
        )
        respiratory_columns = (
            RESPIRATORY_RAW_COLUMNS
            if scenario == "respiratory_raw"
            else RESPIRATORY_FEATURE_COLUMNS
        )
        history = history.merge(
            respiratory_daily[["ds", *respiratory_columns]], on="ds", how="left"
        )
        future = future.merge(
            respiratory_daily[["ds", *respiratory_columns]], on="ds", how="left"
        )
        history, future = _fill_numeric_covariates(history, future, respiratory_columns)

    calendar_columns = [column for column in CALENDAR_CLOSURE_COLUMNS if column in history]
    for column in calendar_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0).astype("float64")
        future[column] = pd.to_numeric(future[column], errors="coerce").fillna(0.0).astype("float64")

    covariates = list(dict.fromkeys([*calendar_columns, *RAW_PLUS_SNOW_COLUMNS, *respiratory_columns]))
    history["id"] = SERIES_ID
    future["id"] = SERIES_ID
    return (
        history[["id", "ds", TARGET, *covariates]],
        future[["id", "ds", *covariates]],
    )


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
) -> pd.DataFrame:
    result = pipeline.predict_df(
        history,
        prediction_length=horizon_days,
        future_df=future,
        id_column="id",
        timestamp_column="ds",
        target=[TARGET],
        quantile_levels=[0.5],
        context_length=min(context_days, MAX_CONTEXT_DAYS, len(history)),
    )
    return result[["ds", "target_name", "predictions"]].rename(columns={"predictions": "prediction"})


def summarize(detail: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    table = detail.groupby(keys, as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)
    control_keys = keys[:-1]
    control = table.loc[table["scenario"].eq("production_base"), [*control_keys, "mae"]].rename(
        columns={"mae": "production_base_mae"}
    )
    table = table.merge(control, on=control_keys, how="left")
    table["mae_improvement_vs_production_pct"] = (
        (table["production_base_mae"] - table["mae"])
        / table["production_base_mae"].replace(0, np.nan)
        * 100
    )
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--respiratory-csv", required=True)
    parser.add_argument("--weather-url", default=weather_bt.WEATHER_URL)
    parser.add_argument("--flow-url", default=FLOW_URL)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--num-cutoffs", type=int, default=16)
    parser.add_argument("--context-days", type=int, default=1095)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-respiratory"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = load_daily_visits(args.flow_url)
    daily_weather = weather_bt.load_daily_weather(args.weather_url)
    respiratory = load_surveillance_csv(args.respiratory_csv)
    cutoffs = select_cutoffs(
        daily,
        daily_weather,
        respiratory,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
        num_cutoffs=args.num_cutoffs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "cutoffs.csv", index=False)
    print(
        f"Respiratory reports: {respiratory['available_date'].min().date()}.."
        f"{respiratory['available_date'].max().date()} ({len(respiratory)} reports)"
    )
    print(f"Cutoffs: {cutoffs}")
    print("Control: current production calendar/closure + raw_plus_snow weather.")
    print("Leakage guard: respiratory reports are frozen at each historical cutoff.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)
    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=args.horizon_days, freq="D")
        actual = daily.loc[daily["ds"].isin(dates), ["ds", TARGET]].rename(columns={TARGET: "actual"})
        actual["target_name"] = TARGET
        for scenario in SCENARIOS:
            print(f"Forecasting cutoff={cutoff.date()} scenario={scenario}")
            history, future = build_frames(
                daily,
                daily_weather,
                respiratory,
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
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_day"] = ((joined["ds"] - cutoff) / pd.Timedelta(days=1)).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(args.output_dir / "detail.csv", index=False)
    summary = summarize(detail, ["target_name", "scenario"]).sort_values("mae")
    by_horizon = summarize(detail, ["target_name", "horizon_day", "scenario"]).sort_values(
        ["horizon_day", "mae"]
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    by_horizon.to_csv(args.output_dir / "by_horizon_day.csv", index=False)
    print("\n=== Respiratory surveillance summary ===")
    print(summary.to_string(index=False))
    print("\n=== By horizon day ===")
    print(by_horizon.to_string(index=False))


if __name__ == "__main__":
    main()
