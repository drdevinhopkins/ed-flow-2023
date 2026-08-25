#!/usr/bin/env python3
"""Native Chronos-2 ablation of engineered weather features for JGH daily ED visits.

This experiment starts from the targeted ``calendar_closure`` holiday feature set and
asks whether weather is more useful when represented as context/events rather than raw
meteorological measurements.  Scenarios progressively add:

* raw_weather: daily summaries of the existing hourly Open-Meteo table;
* anomalies: weather relative to a cutoff-fitted seasonal climatology;
* shocks: abrupt temperature/pressure/wind changes;
* lagged: accumulated exposure and post-snowstorm recovery features;
* compound: cold+wind, snow+wind, freeze/thaw, refreeze, thermal/storm indices;
* all_weather: the complete engineered weather set.

Important validation limitation: the repository's weather.csv is not an archive of the
exact forecast snapshot available at each historical cutoff.  Historical future weather
therefore represents realized/revised weather signal and can make weather performance
optimistic.  The seasonal climatology itself is fit only on rows at or before each cutoff.
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
from weather_features import (
    ALL_ENGINEERED_COLUMNS,
    ANOMALY_COLUMNS,
    COMPOUND_COLUMNS,
    LAGGED_COLUMNS,
    RAW_DAILY_COLUMNS,
    SHOCK_COLUMNS,
    add_weather_features,
    aggregate_hourly_weather,
    fit_climatology,
)

WEATHER_URL = (
    "https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/"
    "weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1"
)

SCENARIOS = [
    "baseline",
    "calendar_closure",
    "raw_weather",
    "anomalies",
    "shocks",
    "lagged",
    "compound",
    "all_weather",
]

WEATHER_EVENT_COLUMNS = [
    "cold_anomaly_event",
    "warm_anomaly_event",
    "windy_anomaly_event",
    "major_snow_event",
    "heavy_precip_event",
    "freeze_thaw_day",
    "post_snow_thaw",
    "refreeze_after_thaw",
    "cold_windy_event",
    "snow_wind_event",
    "day_after_major_snow",
    "two_days_after_major_snow",
    "three_days_after_major_snow",
]


def load_daily_weather(url: str = WEATHER_URL) -> pd.DataFrame:
    hourly = pd.read_csv(url)
    daily = aggregate_hourly_weather(hourly)
    daily = daily.loc[daily["temp_mean"].notna()].copy()
    daily = daily.sort_values("ds").drop_duplicates("ds", keep="last").reset_index(drop=True)
    if daily.empty:
        raise ValueError("Weather source produced no daily rows")
    return daily


def _weather_columns_for_scenario(scenario: str) -> list[str]:
    if scenario == "raw_weather":
        return RAW_DAILY_COLUMNS
    if scenario == "anomalies":
        return ANOMALY_COLUMNS
    if scenario == "shocks":
        return [*ANOMALY_COLUMNS, *SHOCK_COLUMNS]
    if scenario == "lagged":
        return [*ANOMALY_COLUMNS, *SHOCK_COLUMNS, *LAGGED_COLUMNS]
    if scenario == "compound":
        return [*ANOMALY_COLUMNS, *COMPOUND_COLUMNS]
    if scenario == "all_weather":
        return [*RAW_DAILY_COLUMNS, *ALL_ENGINEERED_COLUMNS]
    if scenario in {"baseline", "calendar_closure"}:
        return []
    raise ValueError(f"Unknown scenario: {scenario}")


def _calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    featured = engineer_targeted_features(frame)
    keep = list(frame.columns)
    for column in CALENDAR_CLOSURE_COLUMNS:
        if column not in keep:
            keep.append(column)
    return featured[keep]


def weather_at_cutoff(
    daily_weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    future_end: pd.Timestamp,
) -> pd.DataFrame:
    history_weather = daily_weather.loc[daily_weather["ds"] <= cutoff].copy()
    climatology = fit_climatology(history_weather)
    through_horizon = daily_weather.loc[daily_weather["ds"] <= future_end].copy()
    featured = add_weather_features(through_horizon, climatology)
    return featured


def _fill_covariates(
    history: pd.DataFrame,
    future: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill only covariate gaps; never fill or alter the target."""
    history = history.copy()
    future = future.copy()
    for column in columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
        future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")

        # Lags/differences at the first available weather rows have no predecessor and
        # are naturally neutral.  Other sparse historical values use prior information
        # then a history median fallback.
        if column in SHOCK_COLUMNS or column in LAGGED_COLUMNS:
            history[column] = history[column].fillna(0.0)
        else:
            history[column] = history[column].ffill()
            fallback = history[column].median(skipna=True)
            if not np.isfinite(fallback):
                fallback = 0.0
            history[column] = history[column].fillna(float(fallback))

        if future[column].isna().any():
            # For the exploratory historical replay, a missing future raw field should
            # fail rather than silently invent forecast weather.
            missing_dates = future.loc[future[column].isna(), "ds"].dt.date.tolist()[:5]
            raise ValueError(f"Missing future weather covariate {column} at {missing_dates}")
    return history, future


def build_frames(
    daily: pd.DataFrame,
    daily_weather: pd.DataFrame,
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
    # All scenarios use the same target-history span so weather comparisons are fair.
    history = history.loc[history["ds"] >= daily_weather["ds"].min()].copy()
    if len(history) < min_history_days:
        raise ValueError(
            f"Only {len(history)} target days overlap weather at cutoff {cutoff.date()}"
        )

    future = pd.DataFrame(
        {"ds": pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")}
    )

    if scenario != "baseline":
        history = _calendar_features(history)
        future = _calendar_features(future)

    weather_columns = _weather_columns_for_scenario(scenario)
    if weather_columns:
        future_end = future["ds"].max()
        featured_weather = weather_at_cutoff(
            daily_weather,
            cutoff=cutoff,
            future_end=future_end,
        )
        available_columns = ["ds", *weather_columns]
        history = history.merge(featured_weather[available_columns], on="ds", how="left")
        future = future.merge(featured_weather[available_columns], on="ds", how="left")
        history, future = _fill_covariates(history, future, weather_columns)

    history["id"] = SERIES_ID
    if scenario == "baseline":
        return history[["id", "ds", TARGET]], None

    future["id"] = SERIES_ID
    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    history, future = _fill_covariates(history, future, covariates)
    return (
        history[["id", "ds", TARGET, *covariates]],
        future[["id", "ds", *covariates]],
    )


def _raw_event_score(weather: pd.DataFrame) -> pd.Series:
    temp_change = pd.to_numeric(weather["temp_mean"], errors="coerce").diff().abs()
    snow = pd.to_numeric(weather["snowfall_sum"], errors="coerce").fillna(0)
    precip = pd.to_numeric(weather["precip_sum"], errors="coerce").fillna(0)
    gust = pd.to_numeric(weather["gust_max"], errors="coerce").fillna(0)
    temp_min = pd.to_numeric(weather["temp_min"], errors="coerce")
    temp_max = pd.to_numeric(weather["temp_max"], errors="coerce")
    freeze_thaw = ((temp_min < 0) & (temp_max > 0)).astype(float)
    return (
        (snow / 5.0).clip(lower=0)
        + (precip / 15.0).clip(lower=0)
        + (gust / 45.0).clip(lower=0)
        + (temp_change / 7.0).clip(lower=0)
        + freeze_thaw
    )


def select_cutoffs(
    daily: pd.DataFrame,
    daily_weather: pd.DataFrame,
    *,
    horizon_days: int,
    context_days: int,
    min_history_days: int,
    num_cutoffs: int,
) -> pd.DataFrame:
    common_start = max(
        daily["ds"].min() + pd.Timedelta(days=min_history_days),
        daily_weather["ds"].min() + pd.Timedelta(days=min_history_days),
    )
    common_end = min(
        daily["ds"].max() - pd.Timedelta(days=horizon_days),
        daily_weather["ds"].max() - pd.Timedelta(days=horizon_days),
    )
    if common_end < common_start:
        raise ValueError(f"No eligible daily/weather overlap: {common_start} to {common_end}")

    eligible_weather = daily_weather.loc[
        (daily_weather["ds"] >= common_start + pd.Timedelta(days=1))
        & (daily_weather["ds"] <= common_end + pd.Timedelta(days=1))
    ].copy()
    eligible_weather["event_score"] = _raw_event_score(eligible_weather)
    eligible_weather["cutoff"] = eligible_weather["ds"] - pd.Timedelta(days=1)

    # Half event-rich windows, half evenly spaced general windows.  Event cutoffs are
    # separated so one storm does not dominate the experiment with adjacent horizons.
    n_event = num_cutoffs // 2
    n_regular = num_cutoffs - n_event
    selected_events: list[pd.Timestamp] = []
    for row in eligible_weather.sort_values("event_score", ascending=False).itertuples(index=False):
        cutoff = pd.Timestamp(row.cutoff)
        if all(abs((cutoff - other).days) >= horizon_days for other in selected_events):
            selected_events.append(cutoff)
        if len(selected_events) >= n_event:
            break

    regular_candidates = pd.date_range(common_start, common_end, freq="D")
    regular: list[pd.Timestamp] = []
    if len(regular_candidates) and n_regular:
        positions = np.linspace(0, len(regular_candidates) - 1, num=n_regular)
        regular = [pd.Timestamp(regular_candidates[int(round(pos))]) for pos in positions]

    cutoffs = sorted(set([*selected_events, *regular]))
    if len(cutoffs) > num_cutoffs:
        cutoffs = cutoffs[:num_cutoffs]

    records = []
    score_lookup = eligible_weather.set_index("cutoff")["event_score"]
    for cutoff in cutoffs:
        try:
            history = contiguous_history(
                daily,
                cutoff,
                context_days=context_days,
                min_history_days=min_history_days,
            )
        except ValueError:
            continue
        history = history.loc[history["ds"] >= daily_weather["ds"].min()]
        future_dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")
        actual = daily.loc[daily["ds"].isin(future_dates), TARGET]
        weather_future = daily_weather.loc[daily_weather["ds"].isin(future_dates), "ds"]
        if len(history) < min_history_days or len(actual) != horizon_days or actual.isna().any():
            continue
        if len(weather_future) != horizon_days:
            continue
        records.append(
            {
                "cutoff": cutoff,
                "history_days": len(history),
                "event_score_next_day": float(score_lookup.get(cutoff, np.nan)),
                "selection": "event" if cutoff in selected_events else "regular",
            }
        )
    if not records:
        raise ValueError("No eligible weather backtest cutoffs")
    return pd.DataFrame(records).sort_values("cutoff").reset_index(drop=True)


def actuals_with_weather_labels(
    daily: pd.DataFrame,
    daily_weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    actual = daily.loc[daily["ds"].isin(dates), ["ds", TARGET]].copy()
    featured = weather_at_cutoff(
        daily_weather,
        cutoff=cutoff,
        future_end=dates.max(),
    )
    labels = featured[["ds", *WEATHER_EVENT_COLUMNS]].copy()
    actual = actual.merge(labels, on="ds", how="left")
    actual[WEATHER_EVENT_COLUMNS] = actual[WEATHER_EVENT_COLUMNS].fillna(0).astype(np.int8)
    actual["is_weather_event_day"] = actual[WEATHER_EVENT_COLUMNS].max(axis=1).astype(np.int8)
    actual = actual.rename(columns={TARGET: "actual"})
    actual["target_name"] = TARGET
    return actual


def add_calendar_control_improvement(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
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
    segments = [
        ("all_days", pd.Series(True, index=detail.index)),
        ("weather_event_days", detail["is_weather_event_day"].astype(bool)),
        ("non_event_days", ~detail["is_weather_event_day"].astype(bool)),
        (
            "post_major_snow_1_3d",
            detail[[
                "day_after_major_snow",
                "two_days_after_major_snow",
                "three_days_after_major_snow",
            ]].max(axis=1).astype(bool),
        ),
        ("cold_windy_days", detail["cold_windy_event"].astype(bool)),
        ("freeze_thaw_days", detail["freeze_thaw_day"].astype(bool)),
    ]
    frames: list[pd.DataFrame] = []
    for segment, mask in segments:
        if not mask.any():
            continue
        table = metric_table(detail.loc[mask], ["target_name", "scenario"])
        table.insert(0, "segment", segment)
        frames.append(table)
    result = pd.concat(frames, ignore_index=True)
    result = add_baseline_improvement(result, ["segment", "target_name"])
    result = add_calendar_control_improvement(result, ["segment", "target_name"])
    result["weather_validation_mode"] = "realized/revised weather; not archived forecast snapshots"
    return result.sort_values(["segment", "mae"]).reset_index(drop=True)


def summarize_by_event(detail: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for event in WEATHER_EVENT_COLUMNS:
        subset = detail.loc[detail[event].astype(bool)]
        if subset.empty:
            continue
        table = metric_table(subset, ["target_name", "scenario"])
        table.insert(0, "event", event)
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = add_baseline_improvement(result, ["event", "target_name"])
    result = add_calendar_control_improvement(result, ["event", "target_name"])
    result["weather_validation_mode"] = "realized/revised weather; not archived forecast snapshots"
    return result.sort_values(["event", "mae"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--num-cutoffs", type=int, default=16)
    parser.add_argument("--context-days", type=int, default=1095)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-weather"))
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--flow-url", default=FLOW_URL)
    parser.add_argument("--weather-url", default=WEATHER_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily = load_daily_visits(args.flow_url)
    daily_weather = load_daily_weather(args.weather_url)
    cutoffs = select_cutoffs(
        daily,
        daily_weather,
        horizon_days=args.horizon_days,
        context_days=args.context_days,
        min_history_days=args.min_history_days,
        num_cutoffs=args.num_cutoffs,
    )

    print(f"Daily target range: {daily['ds'].min().date()} to {daily['ds'].max().date()}")
    print(
        f"Daily weather range: {daily_weather['ds'].min().date()} to "
        f"{daily_weather['ds'].max().date()}"
    )
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print("WARNING: weather.csv is not archived forecast snapshots; weather results are signal-potential estimates.")
    print("Cutoffs:")
    print(cutoffs.to_string(index=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id,
        device_map=device,
    )

    frames: list[pd.DataFrame] = []
    for cutoff_row in cutoffs.itertuples(index=False):
        cutoff = pd.Timestamp(cutoff_row.cutoff)
        actual = actuals_with_weather_labels(
            daily,
            daily_weather,
            cutoff=cutoff,
            horizon_days=args.horizon_days,
        )
        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff.date()} scenario={scenario}")
            history, future = build_frames(
                daily,
                daily_weather,
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
            joined["cutoff_selection"] = cutoff_row.selection
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
    by_horizon = metric_table(detail, ["horizon_day", "target_name", "scenario"])
    by_horizon = add_calendar_control_improvement(by_horizon, ["horizon_day", "target_name"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cutoffs.to_csv(args.output_dir / "daily_visit_weather_cutoffs.csv", index=False)
    detail.to_csv(args.output_dir / "daily_visit_weather_detail.csv", index=False)
    summary.to_csv(args.output_dir / "daily_visit_weather_summary.csv", index=False)
    by_event.to_csv(args.output_dir / "daily_visit_weather_by_event.csv", index=False)
    by_horizon.to_csv(args.output_dir / "daily_visit_weather_by_horizon.csv", index=False)
    winners.to_csv(args.output_dir / "daily_visit_weather_winners.csv", index=False)

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


if __name__ == "__main__":
    main()
