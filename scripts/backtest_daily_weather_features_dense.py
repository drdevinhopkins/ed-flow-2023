#!/usr/bin/env python3
"""Run the daily weather ablation with target-aware cutoff selection.

The base weather backtest originally sampled calendar/weather dates and only afterwards
checked whether the ED daily target had enough contiguous complete history. Because the
source inflow series intentionally treats incomplete days as hard boundaries, that could
collapse a requested 16-cutoff experiment to only a handful of scored windows.

This wrapper first enumerates every cutoff that is genuinely scoreable (clean target
history, complete target horizon, complete weather horizon), then draws a balanced mix
of severe-weather and representative cutoffs from that eligible pool. It reuses the
native Chronos-2 forecasting and feature logic unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest_daily_weather_features as base
from backtest_holiday_features import TARGET, contiguous_history


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

    weather_scored = daily_weather.copy().sort_values("ds")
    weather_scored["event_score"] = base._raw_event_score(weather_scored)
    score_lookup = weather_scored.set_index("ds")["event_score"]
    weather_dates = set(weather_scored["ds"])

    eligible: list[dict[str, object]] = []
    for cutoff in pd.date_range(common_start, common_end, freq="D"):
        future_dates = pd.date_range(
            cutoff + pd.Timedelta(days=1), periods=horizon_days, freq="D"
        )
        if not all(date in weather_dates for date in future_dates):
            continue

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
        history = history.loc[history["ds"] >= daily_weather["ds"].min()]
        if len(history) < min_history_days:
            continue

        next_day = cutoff + pd.Timedelta(days=1)
        score = float(score_lookup.get(next_day, np.nan))
        eligible.append(
            {
                "cutoff": pd.Timestamp(cutoff),
                "history_days": len(history),
                "event_score_next_day": score,
            }
        )

    pool = pd.DataFrame(eligible).sort_values("cutoff").reset_index(drop=True)
    if pool.empty:
        raise ValueError("No eligible weather backtest cutoffs")
    if len(pool) <= num_cutoffs:
        pool["selection"] = "eligible"
        return pool

    n_event = num_cutoffs // 2
    n_regular = num_cutoffs - n_event

    # Severe-weather cutoffs are greedily spaced by the forecast horizon so a single
    # multi-day storm does not occupy most of the event half of the experiment.
    event_rows: list[pd.Series] = []
    for _, row in pool.sort_values("event_score_next_day", ascending=False).iterrows():
        cutoff = pd.Timestamp(row["cutoff"])
        if pd.isna(row["event_score_next_day"]):
            continue
        if all(
            abs((cutoff - pd.Timestamp(selected["cutoff"])).days) >= horizon_days
            for selected in event_rows
        ):
            event_rows.append(row)
        if len(event_rows) >= n_event:
            break

    event_dates = {pd.Timestamp(row["cutoff"]) for row in event_rows}
    regular_pool = pool.loc[~pool["cutoff"].isin(event_dates)].sort_values("cutoff")
    regular_rows: list[pd.Series] = []
    if not regular_pool.empty and n_regular:
        positions = np.linspace(0, len(regular_pool) - 1, num=min(n_regular, len(regular_pool)))
        used_positions: set[int] = set()
        for pos in positions:
            idx = int(round(pos))
            if idx in used_positions:
                continue
            used_positions.add(idx)
            regular_rows.append(regular_pool.iloc[idx])

    selected_records: list[dict[str, object]] = []
    for row in event_rows:
        selected_records.append(
            {
                "cutoff": pd.Timestamp(row["cutoff"]),
                "history_days": int(row["history_days"]),
                "event_score_next_day": float(row["event_score_next_day"]),
                "selection": "event",
            }
        )
    for row in regular_rows:
        selected_records.append(
            {
                "cutoff": pd.Timestamp(row["cutoff"]),
                "history_days": int(row["history_days"]),
                "event_score_next_day": float(row["event_score_next_day"]),
                "selection": "regular",
            }
        )

    # If severe-weather spacing left unused slots, fill them from the most evenly spaced
    # remaining eligible dates rather than silently returning fewer requested cutoffs.
    selected_dates = {record["cutoff"] for record in selected_records}
    remaining = pool.loc[~pool["cutoff"].isin(selected_dates)].sort_values("cutoff")
    slots = num_cutoffs - len(selected_records)
    if slots > 0 and not remaining.empty:
        positions = np.linspace(0, len(remaining) - 1, num=min(slots, len(remaining)))
        used_positions: set[int] = set()
        for pos in positions:
            idx = int(round(pos))
            if idx in used_positions:
                continue
            used_positions.add(idx)
            row = remaining.iloc[idx]
            selected_records.append(
                {
                    "cutoff": pd.Timestamp(row["cutoff"]),
                    "history_days": int(row["history_days"]),
                    "event_score_next_day": float(row["event_score_next_day"]),
                    "selection": "fill",
                }
            )
            if len(selected_records) >= num_cutoffs:
                break

    return (
        pd.DataFrame(selected_records)
        .sort_values("cutoff")
        .head(num_cutoffs)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    base.select_cutoffs = select_cutoffs
    base.main()
