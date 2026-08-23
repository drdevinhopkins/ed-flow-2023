"""Montreal social-calendar covariates for JGH daily ED demand forecasting.

This module adds deterministic known-future context that is not well represented by a
smooth annual seasonal pattern:

* Quebec construction vacation, using published CCQ summer-vacation dates.
* Representative French-school (CSSDM) and English-school (EMSB) start dates.
* A Jewish-school calendar proxy with a separate back-to-school transition and major
  Jewish religious closure days.
* Spring, summer, and winter school-break proxies plus transition features.

School calendars vary by establishment (especially pedagogical days). The defaults below
are therefore intentionally system-level / representative features rather than claims that
every Montreal school is closed on the same dates. Callers may pass explicit school-start
overrides when a more specific catchment calendar is available.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Mapping

import numpy as np
import pandas as pd

from holiday_features import add_holiday_features


# Published CCQ mandatory summer construction-vacation periods. Keep these explicit: the
# dates move across Gregorian years, which is precisely why annual seasonality alone is a
# poor representation of the effect.
CONSTRUCTION_SUMMER_VACATIONS: dict[int, tuple[date, date]] = {
    2021: (date(2021, 7, 18), date(2021, 7, 31)),
    2022: (date(2022, 7, 24), date(2022, 8, 6)),
    2023: (date(2023, 7, 23), date(2023, 8, 5)),
    2024: (date(2024, 7, 21), date(2024, 8, 3)),
    2025: (date(2025, 7, 20), date(2025, 8, 2)),
    2026: (date(2026, 7, 19), date(2026, 8, 1)),
    2027: (date(2027, 7, 25), date(2027, 8, 7)),
    2028: (date(2028, 7, 23), date(2028, 8, 5)),
}

# Representative CSSDM student return dates from the board/school calendars used for
# feature engineering. These are catchment-level proxies, not individual-school calendars.
FRENCH_SCHOOL_STARTS: dict[int, date] = {
    2021: date(2021, 8, 26),
    2022: date(2022, 8, 26),
    2023: date(2023, 8, 28),
    2024: date(2024, 8, 27),
    2025: date(2025, 8, 27),
    2026: date(2026, 8, 27),
}

# Representative EMSB start dates. Individual schools can have orientation/progressive
# entry differences, so explicit overrides are supported below.
ENGLISH_SCHOOL_STARTS: dict[int, date] = {
    2021: date(2021, 8, 31),
    2022: date(2022, 8, 30),
    2023: date(2023, 8, 30),
    2024: date(2024, 8, 29),
    2025: date(2025, 9, 2),
    2026: date(2026, 9, 1),
}


def _first_monday(year: int, month: int) -> date:
    current = date(year, month, 1)
    return current + timedelta(days=(7 - current.weekday()) % 7)


def _labour_day(year: int) -> date:
    return _first_monday(year, 9)


def _default_jewish_school_start(year: int) -> date:
    """Representative Jewish day-school start: first weekday after Labour Day.

    Montreal Jewish schools are heterogeneous. This proxy deliberately keeps the Jewish
    back-to-school transition separate from CSSDM/EMSB while major religious closures are
    represented directly from the Jewish calendar.
    """
    current = _labour_day(year) + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _normalize_starts(
    defaults: Mapping[int, date], overrides: Mapping[int, date | str] | None
) -> dict[int, date]:
    result = dict(defaults)
    if overrides:
        for year, raw in overrides.items():
            parsed = pd.Timestamp(raw)
            if pd.isna(parsed):
                continue
            result[int(year)] = parsed.date()
    return result


def _start_for_year(mapping: Mapping[int, date], year: int, fallback_month: int, fallback_day: int) -> date:
    if year in mapping:
        return mapping[year]
    current = date(year, fallback_month, fallback_day)
    # Keep fallback starts on weekdays.
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current


def _window_flag(current: date | None, start: date, end: date) -> int:
    return int(current is not None and start <= current <= end)


def _school_break_proxy(current: date | None, start: date) -> tuple[int, int, int]:
    """Return summer, winter, spring break proxies for a school system/year."""
    if current is None:
        return 0, 0, 0

    # Summer: after Saint-Jean / usual school-year end through the day before return.
    summer_start = date(start.year, 6, 24)
    summer = int(summer_start <= current < start)

    # Winter break: intentionally broad system-level proxy, avoiding school-specific
    # pedagogical details.
    winter = int(
        (current.month == 12 and current.day >= 23)
        or (current.month == 1 and current.day <= 5)
    )

    # Montreal boards commonly use the first full Monday-Friday week of March. Exact
    # establishment calendars can replace this proxy later if it proves useful.
    spring_monday = _first_monday(current.year, 3)
    spring = int(spring_monday <= current <= spring_monday + timedelta(days=4))
    return summer, winter, spring


def _transition_distances(current: date | None, start: date, max_days: int = 14) -> tuple[int, int]:
    sentinel = max_days + 1
    if current is None:
        return sentinel, sentinel
    delta = (current - start).days
    since = delta if 0 <= delta <= max_days else sentinel
    to_start = -delta if -max_days <= delta <= 0 else sentinel
    return int(since), int(to_start)


def add_calendar_context_features(
    df: pd.DataFrame,
    ts_col: str = "ds",
    *,
    french_start_overrides: Mapping[int, date | str] | None = None,
    english_start_overrides: Mapping[int, date | str] | None = None,
    jewish_start_overrides: Mapping[int, date | str] | None = None,
) -> pd.DataFrame:
    """Add construction-vacation and school-calendar known-future covariates."""
    out = df.copy()
    parsed = pd.to_datetime(out[ts_col], errors="coerce")
    dates: list[date | None] = [value.date() if pd.notna(value) else None for value in parsed]

    french_starts = _normalize_starts(FRENCH_SCHOOL_STARTS, french_start_overrides)
    english_starts = _normalize_starts(ENGLISH_SCHOOL_STARTS, english_start_overrides)
    jewish_starts = _normalize_starts({}, jewish_start_overrides)

    construction = []
    construction_start = []
    construction_end = []
    week_before = []
    week_after = []
    construction_day = []

    french_start_flag = []
    english_start_flag = []
    jewish_start_flag = []
    french_bts = []
    english_bts = []
    jewish_bts = []
    french_since = []
    english_since = []
    jewish_since = []
    french_to = []
    english_to = []
    jewish_to = []

    french_summer = []
    english_summer = []
    jewish_summer = []
    french_winter = []
    english_winter = []
    jewish_winter = []
    french_spring = []
    english_spring = []
    jewish_spring = []

    for current in dates:
        if current is None:
            construction.append(0)
            construction_start.append(0)
            construction_end.append(0)
            week_before.append(0)
            week_after.append(0)
            construction_day.append(0)
            starts = [date(2000, 1, 1)] * 3
        else:
            vacation = CONSTRUCTION_SUMMER_VACATIONS.get(current.year)
            if vacation is None:
                c_start = c_end = None
            else:
                c_start, c_end = vacation
            in_construction = int(c_start is not None and c_start <= current <= c_end)
            construction.append(in_construction)
            construction_start.append(int(c_start is not None and current == c_start))
            construction_end.append(int(c_end is not None and current == c_end))
            week_before.append(
                int(c_start is not None and c_start - timedelta(days=7) <= current < c_start)
            )
            week_after.append(
                int(c_end is not None and c_end < current <= c_end + timedelta(days=7))
            )
            construction_day.append((current - c_start).days + 1 if in_construction else 0)

            french_start = _start_for_year(french_starts, current.year, 8, 27)
            english_start = _start_for_year(english_starts, current.year, 8, 30)
            jewish_start = jewish_starts.get(current.year, _default_jewish_school_start(current.year))
            starts = [french_start, english_start, jewish_start]

        system_lists = [
            (french_start_flag, french_bts, french_since, french_to, french_summer, french_winter, french_spring),
            (english_start_flag, english_bts, english_since, english_to, english_summer, english_winter, english_spring),
            (jewish_start_flag, jewish_bts, jewish_since, jewish_to, jewish_summer, jewish_winter, jewish_spring),
        ]
        for start, lists in zip(starts, system_lists):
            start_flags, bts_flags, since_values, to_values, summer_values, winter_values, spring_values = lists
            start_flags.append(int(current is not None and current == start))
            bts_flags.append(_window_flag(current, start - timedelta(days=7), start + timedelta(days=14)))
            since, to_start = _transition_distances(current, start)
            since_values.append(since)
            to_values.append(to_start)
            summer, winter, spring = _school_break_proxy(current, start)
            summer_values.append(summer)
            winter_values.append(winter)
            spring_values.append(spring)

    out["is_construction_holiday"] = np.asarray(construction, dtype=np.int8)
    out["is_construction_holiday_start"] = np.asarray(construction_start, dtype=np.int8)
    out["is_construction_holiday_end"] = np.asarray(construction_end, dtype=np.int8)
    out["is_week_before_construction_holiday"] = np.asarray(week_before, dtype=np.int8)
    out["is_week_after_construction_holiday"] = np.asarray(week_after, dtype=np.int8)
    out["construction_holiday_day"] = np.asarray(construction_day, dtype=np.int8)

    out["is_french_school_start"] = np.asarray(french_start_flag, dtype=np.int8)
    out["is_english_school_start"] = np.asarray(english_start_flag, dtype=np.int8)
    out["is_jewish_school_start_proxy"] = np.asarray(jewish_start_flag, dtype=np.int8)
    out["is_french_back_to_school_window"] = np.asarray(french_bts, dtype=np.int8)
    out["is_english_back_to_school_window"] = np.asarray(english_bts, dtype=np.int8)
    out["is_jewish_back_to_school_window"] = np.asarray(jewish_bts, dtype=np.int8)
    out["days_since_french_school_start"] = np.asarray(french_since, dtype=np.int8)
    out["days_since_english_school_start"] = np.asarray(english_since, dtype=np.int8)
    out["days_since_jewish_school_start"] = np.asarray(jewish_since, dtype=np.int8)
    out["days_to_french_school_start"] = np.asarray(french_to, dtype=np.int8)
    out["days_to_english_school_start"] = np.asarray(english_to, dtype=np.int8)
    out["days_to_jewish_school_start"] = np.asarray(jewish_to, dtype=np.int8)

    out["is_french_summer_break_proxy"] = np.asarray(french_summer, dtype=np.int8)
    out["is_english_summer_break_proxy"] = np.asarray(english_summer, dtype=np.int8)
    out["is_jewish_summer_break_proxy"] = np.asarray(jewish_summer, dtype=np.int8)
    out["is_french_winter_break_proxy"] = np.asarray(french_winter, dtype=np.int8)
    out["is_english_winter_break_proxy"] = np.asarray(english_winter, dtype=np.int8)
    out["is_jewish_winter_break_proxy"] = np.asarray(jewish_winter, dtype=np.int8)
    out["is_french_spring_break_proxy"] = np.asarray(french_spring, dtype=np.int8)
    out["is_english_spring_break_proxy"] = np.asarray(english_spring, dtype=np.int8)
    out["is_jewish_spring_break_proxy"] = np.asarray(jewish_spring, dtype=np.int8)

    # Re-use the already tested major-Jewish calendar to construct a conservative school
    # religious-closure proxy without treating every Israeli civic holiday as a closure.
    religious = add_holiday_features(out[[ts_col]].copy(), ts_col=ts_col, feature_set="calendars")
    out["is_jewish_school_religious_break_proxy"] = (
        religious["is_major_jewish_holiday"].astype(bool)
        | religious["is_major_jewish_holiday_eve"].astype(bool)
    ).astype(np.int8)

    out["is_french_school_break_proxy"] = out[
        ["is_french_summer_break_proxy", "is_french_winter_break_proxy", "is_french_spring_break_proxy"]
    ].max(axis=1).astype(np.int8)
    out["is_english_school_break_proxy"] = out[
        ["is_english_summer_break_proxy", "is_english_winter_break_proxy", "is_english_spring_break_proxy"]
    ].max(axis=1).astype(np.int8)
    out["is_jewish_school_break_proxy"] = out[
        [
            "is_jewish_summer_break_proxy",
            "is_jewish_winter_break_proxy",
            "is_jewish_spring_break_proxy",
            "is_jewish_school_religious_break_proxy",
        ]
    ].max(axis=1).astype(np.int8)

    out["school_systems_closed_count"] = out[
        ["is_french_school_break_proxy", "is_english_school_break_proxy", "is_jewish_school_break_proxy"]
    ].sum(axis=1).astype(np.int8)
    out["is_any_school_break_proxy"] = (out["school_systems_closed_count"] >= 1).astype(np.int8)
    out["school_transition_intensity"] = out[
        ["is_french_back_to_school_window", "is_english_back_to_school_window", "is_jewish_back_to_school_window"]
    ].sum(axis=1).astype(np.int8)
    out["is_any_back_to_school_window"] = (out["school_transition_intensity"] >= 1).astype(np.int8)
    out["is_split_school_transition"] = out["school_transition_intensity"].between(1, 2).astype(np.int8)
    return out
