"""Montreal/JGH holiday covariates for native Chronos-2 forecasts.

The feature builder intentionally separates several calendars that can affect ED demand
and hospital operations differently:

* Quebec statutory/public holidays.
* Canada-wide federal holidays.
* RAMQ medical-professional holiday families (13 annual holidays). Establishments may
  publish their own observed dates, so callers can optionally provide explicit overrides.
* Jewish holidays, with an additional flag for major religious holidays and their eves.
* Health-system access closure/rebound structure around weekends and non-Jewish holidays.

All features are deterministic known-future covariates. They are date based in the
America/Montreal timezone and can therefore be safely built for both history and the
forecast horizon.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable, Mapping

import holidays
import numpy as np
import pandas as pd

DEFAULT_TZ = "America/Montreal"
MAX_PROXIMITY_DAYS = 7
MAX_CLOSURE_STREAK = 7

# Israel public holidays also contain modern civic holidays. These name fragments retain
# the religious holidays most likely to affect the Montreal Jewish community.
MAJOR_JEWISH_NAME_FRAGMENTS = (
    "Rosh Hashanah",
    "Rosh Hashana",
    "Yom Kippur",
    "Sukkot",
    "Shemini Atzeret",
    "Simchat Torah",
    "Passover",
    "Pesach",
    "Shavuot",
)


def _easter_sunday(year: int) -> date:
    """Gregorian Easter Sunday using the Meeus/Jones/Butcher algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _first_weekday(year: int, month: int, weekday: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7)


def _second_weekday(year: int, month: int, weekday: int) -> date:
    return _first_weekday(year, month, weekday) + timedelta(days=7)


def _monday_before_may_25(year: int) -> date:
    current = date(year, 5, 24)
    while current.weekday() != 0:
        current -= timedelta(days=1)
    return current


def build_ramq_nominal_calendar(years: Iterable[int]) -> dict[date, str]:
    """Return the 13 RAMQ medical-professional holiday families on nominal dates.

    RAMQ establishment calendars can move fixed-date holidays to specific weekdays.
    Therefore these dates are deliberately called *nominal*. Use ``ramq_overrides`` in
    :func:`add_holiday_features` when JGH-specific observed dates are available.
    """
    calendar: dict[date, str] = {}
    for year in sorted(set(int(y) for y in years)):
        easter = _easter_sunday(year)
        entries = {
            date(year, 1, 1): "New Year's Day",
            date(year, 1, 2): "Day after New Year's Day",
            easter - timedelta(days=2): "Good Friday",
            easter + timedelta(days=1): "Easter Monday",
            _monday_before_may_25(year): "National Patriots' Day",
            date(year, 6, 24): "Saint-Jean-Baptiste Day",
            date(year, 7, 1): "Canada Day",
            _first_weekday(year, 9, 0): "Labour Day",
            _second_weekday(year, 10, 0): "Thanksgiving",
            date(year, 12, 24): "Christmas Eve",
            date(year, 12, 25): "Christmas Day",
            date(year, 12, 26): "Boxing Day",
            date(year, 12, 31): "New Year's Eve",
        }
        calendar.update(entries)
    return calendar


def _normalize_ramq_overrides(
    ramq_overrides: Mapping[date | str, str] | Iterable[date | str] | None,
) -> dict[date, str]:
    if ramq_overrides is None:
        return {}
    if isinstance(ramq_overrides, Mapping):
        items = ramq_overrides.items()
    else:
        items = ((value, "JGH/RAMQ observed holiday") for value in ramq_overrides)

    normalized: dict[date, str] = {}
    for raw_date, name in items:
        parsed = pd.Timestamp(raw_date)
        if pd.isna(parsed):
            continue
        normalized[parsed.date()] = str(name)
    return normalized


def _local_dates(series: pd.Series, local_tz: str) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(local_tz)
    return parsed.dt.date


def _nearest_distances(
    dates: list[date | None], holiday_dates: set[date], max_days: int
) -> tuple[np.ndarray, np.ndarray]:
    """Days to next / since previous holiday, clipped to ``max_days + 1``."""
    sentinel = max_days + 1
    to_next = np.full(len(dates), sentinel, dtype=np.int16)
    since_prev = np.full(len(dates), sentinel, dtype=np.int16)
    if not holiday_dates:
        return to_next, since_prev

    for idx, current in enumerate(dates):
        if current is None or pd.isna(current):
            continue
        for distance in range(0, max_days + 1):
            if current + timedelta(days=distance) in holiday_dates:
                to_next[idx] = distance
                break
        for distance in range(0, max_days + 1):
            if current - timedelta(days=distance) in holiday_dates:
                since_prev[idx] = distance
                break
    return to_next, since_prev


def _closure_features(
    dates: list[date | None], system_holidays: set[date]
) -> dict[str, np.ndarray]:
    """Describe outpatient-access closure streaks surrounding each local calendar date.

    This intentionally excludes Jewish religious holidays from the definition of a general
    health-system closure. Their demand/staffing effects remain separate covariates.
    """

    def is_closed(day: date) -> bool:
        return day.weekday() >= 5 or day in system_holidays

    n = len(dates)
    closed_today = np.zeros(n, dtype=np.int8)
    before = np.zeros(n, dtype=np.int8)
    ahead = np.zeros(n, dtype=np.int8)
    prev_7d = np.zeros(n, dtype=np.int8)
    next_7d = np.zeros(n, dtype=np.int8)

    for idx, current in enumerate(dates):
        if current is None:
            continue
        closed_today[idx] = int(is_closed(current))
        prev_7d[idx] = sum(is_closed(current - timedelta(days=i)) for i in range(1, 8))
        next_7d[idx] = sum(is_closed(current + timedelta(days=i)) for i in range(1, 8))

        for distance in range(1, MAX_CLOSURE_STREAK + 1):
            if is_closed(current - timedelta(days=distance)):
                before[idx] += 1
            else:
                break
        for distance in range(1, MAX_CLOSURE_STREAK + 1):
            if is_closed(current + timedelta(days=distance)):
                ahead[idx] += 1
            else:
                break

    business_day = 1 - closed_today
    return {
        "is_system_closed_day": closed_today,
        "closed_days_immediately_before": before,
        "closed_days_immediately_ahead": ahead,
        "closed_days_previous_7d": prev_7d,
        "closed_days_next_7d": next_7d,
        "is_first_business_day_after_closure": (business_day * (before >= 1)).astype(np.int8),
        "is_rebound_after_long_closure": (business_day * (before >= 3)).astype(np.int8),
        "is_last_business_day_before_closure": (business_day * (ahead >= 1)).astype(np.int8),
        "is_pre_long_closure": (business_day * (ahead >= 3)).astype(np.int8),
    }


def add_holiday_features(
    df: pd.DataFrame,
    ts_col: str = "ds",
    local_tz: str = DEFAULT_TZ,
    observed: bool = True,
    feature_set: str = "rich",
    ramq_overrides: Mapping[date | str, str] | Iterable[date | str] | None = None,
) -> pd.DataFrame:
    """Add known-future holiday covariates for Montreal ED forecasting.

    ``feature_set`` supports progressively richer ablation groups:

    * ``legacy``: existing Quebec + Israel binary flags.
    * ``calendars``: separate Quebec, federal, RAMQ, Jewish, and major-Jewish flags.
    * ``shoulders``: calendars plus pre/post-holiday and long-weekend edge flags.
    * ``rich``: shoulders plus holiday proximity and seasonal holiday-cluster features.
    * ``closures``: rich plus outpatient-access closure/rebound structure.
    """
    allowed = {"legacy", "calendars", "shoulders", "rich", "closures"}
    if feature_set not in allowed:
        raise ValueError(f"feature_set must be one of {sorted(allowed)}")

    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")
    local_dates_series = _local_dates(out[ts_col], local_tz)
    dates: list[date | None] = [d if pd.notna(d) else None for d in local_dates_series]
    valid_dates = [d for d in dates if d is not None]
    if not valid_dates:
        raise ValueError("No valid datetimes found when building holiday covariates.")

    # Include an extra year on both sides so proximity/shoulder features work around New Year.
    min_year = min(d.year for d in valid_dates) - 1
    max_year = max(d.year for d in valid_dates) + 1
    years = list(range(min_year, max_year + 1))

    qc = holidays.Canada(subdiv="QC", years=years, observed=observed)
    federal = holidays.Canada(years=years, observed=observed)
    jewish = holidays.Israel(years=years, observed=False, language="en_US")
    ramq_nominal = build_ramq_nominal_calendar(years)
    ramq_override_map = _normalize_ramq_overrides(ramq_overrides)
    ramq = dict(ramq_nominal)
    ramq.update(ramq_override_map)

    qc_dates = set(qc.keys())
    federal_dates = set(federal.keys())
    jewish_dates = set(jewish.keys())
    ramq_dates = set(ramq.keys())

    major_jewish_dates = {
        d
        for d, name in jewish.items()
        if any(fragment.lower() in str(name).lower() for fragment in MAJOR_JEWISH_NAME_FRAGMENTS)
    }

    out["is_qc_holiday"] = np.array([int(d in qc_dates) if d else 0 for d in dates], dtype=np.int8)
    out["is_jewish_holiday"] = np.array(
        [int(d in jewish_dates) if d else 0 for d in dates], dtype=np.int8
    )

    if feature_set == "legacy":
        return out

    out["is_federal_holiday"] = np.array(
        [int(d in federal_dates) if d else 0 for d in dates], dtype=np.int8
    )
    out["is_ramq_holiday"] = np.array(
        [int(d in ramq_dates) if d else 0 for d in dates], dtype=np.int8
    )
    out["is_major_jewish_holiday"] = np.array(
        [int(d in major_jewish_dates) if d else 0 for d in dates], dtype=np.int8
    )
    out["is_major_jewish_holiday_eve"] = np.array(
        [int(d is not None and d + timedelta(days=1) in major_jewish_dates) for d in dates],
        dtype=np.int8,
    )

    # An "any holiday" event uses the major Jewish subset rather than every Israeli civic
    # holiday, while the broad is_jewish_holiday flag remains available to Chronos.
    primary_holidays = qc_dates | federal_dates | ramq_dates | major_jewish_dates
    out["is_any_holiday"] = np.array(
        [int(d in primary_holidays) if d else 0 for d in dates], dtype=np.int8
    )

    if feature_set == "calendars":
        return out

    out["is_day_before_holiday"] = np.array(
        [int(d is not None and d + timedelta(days=1) in primary_holidays) for d in dates],
        dtype=np.int8,
    )
    out["is_day_after_holiday"] = np.array(
        [int(d is not None and d - timedelta(days=1) in primary_holidays) for d in dates],
        dtype=np.int8,
    )
    out["is_friday_before_monday_holiday"] = np.array(
        [
            int(d is not None and d.weekday() == 4 and d + timedelta(days=3) in primary_holidays)
            for d in dates
        ],
        dtype=np.int8,
    )
    out["is_tuesday_after_monday_holiday"] = np.array(
        [
            int(d is not None and d.weekday() == 1 and d - timedelta(days=1) in primary_holidays)
            for d in dates
        ],
        dtype=np.int8,
    )
    out["is_thursday_before_friday_holiday"] = np.array(
        [
            int(d is not None and d.weekday() == 3 and d + timedelta(days=1) in primary_holidays)
            for d in dates
        ],
        dtype=np.int8,
    )
    out["is_monday_after_friday_holiday"] = np.array(
        [
            int(d is not None and d.weekday() == 0 and d - timedelta(days=3) in primary_holidays)
            for d in dates
        ],
        dtype=np.int8,
    )
    out["is_long_weekend_edge"] = (
        out[
            [
                "is_friday_before_monday_holiday",
                "is_tuesday_after_monday_holiday",
                "is_thursday_before_friday_holiday",
                "is_monday_after_friday_holiday",
            ]
        ]
        .max(axis=1)
        .astype(np.int8)
    )

    if feature_set == "shoulders":
        return out

    days_to, days_since = _nearest_distances(dates, primary_holidays, MAX_PROXIMITY_DAYS)
    out["days_to_next_holiday"] = days_to
    out["days_since_previous_holiday"] = days_since
    out["holiday_within_2_days"] = np.array(
        [int(min(a, b) <= 2) for a, b in zip(days_to, days_since)], dtype=np.int8
    )
    out["holiday_within_7_days"] = np.array(
        [int(min(a, b) <= 7) for a, b in zip(days_to, days_since)], dtype=np.int8
    )

    out["is_christmas_newyear_period"] = np.array(
        [
            int(
                d is not None
                and ((d.month == 12 and d.day >= 23) or (d.month == 1 and d.day <= 3))
            )
            for d in dates
        ],
        dtype=np.int8,
    )
    out["is_quebec_canada_day_period"] = np.array(
        [
            int(
                d is not None
                and ((d.month == 6 and d.day >= 23) or (d.month == 7 and d.day <= 2))
            )
            for d in dates
        ],
        dtype=np.int8,
    )

    if feature_set == "rich":
        return out

    # General health-system closure logic excludes Jewish holidays because those may affect
    # JGH demand/staffing without closing the broader Montreal outpatient system.
    system_holidays = qc_dates | federal_dates | ramq_dates
    for column, values in _closure_features(dates, system_holidays).items():
        out[column] = values
    return out


def holiday_feature_columns(feature_set: str = "rich") -> list[str]:
    """Return feature names in stable order without requiring a real input frame."""
    probe = pd.DataFrame({"ds": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    featured = add_holiday_features(probe, feature_set=feature_set)
    return [column for column in featured.columns if column != "ds"]
