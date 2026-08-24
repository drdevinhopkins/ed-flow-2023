"""Known-future calendar covariates for JGH hourly Chronos-2 forecasts.

This module keeps two concepts separate:

* demand/system calendar: Quebec/federal/nominal RAMQ/Jewish calendar effects and
  outpatient-access closure/rebound structure.
* JGH institutional calendar: exact RAMQ establishment 0011X dates, used as a
  capacity/access signal and especially when those dates differ from nominal RAMQ.

All features are deterministic from the timestamp and checked-in JGH RAMQ calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from holiday_features import add_holiday_features

DEMAND_CALENDAR_COLUMNS = [
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

JGH_FLAG_COLUMNS = ["is_jgh_ramq_holiday"]
JGH_MISMATCH_COLUMNS = [
    "is_jgh_only_ramq_holiday",
    "is_nominal_only_ramq_holiday",
    "is_ramq_calendar_mismatch",
]
HOURLY_INTERACTION_COLUMNS = [
    "is_jgh_only_daytime",
    "is_jgh_only_evening",
    "is_jgh_only_overnight",
    "is_nominal_only_daytime",
    "is_first_morning_after_jgh_holiday",
    "is_evening_before_jgh_holiday",
    "is_rebound_morning",
    "is_pre_long_closure_evening",
]


def _montreal_local(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("America/Montreal").dt.tz_localize(None)
    return parsed


def _jgh_ramq_for_timestamps(series: pd.Series, ts_col: str) -> pd.Series:
    probe = pd.DataFrame({ts_col: _montreal_local(series)})
    return add_holiday_features(
        probe,
        ts_col=ts_col,
        feature_set="calendars",
        ramq_calendar="jgh",
    )["is_ramq_holiday"].astype(np.int8)


def add_hourly_calendar_features(df: pd.DataFrame, ts_col: str = "ds") -> pd.DataFrame:
    """Add demand-calendar, JGH mismatch, and hourly interaction features."""
    out = df.copy()
    out[ts_col] = _montreal_local(out[ts_col])

    # Nominal RAMQ is retained for the demand/system closure representation because it
    # won the daily-arrivals ablation. Exact JGH dates are kept separate below.
    nominal = add_holiday_features(
        out,
        ts_col=ts_col,
        feature_set="closures",
        ramq_calendar="nominal",
    )
    jgh_ramq = _jgh_ramq_for_timestamps(out[ts_col], ts_col)

    out = nominal
    nominal_ramq = out["is_ramq_holiday"].astype(np.int8)

    out["is_jgh_ramq_holiday"] = jgh_ramq.to_numpy(dtype=np.int8)
    out["is_jgh_only_ramq_holiday"] = (
        (jgh_ramq.eq(1)) & (nominal_ramq.eq(0))
    ).to_numpy(dtype=np.int8)
    out["is_nominal_only_ramq_holiday"] = (
        (nominal_ramq.eq(1)) & (jgh_ramq.eq(0))
    ).to_numpy(dtype=np.int8)
    out["is_ramq_calendar_mismatch"] = jgh_ramq.ne(nominal_ramq).to_numpy(dtype=np.int8)

    local = _montreal_local(out[ts_col])
    hour = local.dt.hour.fillna(-1).astype(int)
    daytime = hour.between(8, 15)
    evening = hour.between(16, 23)
    overnight = hour.between(0, 7)

    jgh_only = out["is_jgh_only_ramq_holiday"].astype(bool)
    nominal_only = out["is_nominal_only_ramq_holiday"].astype(bool)
    out["is_jgh_only_daytime"] = (jgh_only & daytime).astype(np.int8)
    out["is_jgh_only_evening"] = (jgh_only & evening).astype(np.int8)
    out["is_jgh_only_overnight"] = (jgh_only & overnight).astype(np.int8)
    out["is_nominal_only_daytime"] = (nominal_only & daytime).astype(np.int8)

    # Probe adjacent calendar dates directly so boundary rows in a 24h future frame are
    # still labeled correctly even when the adjacent date is outside the supplied frame.
    prev_jgh = _jgh_ramq_for_timestamps(local - pd.Timedelta(days=1), ts_col).astype(bool)
    next_jgh = _jgh_ramq_for_timestamps(local + pd.Timedelta(days=1), ts_col).astype(bool)

    out["is_first_morning_after_jgh_holiday"] = (
        prev_jgh & hour.between(6, 11)
    ).astype(np.int8)
    out["is_evening_before_jgh_holiday"] = (
        next_jgh & hour.between(16, 23)
    ).astype(np.int8)
    out["is_rebound_morning"] = (
        out["is_rebound_after_long_closure"].astype(bool) & hour.between(6, 11)
    ).astype(np.int8)
    out["is_pre_long_closure_evening"] = (
        out["is_pre_long_closure"].astype(bool) & hour.between(16, 23)
    ).astype(np.int8)

    return out


def scenario_columns(scenario: str) -> list[str]:
    """Return covariates for an hourly calendar ablation scenario."""
    if scenario == "baseline":
        return []
    if scenario == "demand_calendar":
        return list(DEMAND_CALENDAR_COLUMNS)
    if scenario == "demand_plus_jgh_flag":
        return [*DEMAND_CALENDAR_COLUMNS, *JGH_FLAG_COLUMNS]
    if scenario == "demand_plus_jgh_mismatch":
        return [*DEMAND_CALENDAR_COLUMNS, *JGH_MISMATCH_COLUMNS]
    if scenario == "demand_plus_jgh_interactions":
        return [
            *DEMAND_CALENDAR_COLUMNS,
            *JGH_MISMATCH_COLUMNS,
            *HOURLY_INTERACTION_COLUMNS,
        ]
    raise ValueError(f"Unknown hourly calendar scenario: {scenario}")
