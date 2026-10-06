"""Effective-dated L1/L2 deployment rules, shared by forecasts and blurbs."""
from __future__ import annotations

import pandas as pd

LOCAL_TZ = "America/Montreal"
# Operational start confirmed by Devin Hopkins. Earlier L2 assignments were
# four-hour training/return-to-practice shifts, not the new POD coverage.
L1_L2_EFFECTIVE_DATE = "2026-10-01"
STAFFING_ROLE_VERSION = f"l1-l2-hourly-v1-from-{L1_L2_EFFECTIVE_DATE}"


def local_times(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="mixed", errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return parsed


def resolve_hourly_roles(
    expanded: pd.DataFrame, *, effective_date: str = L1_L2_EFFECTIVE_DATE,
) -> pd.DataFrame:
    """Resolve roles using actual simultaneous assignments, retaining raw codes.

    Before the boundary all legacy roles remain unchanged. From the boundary,
    L1 is flexible unless L2 is active in the same Mon–Thu hour. In those hours,
    L1 is vertical and L2 is pod. Other L2 deployments retain their legacy role.
    """
    out = expanded.copy()
    hours = local_times(out["ds"])
    start = pd.Timestamp(effective_date)
    if pd.isna(start) or start.tzinfo is not None or start != start.normalize():
        raise ValueError("L1/L2 effective date must be a valid local calendar date")
    out["legacy_shift_type"] = out["shift_type"]
    out["role_assignment_rule"] = "legacy"
    modern = hours.ge(start)
    eligible = modern & hours.dt.dayofweek.le(3)
    l1 = out["shift_short_name"].eq("L1")
    l2 = out["shift_short_name"].eq("L2")
    l2_hours = set(hours[eligible & l2])
    paired = eligible & hours.isin(l2_hours)
    out.loc[modern & l1, "shift_type"] = "flexible"
    out.loc[modern & l1, "role_assignment_rule"] = "l1_flexible"
    out.loc[paired & l1, "shift_type"] = "vertical"
    out.loc[eligible & l2, "shift_type"] = "pod"
    out.loc[(paired & l1) | (eligible & l2), "role_assignment_rule"] = "l1_l2_split"
    return out


def l1_l2_schedule_context(shifts: pd.DataFrame, data_hour: pd.Timestamp) -> dict:
    """Zone assignments from a schedule; no assertion of actual attendance."""
    hour = pd.Timestamp(data_hour)
    if hour.tzinfo is not None:
        hour = hour.tz_convert(LOCAL_TZ).tz_localize(None)
    active = shifts.copy()
    starts = local_times(active["shift_start"])
    ends = local_times(active["shift_end"])
    active = active.loc[starts.le(hour) & ends.gt(hour)].copy()
    active["ds"] = hour
    active["shift_type"] = "overlap"
    resolved = resolve_hourly_roles(active)
    roles = {}
    for code in ("L1", "L2"):
        matches = resolved.loc[resolved.shift_short_name.eq(code), "shift_type"]
        roles[code.lower() + "_role"] = matches.iloc[0] if len(matches) else "absent"
    roles["l1_l2_split"] = roles["l1_role"] == "vertical" and roles["l2_role"] == "pod"
    roles["staffing_role_version"] = STAFFING_ROLE_VERSION
    return roles
