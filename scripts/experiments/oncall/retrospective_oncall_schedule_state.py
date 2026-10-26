from __future__ import annotations

"""Audit daily on-call reserve state from the physician schedule.

This is intentionally a schedule-state reconstruction, not an inference about WHY a
shift changed.  In particular, a nominal on-call physician who is also assigned a
regular clinical shift for several core daytime/evening hours is flagged as a proxy
for reserve capacity being consumed/reassigned; the code does not call that sick
leave unless a manually curated exception table later confirms the reason.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from forecast_oncall_probability import SHIFT_DATA_URL, SHIFT_TYPES  # noqa: E402

OCU_CODE = "ocu"
B2_CODE = "b2"
DEFAULT_START_DATE = "2026-05-01"
CORE_START_HOUR = 8
CORE_END_HOUR = 22
REASSIGNED_CORE_HOURS_THRESHOLD = 4.0


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _overlap_hours(start: pd.Timestamp, end: pd.Timestamp, a: pd.Timestamp, b: pd.Timestamp) -> float:
    left = max(start, a)
    right = min(end, b)
    return max((right - left).total_seconds() / 3600.0, 0.0)


def build_daily_schedule_state(start_date: str = DEFAULT_START_DATE) -> pd.DataFrame:
    shifts = pd.read_csv(SHIFT_DATA_URL)
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"], errors="coerce")
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"], errors="coerce")
    shifts = shifts.dropna(subset=["shift_start", "shift_end"]).copy()
    shifts["date"] = shifts["shift_start"].dt.date
    shifts["shift_code_norm"] = _text(shifts["shift_short_name"]).str.casefold()
    shifts["physician_name"] = (
        _text(shifts["first_name"]) + " " + _text(shifts["last_name"])
    ).str.strip()

    oncall_codes = {code.casefold() for code, role in SHIFT_TYPES.items() if role == "oncall"}
    first = pd.Timestamp(start_date).date()
    last = shifts["date"].max()
    dates = pd.date_range(first, last, freq="D")

    rows: list[dict[str, object]] = []
    for stamp in dates:
        date = stamp.date()
        day = shifts[shifts["date"] == date].copy()
        nominal = day[day["shift_code_norm"].isin(oncall_codes)].copy()
        ocu = day[day["shift_code_norm"].eq(OCU_CODE)].copy()
        b2 = day[day["shift_code_norm"].eq(B2_CODE)].copy()

        nominal_names = sorted({x for x in _text(nominal["physician_name"]) if x})
        ocu_names = sorted({x for x in _text(ocu["physician_name"]) if x})

        regular = day[~day["shift_code_norm"].isin(oncall_codes | {OCU_CODE})].copy()
        regular_for_oc = regular[regular["physician_name"].isin(nominal_names)].copy()
        core_start = pd.Timestamp(date) + pd.Timedelta(hours=CORE_START_HOUR)
        core_end = pd.Timestamp(date) + pd.Timedelta(hours=CORE_END_HOUR)

        overlap_rows = []
        for r in regular_for_oc.itertuples(index=False):
            hours = _overlap_hours(r.shift_start, r.shift_end, core_start, core_end)
            if hours > 0:
                overlap_rows.append((r.physician_name, r.shift_short_name, hours))

        core_overlap_hours = sum(item[2] for item in overlap_rows)
        overlap_codes = "|".join(sorted({str(item[1]) for item in overlap_rows}))
        overlap_physicians = "|".join(sorted({str(item[0]) for item in overlap_rows}))
        reserve_reassigned_proxy = core_overlap_hours >= REASSIGNED_CORE_HOURS_THRESHOLD

        weekday = stamp.dayofweek
        b2_expected = weekday in (0, 1, 2, 3)
        b2_absent = bool(b2_expected and b2.empty)

        if nominal.empty:
            supply_state = "no_nominal_oc_row"
        elif reserve_reassigned_proxy and b2_absent:
            supply_state = "reserve_reassigned_and_b2_absent_proxy"
        elif reserve_reassigned_proxy:
            supply_state = "reserve_reassigned_regular_shift_proxy"
        elif b2_absent:
            supply_state = "nominal_reserve_present_b2_absent"
        else:
            supply_state = "nominal_reserve_present_proxy"

        rows.append(
            {
                "date": date,
                "nominal_oc_row_present": bool(len(nominal)),
                "nominal_oc_physicians": "|".join(nominal_names),
                "ocu_positive": bool(len(ocu_names)),
                "ocu_physicians": "|".join(ocu_names),
                "ocu_matches_nominal_oc": bool(set(ocu_names).intersection(nominal_names)),
                "nominal_oc_regular_core_overlap_hours": core_overlap_hours,
                "nominal_oc_regular_core_overlap_codes": overlap_codes,
                "nominal_oc_regular_core_overlap_physicians": overlap_physicians,
                "reserve_reassigned_regular_shift_proxy": reserve_reassigned_proxy,
                "b2_expected_mon_thu": b2_expected,
                "b2_row_present": bool(len(b2)),
                "b2_absent_on_mon_thu": b2_absent,
                "schedule_supply_state": supply_state,
                "reassignment_reason_known": False,
            }
        )
    return pd.DataFrame(rows)


def summarize(state: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        ("first_date", state["date"].min()),
        ("last_date", state["date"].max()),
        ("calendar_days", len(state)),
        ("days_no_nominal_oc_row", int((~state["nominal_oc_row_present"]).sum())),
        ("days_reserve_reassigned_regular_shift_proxy", int(state["reserve_reassigned_regular_shift_proxy"].sum())),
        ("days_b2_absent_when_expected", int(state["b2_absent_on_mon_thu"].sum())),
        ("days_ocu_positive", int(state["ocu_positive"].sum())),
        ("ocu_positive_reserve_reassigned_proxy", int((state["ocu_positive"] & state["reserve_reassigned_regular_shift_proxy"]).sum())),
        ("ocu_positive_b2_absent", int((state["ocu_positive"] & state["b2_absent_on_mon_thu"]).sum())),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct daily on-call reserve/B2 schedule state.")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    state = build_daily_schedule_state(args.start_date)
    summary = summarize(state)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state.to_csv(out / "oncall_schedule_state_daily.csv", index=False)
    summary.to_csv(out / "oncall_schedule_state_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nSchedule exception days:")
    exceptions = state[
        (~state["nominal_oc_row_present"])
        | state["reserve_reassigned_regular_shift_proxy"]
        | state["b2_absent_on_mon_thu"]
        | state["ocu_positive"]
    ]
    print(exceptions.to_string(index=False))


if __name__ == "__main__":
    main()
