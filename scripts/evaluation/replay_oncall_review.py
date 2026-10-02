"""Replay provisional workload rules from canonical hourly observations.

This is an observed-state replay, not a forecast backtest or causal evaluation.
Schedule exports may have retrospective edits; they do not verify availability.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("review_facts", ROOT / "scripts/automation/compute_blurb_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)


def replay(hourly: pd.DataFrame, shifts: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    hourly = hourly.copy()
    hourly["ds"] = pd.to_datetime(hourly["ds"])
    # Localize only the requested replay window; old exports contain synthetic
    # local 02:00 rows at spring DST transitions.
    lower, upper = pd.Timestamp(start) - pd.Timedelta(hours=2), pd.Timestamp(end) + pd.Timedelta(hours=6)
    hourly = hourly[hourly.ds.between(lower, upper)].copy()
    hourly["ds"] = facts._localize(hourly["ds"])
    shifts = shifts.copy()
    for column in ("shift_start", "shift_end"):
        shifts[column] = pd.to_datetime(shifts[column])
    shifts = shifts[(shifts.shift_start <= upper + pd.Timedelta(days=1)) & (shifts.shift_end >= lower)]
    hourly = hourly.sort_values("ds").drop_duplicates("ds").set_index("ds")
    # Use an existing canonical column; never sum raw report fields.
    column = "Total_TBS" if "Total_TBS" in hourly else "total_tbs"
    if column not in hourly:
        raise ValueError("Replay requires an existing canonical Total_TBS/total_tbs column")
    history = hourly[[column]].rename(columns={column: "actual"})
    future = pd.DataFrame({"forecast": []}, index=pd.DatetimeIndex([], tz="America/Montreal"))
    rows = []
    for hour in hourly.loc[start:end].index:
        schedule = facts.schedule_context(shifts, hour)
        review = facts.staffing_review(history.loc[:hour], future, hour, {"schedule": schedule})
        outcome = history.actual.reindex(pd.date_range(hour + pd.Timedelta(hours=1), periods=6, freq="h"))
        rows.append({"data_hour": hour, "total_tbs": history.loc[hour, "actual"],
                     "review_required": review["required"], "reasons": "; ".join(review["reasons"]),
                     "availability": "unknown", "late_activation_window": review["late_activation_window"],
                     **{k: v for k, v in schedule.items() if k != "availability"},
                     "subsequent_hours_observed": int(outcome.notna().sum()),
                     "subsequent_hours_ge_45": int(outcome.ge(45).sum()),
                     "subsequent_peak_tbs": outcome.max()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--shifts", type=Path, required=True)
    parser.add_argument("--start", default="2026-09-18")
    parser.add_argument("--end", default="2026-10-01 23:00")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(pd.read_csv(args.hourly), pd.read_csv(args.shifts), args.start, args.end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"{result.review_required.sum()} of {len(result)} observed hours flagged for staffing review")
