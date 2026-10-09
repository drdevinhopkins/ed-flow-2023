from __future__ import annotations

"""Audit and derive day-level on-call-use labels from the schedule's ocU shift.

Starting in 2026-05, the schedule includes a special ``ocU`` (On Call Used) shift.
A physician name assigned to that shift is evidence that on-call was used on that
calendar day, but it does not provide a trustworthy activation timestamp.

This module therefore produces DAY-LEVEL labels only. It never invents an hourly
activation time. If an ocU placeholder exists for a day but has no physician
assigned, the output records ``no_recorded_use`` with lower confidence than the
explicit historical hourly label file. Days with no ocU row at all are ``unknown``.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from forecast_oncall_probability import SHIFT_DATA_URL  # noqa: E402

OCU_CODE = "ocu"


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def load_ocu_day_labels() -> tuple[pd.DataFrame, pd.DataFrame]:
    shifts = pd.read_csv(SHIFT_DATA_URL)
    required = {"shift_start", "shift_short_name", "first_name", "last_name"}
    missing = required.difference(shifts.columns)
    if missing:
        raise ValueError(f"Missing required shift columns: {sorted(missing)}")

    shifts = shifts.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"], errors="coerce")
    shifts["shift_code_norm"] = _text(shifts["shift_short_name"]).str.casefold()
    ocu = shifts[shifts["shift_code_norm"].eq(OCU_CODE)].copy()
    if ocu.empty:
        raise ValueError("No ocU shift rows found in all_shifts.csv")

    ocu["date"] = ocu["shift_start"].dt.date
    ocu["physician_id"] = (_text(ocu["first_name"]) + " " + _text(ocu["last_name"])).str.strip()
    ocu["physician_assigned"] = ocu["physician_id"].ne("")

    def join_ids(values: pd.Series) -> str:
        ids = sorted({str(value).strip() for value in values if str(value).strip()})
        return "|".join(ids)

    daily = (
        ocu.groupby("date", as_index=False)
        .agg(
            ocu_rows=("shift_short_name", "size"),
            ocu_assigned_rows=("physician_assigned", "sum"),
            ocu_physician_ids=("physician_id", join_ids),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["ocu_used_day"] = daily["ocu_assigned_rows"].gt(0)
    daily["ocu_day_label"] = daily["ocu_used_day"].map(
        {True: "used_day_positive", False: "no_recorded_use"}
    )
    daily["label_resolution"] = "calendar_day"
    daily["activation_timestamp_known"] = False
    daily["label_source"] = "schedule_ocU"
    daily["label_confidence"] = daily["ocu_used_day"].map(
        {True: "positive_high", False: "negative_lower"}
    )

    # Coverage is defined only where an ocU row actually exists. We do not silently
    # turn a missing ocU row into a negative day.
    start = pd.Timestamp(min(daily["date"]))
    end = pd.Timestamp(max(daily["date"]))
    full_dates = pd.DataFrame({"date": pd.date_range(start, end, freq="D").date})
    coverage = full_dates.merge(daily[["date", "ocu_day_label"]], on="date", how="left")
    coverage["coverage_status"] = coverage["ocu_day_label"].notna().map(
        {True: "ocu_row_present", False: "unknown_no_ocu_row"}
    )
    return daily, coverage


def summary_table(daily: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [
        ("first_ocu_date", min(daily["date"])),
        ("last_ocu_date", max(daily["date"])),
        ("calendar_days_span", int(len(coverage))),
        ("days_with_ocu_row", int(len(daily))),
        ("days_missing_ocu_row_inside_span", int((coverage["coverage_status"] == "unknown_no_ocu_row").sum())),
        ("days_ocu_used", int(daily["ocu_used_day"].sum())),
        ("days_no_recorded_use", int((~daily["ocu_used_day"]).sum())),
        ("positive_day_rate_among_ocu_rows", float(daily["ocu_used_day"].mean())),
        ("activation_timestamp_available", False),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive day-level on-call-use labels from ocU schedule rows.")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    daily, coverage = load_ocu_day_labels()
    summary = summary_table(daily, coverage)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out / "oncall_ocu_day_labels.csv", index=False)
    coverage.to_csv(out / "oncall_ocu_day_coverage.csv", index=False)
    summary.to_csv(out / "oncall_ocu_day_summary.csv", index=False)

    print(summary.to_string(index=False))
    positives = daily[daily["ocu_used_day"]]
    print("\nocU positive days:")
    if positives.empty:
        print("none")
    else:
        print(positives[["date", "ocu_physician_ids", "ocu_assigned_rows"]].to_string(index=False))


if __name__ == "__main__":
    main()
