from __future__ import annotations

"""Build a label-safe retrospective on-call replay dataset.

The output deliberately separates three concepts that were previously easy to blur:

1. exact-hour activation labels (trusted only through 2026-04-30 23:00);
2. day-level ``ocU`` positives from the schedule (May 2026 onward; positive-only);
3. reserve feasibility/context (nominal reserve, reassignment proxy, B2 gap, late/rest
   constraints).

Missing ``ocU`` after the trusted hourly boundary is never interpreted as "not used".
A reserve consumed by a regular-shift reassignment is also not labelled a missed call.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecast_oncall_probability import TS_COL, load_dataset  # noqa: E402
from retrospective_oncall_availability_adjusted import add_callability_features  # noqa: E402
from retrospective_oncall_label_coverage import TRUSTED_HOURLY_LABEL_END  # noqa: E402
from retrospective_oncall_schedule_state import (  # noqa: E402
    DEFAULT_START_DATE as SCHEDULE_STATE_START_DATE,
    build_daily_schedule_state,
)

FLOW_COLUMNS = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "stretcher_occupancy",
    "overflow",
    "WAITINGADM",
    "Inflow_Total",
    "TRG_HALLWAY_TBS",
    "RESUS",
)
STAFFING_COLUMNS = (
    "n_flow",
    "n_pod",
    "n_vertical",
    "n_overlap",
    "n_teaching",
    "n_night",
    "n_oncall",
    "n_total_scheduled",
    "oncall_physician_id",
)
SCHEDULE_STATE_COLUMNS = (
    "nominal_oc_row_present",
    "nominal_oc_physicians",
    "ocu_positive",
    "ocu_physicians",
    "ocu_matches_nominal_oc",
    "nominal_oc_regular_core_overlap_hours",
    "nominal_oc_regular_core_overlap_codes",
    "nominal_oc_regular_core_overlap_physicians",
    "reserve_reassigned_regular_shift_proxy",
    "b2_expected_mon_thu",
    "b2_row_present",
    "b2_absent_on_mon_thu",
    "schedule_supply_state",
    "reassignment_reason_known",
)


def add_schedule_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TS_COL] = pd.to_datetime(out[TS_COL], errors="coerce")
    out["date"] = out[TS_COL].dt.date

    schedule = build_daily_schedule_state(SCHEDULE_STATE_START_DATE).copy()
    schedule["date"] = pd.to_datetime(schedule["date"], errors="coerce").dt.date
    keep = ["date", *[c for c in SCHEDULE_STATE_COLUMNS if c in schedule.columns]]
    out = out.merge(schedule[keep], on="date", how="left", validate="many_to_one")
    return out


def label_status(row: pd.Series) -> str:
    stamp = pd.Timestamp(row[TS_COL])
    if stamp <= TRUSTED_HOURLY_LABEL_END:
        return "trusted_exact_hour"
    ocu_value = row.get("ocu_positive", False)
    if pd.notna(ocu_value) and bool(ocu_value):
        return "positive_only_day_exact_hour_unknown"
    return "unknown_post_cutoff"


def feasibility_state(row: pd.Series) -> str:
    stamp = pd.Timestamp(row[TS_COL])
    post_schedule_audit = stamp.date() >= pd.Timestamp(SCHEDULE_STATE_START_DATE).date()

    if post_schedule_audit:
        nominal = row.get("nominal_oc_row_present")
        reassigned = row.get("reserve_reassigned_regular_shift_proxy")
        if pd.notna(nominal) and not bool(nominal):
            return "unavailable_no_nominal_reserve"
        if pd.notna(reassigned) and bool(reassigned):
            return "reserve_consumed_regular_shift_proxy"

    status = str(row.get("callability_status", "unknown"))
    if status == "unavailable_no_oncall_scheduled":
        return "unavailable_no_oncall_scheduled"
    if status.startswith("constrained_"):
        return status
    if status == "callable_proxy":
        return "callable_proxy"
    return "unknown"


def add_label_and_feasibility_semantics(
    df: pd.DataFrame,
    late_call_hour: int,
    min_rest_buffer_hours: float,
) -> pd.DataFrame:
    out = add_callability_features(df, late_call_hour, min_rest_buffer_hours)
    out = add_schedule_context(out)

    ts = pd.to_datetime(out[TS_COL], errors="coerce")
    raw_active = pd.to_numeric(out["oncall_active"], errors="coerce").clip(0, 1)
    trusted = ts.le(TRUSTED_HOURLY_LABEL_END)

    out["activation_label_status"] = out.apply(label_status, axis=1)
    out["activation_exact_hour"] = raw_active.where(trusted, np.nan)
    out["activation_exact_hour_known"] = trusted
    if "ocu_positive" in out.columns:
        out["ocu_positive_day"] = out["ocu_positive"].fillna(False).astype(bool)
    else:
        out["ocu_positive_day"] = False
    out["post_cutoff_missing_is_negative"] = False
    out["reserve_feasibility_state"] = out.apply(feasibility_state, axis=1)
    out["reserve_callable_proxy"] = out["reserve_feasibility_state"].eq("callable_proxy")
    out["reserve_consumed_proxy"] = out["reserve_feasibility_state"].eq(
        "reserve_consumed_regular_shift_proxy"
    )
    if "b2_absent_on_mon_thu" in out.columns:
        out["b2_gap_context"] = out["b2_absent_on_mon_thu"].fillna(False).astype(bool)
    else:
        out["b2_gap_context"] = False

    out["replay_interpretation"] = "context_only"
    out.loc[
        out["activation_label_status"].eq("trusted_exact_hour")
        & out["activation_exact_hour"].eq(1),
        "replay_interpretation",
    ] = "activated_exact_hour"
    out.loc[
        out["activation_label_status"].eq("positive_only_day_exact_hour_unknown"),
        "replay_interpretation",
    ] = "known_used_day_timing_unknown"
    out.loc[
        out["reserve_consumed_proxy"] & ~out["ocu_positive_day"].fillna(False).astype(bool),
        "replay_interpretation",
    ] = "reserve_consumed_not_missed_call"
    out.loc[
        out["reserve_feasibility_state"].isin(
            ["unavailable_no_nominal_reserve", "unavailable_no_oncall_scheduled"]
        ),
        "replay_interpretation",
    ] = "reserve_unavailable"

    return out


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        TS_COL,
        "date",
        "activation_exact_hour",
        "activation_exact_hour_known",
        "activation_label_status",
        "ocu_positive_day",
        "reserve_feasibility_state",
        "reserve_callable_proxy",
        "reserve_consumed_proxy",
        "callability_status",
        "oncall_scheduled",
        "hours_to_next_non_oncall_shift",
        "b2_gap_context",
        "replay_interpretation",
        "post_cutoff_missing_is_negative",
        *FLOW_COLUMNS,
        *STAFFING_COLUMNS,
        *SCHEDULE_STATE_COLUMNS,
    ]
    seen: set[str] = set()
    keep = []
    for col in ordered:
        if col in df.columns and col not in seen:
            keep.append(col)
            seen.add(col)
    return df[keep].copy()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df[TS_COL])
    post = ts.gt(TRUSTED_HOURLY_LABEL_END)
    status = df["activation_label_status"]
    feasibility = df["reserve_feasibility_state"]

    metrics: list[tuple[str, object]] = [
        ("first_hour", ts.min()),
        ("last_hour", ts.max()),
        ("trusted_hourly_label_end", TRUSTED_HOURLY_LABEL_END),
        ("rows_total", int(len(df))),
        ("rows_trusted_exact_hour", int(status.eq("trusted_exact_hour").sum())),
        (
            "rows_post_cutoff_positive_only_day",
            int(status.eq("positive_only_day_exact_hour_unknown").sum()),
        ),
        ("rows_post_cutoff_unknown", int(status.eq("unknown_post_cutoff").sum())),
        (
            "post_cutoff_rows_never_interpreted_as_negative",
            int((post & ~df["activation_exact_hour_known"]).sum()),
        ),
        ("rows_callable_proxy", int(feasibility.eq("callable_proxy").sum())),
        (
            "rows_reserve_consumed_regular_shift_proxy",
            int(feasibility.eq("reserve_consumed_regular_shift_proxy").sum()),
        ),
        (
            "rows_no_nominal_or_scheduled_reserve",
            int(
                feasibility.isin(
                    ["unavailable_no_nominal_reserve", "unavailable_no_oncall_scheduled"]
                ).sum()
            ),
        ),
        ("rows_b2_gap_context", int(df["b2_gap_context"].sum())),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-safe retrospective on-call replay dataset.")
    parser.add_argument("--late-call-hour", type=int, default=21)
    parser.add_argument("--min-rest-buffer-hours", type=float, default=10.0)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    raw = load_dataset()
    replay = add_label_and_feasibility_semantics(
        raw,
        late_call_hour=args.late_call_hour,
        min_rest_buffer_hours=args.min_rest_buffer_hours,
    )
    replay = select_output_columns(replay)
    summary = summarize(replay)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay.to_csv(out / "oncall_replay_dataset.csv", index=False)
    summary.to_csv(out / "oncall_replay_dataset_summary.csv", index=False)

    print(summary.to_string(index=False))
    print(
        "\nSemantics: post-cutoff missing ocU is UNKNOWN, reserve-consumed proxy is NOT a "
        "missed call, and B2 absence is retained as context rather than reserve unavailability."
    )


if __name__ == "__main__":
    main()
