#!/usr/bin/env python3
"""Deterministic blurb-fact computer for the ed-flow hourly blurb skill.

Run from the repo root with the repo venv after downloading the seven blurb
inputs to a scratch dir (see SKILL.md):

    cd /opt/hermes/ed-flow-2023
    .venv/bin/python <this script> /tmp/blurb

Prints the readiness-gate result and every number the blurb prose needs:
current canonical values, hourly trajectory, peak, midnight handoff value +
band (from the reference's own plain_language_bands), vertical-vs-POD gate
inputs + trigger, on-call probabilities, on-call impact summary, and
staffing/weather effects. All trajectory values come from the canonical
forecast rows of forecast-v2.1.csv (never reconstructed from current.csv).
Exit code 0 = ready, 1 = readiness failed (do not generate).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staffing_roles import l1_l2_schedule_context

STRETCHER_CAPACITY = 53
ROUTINE_HOURS = {"07:00", "11:00", "15:00", "19:00"}
TARGETS = [
    "Total_TBS",
    "POD_TBS",
    "Vertical_TBS",
    "TTStr",
    "Overflow",
    "WAITINGADM",
    "TRG_HALLWAY1",
    "TRG_HALLWAY_TBS",
]


def _band(value: float, bucket: dict) -> str | None:
    """Classify value using the reference's own plain_language_bands."""
    p10, p25, p75, p90 = bucket.get("p10"), bucket.get("p25"), bucket.get("p75"), bucket.get("p90")
    if None in (p10, p25, p75, p90):
        return None
    if value > p90:
        return "very_heavy"
    if value > p75:
        return "heavy"
    if value < p25:
        return "light"
    return "typical"


def _localize(ts):
    return ts.dt.tz_localize("America/Montreal") if ts.dt.tz is None else ts.dt.tz_convert("America/Montreal")


def schedule_context(shifts: pd.DataFrame, data_hour: pd.Timestamp) -> dict:
    """Schedule evidence is separate from confirmation of on-call availability."""
    shifts = shifts.copy()
    for column in ("shift_start", "shift_end"):
        times = pd.to_datetime(shifts[column])
        shifts[column] = (times.dt.tz_localize("America/Montreal", ambiguous="NaT", nonexistent="NaT")
                          if times.dt.tz is None else times.dt.tz_convert("America/Montreal"))
    is_oncall = shifts["shift_short_name"].isin(["OC1", "OC2", "WOC1", "WOC2", "WOC3"])
    is_teaching = shifts["shift_short_name"].isin(["H1"])
    now = shifts[(shifts.shift_start <= data_hour) & (shifts.shift_end > data_hour)]
    oncall = now[is_oncall.loc[now.index]]
    regular = now[~is_oncall.loc[now.index] & ~is_teaching.loc[now.index]]
    later = shifts[(shifts.shift_start <= data_hour + pd.Timedelta(hours=4))
                   & (shifts.shift_end > data_hour + pd.Timedelta(hours=4))
                   & ~is_oncall & ~is_teaching]
    next_day = data_hour.normalize() + pd.DateOffset(days=1)
    morning = shifts[(shifts.shift_start >= next_day)
                     & (shifts.shift_start < next_day + pd.Timedelta(hours=12)) & ~is_oncall]
    return {
        **l1_l2_schedule_context(shifts, data_hour),
        "scheduled_working_physicians_now": int(regular.user_id.nunique()),
        "scheduled_working_physicians_in_4h": int(later.user_id.nunique()),
        "scheduled_oncall_slots_now": int(oncall.user_id.nunique()),
        "oncall_also_scheduled_regular_shift": bool(set(oncall.user_id) & set(regular.user_id)),
        "oncall_has_next_morning_shift": bool(set(oncall.user_id) & set(morning.user_id)),
        "availability": "unknown",
    }


def staffing_review(history: pd.DataFrame, future: pd.DataFrame,
                    data_hour: pd.Timestamp, context: dict | None = None) -> dict:
    """Provisional workload escalation, independent of activation/impact models.

    Frames contain canonical Total_TBS only, indexed by local ds. Outcomes after
    the decision hour must never appear in history. Availability is explicitly
    confirmed context, not inferred from an empty schedule slot.
    """
    context = context or {}
    recent = history.loc[history.index <= data_hour, "actual"]
    recent = recent.reindex(pd.date_range(data_hour - pd.Timedelta(hours=2), data_hour, freq="h"))
    now = recent.iloc[-1]
    next_hours = future["forecast"].reindex(pd.date_range(
        data_hour + pd.Timedelta(hours=1), periods=6, freq="h"
    ))
    count = int(next_hours.ge(45).sum())
    reasons = []
    if pd.notna(now) and now >= 50:
        reasons.append(f"Current treatment backlog is {now:.0f} TBS (review threshold 50)")
    if recent.notna().all() and recent.ge(45).all():
        reasons.append("TBS has been at least 45 at three consecutive hourly observations")
    if recent.notna().all() and now >= 40 and now - recent.iloc[0] >= 10:
        reasons.append(f"TBS rose by {now - recent.iloc[0]:.0f} over two hours to {now:.0f}")
    if next_hours.notna().all() and count >= 3:
        reasons.append(f"Forecast TBS is at least 45 for {count} of the next 6 hourly endpoints")
    availability = context.get("availability", "unknown")
    if availability not in {"available", "unavailable", "already_active", "unknown"}:
        availability = "unknown"
    late = data_hour.hour >= 21 or data_hour.hour < 7
    return {
        "required": bool(reasons), "reasons": reasons, "thresholds_provisional": True,
        "forecast_hours_ge_45_next_6h": count,
        "forecast_hours_available_next_6h": int(next_hours.notna().sum()),
        "availability": availability, "late_activation_window": late,
        "schedule": context.get("schedule", {}),
    }


def compute(scratch: Path, data_hour: pd.Timestamp | None = None) -> dict:
    """Core readiness + fact computation.

    `data_hour` (aware, America/Montreal) is the hour the blurb is keyed to.
    If None, it defaults to the forecast's own origin (the LATEST available
    data hour) — which is what the automation uses. For the manual skill
    (SKILL.md) pass the box-clock hour so a not-yet-generated hour fails the
    readiness gate instead of silently building on stale data.

    Returns a dict with 'ready' (bool), 'failures' (list[str]), and all the
    facts, or raises on a hard error.
    """
    failures: list[str] = []
    scratch = Path(scratch)

    cur = pd.read_csv(scratch / "current.csv")
    cur["ds"] = _localize(pd.to_datetime(cur["ds"]))
    cur_ds = cur["ds"].max()

    fc = pd.read_csv(scratch / "forecast-v2.1.csv")
    fc["ds"] = pd.to_datetime(fc["ds"])
    fc["forecast_origin"] = pd.to_datetime(fc["forecast_origin"])
    origin = _localize(fc["forecast_origin"]).max()
    if data_hour is None:
        data_hour = origin
    data_hour = data_hour.floor("h")

    f = fc[_localize(fc["forecast_origin"]) == origin].copy()
    f["ds_local"] = _localize(f["ds"])

    dup = f.duplicated(subset=["ds", "target_name"], keep=False)
    if dup.any():
        failures.append(f"{int(dup.sum())} duplicate ds+target_name rows")
    missing = [t for t in TARGETS if t not in f["target_name"].unique()]
    if missing:
        failures.append(f"missing targets: {missing}")

    model_warnings = []
    try:
        onp = pd.read_csv(scratch / "oncall_need_probability.csv")
        if not {"ds", "horizon_hours", "calibrated_probability"}.issubset(onp.columns):
            raise ValueError("Missing probability fields")
        onp["ds"] = _localize(pd.to_datetime(onp["ds"]))
        onp = onp[onp["ds"].eq(data_hour)].copy()
        onp["horizon_hours"] = pd.to_numeric(onp["horizon_hours"], errors="coerce")
        onp["calibrated_probability"] = pd.to_numeric(onp["calibrated_probability"], errors="coerce")
        onp = onp[onp.horizon_hours.isin([4, 6, 8]) & onp.calibrated_probability.between(0, 1)]
        if onp.horizon_hours.duplicated().any():
            raise ValueError("Duplicate probability horizons")
        if not onp["ds"].eq(data_hour).any():
            model_warnings.append("Activation probability is stale or unavailable")
            onp = onp.iloc[:0]
    except (OSError, ValueError, KeyError):
        onp = pd.DataFrame(columns=["ds", "horizon_hours", "calibrated_probability"])
        model_warnings.append("Activation probability is unavailable")
    try:
        oni = pd.read_csv(scratch / "oncall_impact_summary.csv")
    except (OSError, ValueError):
        oni = pd.DataFrame(columns=["estimated_improvement", "target_name"])
        model_warnings.append("Associational impact is unavailable")

    # readiness: key hour must match what the data actually describes
    if cur_ds != data_hour:
        failures.append(f"current.csv latest hour {cur_ds} != data hour {data_hour}")
    if origin != data_hour:
        failures.append(f"forecast origin {origin} != data hour {data_hour}")

    result = {
        "ready": not failures, "failures": failures,
        "origin": origin, "data_hour": data_hour,
        "now": {}, "ttstr_occupancy": 0.0,
        "peak_tbs": None, "peak_horizon": None, "peak_time": None,
        "midnight": None, "midnight_band": None,
        "anomalies": [],
        "oncall_all_low": False, "reassign_trigger": False, "pod_pressure": False,
        "oncall_recommendation": "NO CLEAR RECOMMENDATION",
        "oncall_probabilities": {}, "oncall_impact_summary": {},
        "model_warnings": model_warnings,
    }
    if failures:
        return result

    obs = f[(f["row_type"] == "observed") & (f["ds_local"] == data_hour)]
    for _, r in obs.iterrows():
        result["now"][r["target_name"]] = float(r["actual"])
    missing_observed = [t for t in TARGETS if t not in result["now"] or pd.isna(result["now"][t])]
    if missing_observed:
        result["failures"].append(f"missing current canonical values: {missing_observed}")
        result["ready"] = False
        return result
    result["ttstr_occupancy"] = result["now"].get("TTStr", 0) / STRETCHER_CAPACITY * 100

    fut = f[(f["row_type"] == "forecast") & (f["horizon_hour"] > 0) & (f["horizon_hour"] <= 24)]
    observed_anomalies = obs[obs["actual_anomaly"].astype(str).str.lower().eq("yes")]
    for _, row in observed_anomalies.iterrows():
        result["anomalies"].append({
            "target": row["target_name"], "status": "current",
            "value": float(row["actual"]), "horizon": 0,
        })
    forecast_anomalies = fut[
        fut["forecast_anomaly"].astype(str).str.lower().eq("yes")
        & fut["horizon_hour"].le(4)
    ]
    for _, row in forecast_anomalies.sort_values("horizon_hour").iterrows():
        if not any(a["target"] == row["target_name"] and a["status"] == "current"
                   for a in result["anomalies"]):
            result["anomalies"].append({
                "target": row["target_name"], "status": "next_4h",
                "value": float(row["forecast"]), "horizon": int(row["horizon_hour"]),
            })
    pk = fut[fut["target_name"] == "Total_TBS"].sort_values("forecast", ascending=False)
    if not pk.empty:
        result["peak_tbs"] = float(pk.iloc[0]["forecast"])
        result["peak_horizon"] = int(pk.iloc[0]["horizon_hour"])
        result["peak_time"] = pk.iloc[0]["ds_local"]

    today = fut[(fut["target_name"] == "Total_TBS")
                & (fut["ds_local"].dt.date == data_hour.date())]
    if not today.empty:
        today_peak = today.sort_values("forecast", ascending=False).iloc[0]
        result["remaining_today_peak_tbs"] = float(today_peak["forecast"])
        result["remaining_today_peak_horizon"] = int(today_peak["horizon_hour"])
    history_tbs = f[(f["row_type"] == "observed") & f["target_name"].eq("Total_TBS")].set_index("ds_local")
    future_tbs = fut[fut["target_name"].eq("Total_TBS")].set_index("ds_local")
    # Optional manually confirmed context is accepted only for the exact origin.
    context_path = scratch / "oncall_operational_context.json"
    try:
        context = json.loads(context_path.read_text()) if context_path.exists() else {}
        if not isinstance(context, dict):
            raise ValueError("Operational context must be an object")
    except (OSError, ValueError):
        context = {}
        result["model_warnings"].append("Confirmed operational context is unavailable")
    if str(context.get("data_hour")) != data_hour.isoformat():
        context = {}
    schedule_path = scratch / "all_shifts.csv"
    if schedule_path.exists():
        try:
            context["schedule"] = schedule_context(pd.read_csv(schedule_path), data_hour)
        except (OSError, ValueError, KeyError):
            result["model_warnings"].append("Staffing schedule context is unavailable")
    result["staffing_review"] = staffing_review(history_tbs, future_tbs, data_hour, context)

    midnight = data_hour.normalize() + pd.DateOffset(days=1)
    mid = f[(f["target_name"] == "Total_TBS") & (f["row_type"] == "forecast") & (f["ds_local"] == midnight)]
    if not mid.empty:
        mv = float(mid.iloc[0]["forecast"])
        result["midnight"] = mv
        ref = json.loads((scratch / "blurb_reference_stats.json").read_text())
        mref = ref.get("midnight_total_tbs", {})
        day = data_hour.day_name()
        by_day = mref.get("by_prior_evening_day", {})
        bucket = by_day.get(day) or mref.get("weekday") or mref.get("overall") or {}
        result["midnight_band"] = _band(mv, bucket)

    ev = json.loads((scratch / "blurb_reference_stats.json").read_text()).get("evening_vertical_vs_pod", {})
    now_v = obs[obs["target_name"] == "Vertical_TBS"]["actual"]
    now_p = obs[obs["target_name"] == "POD_TBS"]["actual"]
    near = fut[(fut["horizon_hour"] > 0) & (fut["horizon_hour"] <= 3)]
    near_v = near[near["target_name"] == "Vertical_TBS"]["forecast"]
    near_p = near[near["target_name"] == "POD_TBS"]["forecast"]
    vt, vp, vg = ev.get("vertical_tbs", {}), ev.get("pod_tbs", {}), ev.get("vertical_minus_pod", {})
    if len(now_v) and len(now_p):
        v, p = float(now_v.iloc[0]), float(now_p.iloc[0])
        gap = v - p
        pod_pressure = p >= float(vp.get("p75", float("inf")))
        result["pod_pressure"] = pod_pressure
        v75, g75 = v >= float(vt.get("p75", float("inf"))), gap >= float(vg.get("p75", float("inf")))
        v90, g90 = v >= float(vt.get("p90", float("inf"))), gap >= float(vg.get("p90", float("inf")))
        result["reassign_trigger"] = (v75 and g75) or ((v90 or g90) and not pod_pressure)

    if len(onp):
        if "current_activation_status" in onp:
            result["current_activation_status"] = str(onp.iloc[-1]["current_activation_status"])
        if "activation_label_latest" in onp:
            result["activation_label_latest"] = str(onp.iloc[-1]["activation_label_latest"])
        onp = onp[onp["ds"].eq(data_hour)]
        onp = onp.dropna(subset=["horizon_hours", "calibrated_probability"]).copy()
        result["oncall_probabilities"] = {
            int(row["horizon_hours"]): float(row["calibrated_probability"])
            for _, row in onp.iterrows()
        }
        result["oncall_all_low"] = bool(len(onp) and (onp["calibrated_probability"] < 0.35).all())

    try:
        impact = oni.copy()
        # Legacy files without an origin cannot be certified as current.
        if not {"forecast_origin", "target_name", "estimated_improvement"}.issubset(impact.columns):
            raise ValueError("Missing impact fields/origin")
        impact_origin = _localize(pd.to_datetime(impact["forecast_origin"], errors="coerce"))
        impact = impact[impact_origin.eq(data_hour)].copy()
        impact["estimated_improvement"] = pd.to_numeric(impact["estimated_improvement"], errors="coerce")
        impact = impact[impact.estimated_improvement.notna()
                        & ~impact.estimated_improvement.isin([float("inf"), float("-inf")])]
        if impact.empty:
            raise ValueError("Impact estimates are unavailable or stale")
    except (OSError, ValueError, KeyError):
        impact = pd.DataFrame(columns=["target_name", "estimated_improvement"])
        result["model_warnings"].append("Associational impact is unavailable or stale")
    if len(impact):
        values = impact["estimated_improvement"]
        positive_fraction = float((values > 0).mean())
        negative_fraction = float((values < 0).mean())
        stretcher = impact[impact["target_name"].eq("stretcher_occupancy")]["estimated_improvement"]
        result["oncall_impact_summary"] = {
            "direction": ("improves" if positive_fraction >= 0.60 else
                          "worsens" if negative_fraction >= 0.50 else "mixed"),
            "positive_fraction": positive_fraction,
            "negative_fraction": negative_fraction,
            # Impact target is occupancy percentage points; convert to patients.
            "max_adverse_stretcher": (float(abs(stretcher.min())) * STRETCHER_CAPACITY / 100
                                       if len(stretcher) and stretcher.min() < 0 else 0.0),
        }
    pmax = max(result["oncall_probabilities"].values(), default=0.0)
    review = result["staffing_review"]
    # Neither a low behavior probability nor an adverse associational contrast
    # can veto a workload review. These models cannot justify automatic USE.
    if review["required"]:
        result["oncall_recommendation"] = "STAFFING REVIEW REQUIRED"
    elif pmax >= 0.50 or (pmax >= 0.20 and result["oncall_impact_summary"].get("direction") == "improves"):
        result["oncall_recommendation"] = "CONSIDER"
    elif set(result["oncall_probabilities"]) == {4, 6, 8}:
        result["oncall_recommendation"] = "NO ESCALATION DETECTED"
    return result


def main(scratch: Path) -> int:
    """CLI entry for the manual skill: keys off the box-clock hour so a not-yet-
    generated data hour fails the readiness gate (do not build on stale data)."""
    expected_hour = pd.Timestamp.now(tz="America/Montreal").floor("h")
    r = compute(scratch, data_hour=expected_hour)
    print(f"expected data hour (America/Montreal): {expected_hour}")
    print(f"forecast-v2.1 origin: {r['origin']}")
    if not r["ready"]:
        print("\nREADINESS FAILED — do not generate a blurb:")
        for msg in r["failures"]:
            print(f"  - {msg}")
        return 1
    print("\nREADINESS OK\n")
    _print_report(r)
    return 0


def _print_report(r: dict) -> None:
    print("== NOW (observed at origin) ==")
    for t in TARGETS:
        if t in r["now"]:
            extra = f"  (occupancy {r['ttstr_occupancy']:.1f}%)" if t == "TTStr" else ""
            print(f"  {t:<18} {r['now'][t]:.1f}{extra}")
    if r["peak_tbs"] is not None:
        print(f"\nPEAK Total_TBS: {r['peak_tbs']:.1f} ({r['peak_horizon']}h ahead at {r['peak_time']})")
    print("\n== MIDNIGHT HANDOFF ==")
    if r["midnight"] is not None:
        print(f"  midnight Total_TBS = {r['midnight']:.1f}  band = {r['midnight_band']}")
    else:
        print("  midnight not in forecast horizon")
    print("\n== ON-CALL ==")
    print(f"  recommendation: {r['oncall_recommendation']}")
    print(f"  staffing review: {r.get('staffing_review', {})}")
    print(f"  model warnings: {r.get('model_warnings', [])}")
    print(f"  all horizons < 0.35: {r['oncall_all_low']}")
    print(f"  pod under unusual pressure (>=p75): {r['pod_pressure']}")
    print(f"  reassignment trigger met: {r['reassign_trigger']}")
    hh = r["data_hour"].strftime("%H:%M")
    print(f"\nroutine send hour: {hh in ROUTINE_HOURS}  blurb_id={r['data_hour'].strftime('%Y%m%d-%H00')}")


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/blurb")))
