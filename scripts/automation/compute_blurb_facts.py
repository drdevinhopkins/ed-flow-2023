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

    onp = pd.read_csv(scratch / "oncall_need_probability.csv")
    onp["ds"] = _localize(pd.to_datetime(onp["ds"]))
    onp_hour = onp["ds"].max() if len(onp) else None
    oni = pd.read_csv(scratch / "oncall_impact_summary.csv")

    # readiness: key hour must match what the data actually describes
    if cur_ds != data_hour:
        failures.append(f"current.csv latest hour {cur_ds} != data hour {data_hour}")
    if origin != data_hour:
        failures.append(f"forecast origin {origin} != data hour {data_hour}")
    if onp_hour is None or onp_hour != data_hour:
        failures.append(f"oncall need file hour {onp_hour} != data hour {data_hour}")

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
    }
    if failures:
        return result

    obs = f[(f["row_type"] == "observed") & (f["ds_local"] == data_hour)]
    for _, r in obs.iterrows():
        result["now"][r["target_name"]] = float(r["actual"])
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

    midnight = (data_hour.normalize() + pd.Timedelta(hours=24)).tz_convert("America/Montreal")
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
        onp = onp.dropna(subset=["horizon_hours", "calibrated_probability"]).copy()
        result["oncall_probabilities"] = {
            int(row["horizon_hours"]): float(row["calibrated_probability"])
            for _, row in onp.iterrows()
        }
        result["oncall_all_low"] = bool((onp["calibrated_probability"] < 0.35).all())

    impact = oni.copy()
    impact["estimated_improvement"] = pd.to_numeric(
        impact.get("estimated_improvement"), errors="coerce"
    )
    impact = impact.dropna(subset=["estimated_improvement"])
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
            "max_adverse_stretcher": (float(abs(stretcher.min()))
                                       if len(stretcher) and stretcher.min() < 0 else 0.0),
        }
        pmax = max(result["oncall_probabilities"].values(), default=0.0)
        benefit = positive_fraction >= 0.60
        if pmax >= 0.50 and benefit:
            result["oncall_recommendation"] = "USE"
        elif pmax >= 0.20 and benefit:
            result["oncall_recommendation"] = "CONSIDER"
        elif not benefit and pmax < 0.50:
            result["oncall_recommendation"] = "NOT INDICATED"
        elif pmax < 0.05:
            result["oncall_recommendation"] = "NOT INDICATED"
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
    print(f"  all horizons < 0.35: {r['oncall_all_low']}")
    print(f"  pod under unusual pressure (>=p75): {r['pod_pressure']}")
    print(f"  reassignment trigger met: {r['reassign_trigger']}")
    hh = r["data_hour"].strftime("%H:%M")
    print(f"\nroutine send hour: {hh in ROUTINE_HOURS}  blurb_id={r['data_hour'].strftime('%Y%m%d-%H00')}")


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/blurb")))
