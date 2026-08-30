#!/usr/bin/env python3
"""Evaluate shadow-pipeline health without publishing or altering forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

LOCAL_TZ = ZoneInfo("America/Montreal")
SHADOW_HOURS = tuple(range(11, 19))


def evaluate_shadow_health(
    status: dict[str, object],
    history: pd.DataFrame,
    forecasts: pd.DataFrame,
    manifest: dict[str, object],
    readiness: dict[str, object],
    *,
    now: pd.Timestamp,
    max_status_age_minutes: int = 120,
) -> dict[str, object]:
    alerts: list[dict[str, str]] = []
    now = pd.Timestamp(now)
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    local_now = now.tz_convert(LOCAL_TZ)

    generated = pd.to_datetime(status.get("generated_at_utc"), utc=True, errors="coerce")
    age_minutes = None if pd.isna(generated) else (now - generated).total_seconds() / 60.0
    if age_minutes is None or age_minutes > max_status_age_minutes:
        alerts.append({"severity": "critical", "code": "stale_status"})

    state = str(status.get("status", "missing"))
    reason = str(status.get("reason", ""))
    scheduled_idle = (
        state == "suppressed_data_quality"
        and "outside the shadow window" in reason
    )
    if state == "shadow_fallback":
        alerts.append({"severity": "warning", "code": "model_fallback"})
    elif state == "suppressed_data_quality" and not scheduled_idle:
        alerts.append({"severity": "warning", "code": "data_quality_suppression"})
    elif state not in {"shadow_only", "suppressed_data_quality"}:
        alerts.append({"severity": "critical", "code": "unknown_status"})

    if not history.empty and "status" in history:
        recent = history.tail(2).copy()
        recent_states = recent["status"].astype(str)
        recent_reasons = recent.get("reason", pd.Series("", index=recent.index)).fillna("").astype(str)
        expected_idle = recent_states.eq("suppressed_data_quality") & recent_reasons.str.contains(
            "outside the shadow window", regex=False
        )
        abnormal = recent_states.isin(["shadow_fallback", "suppressed_data_quality"]) & ~expected_idle
        if len(recent) == 2 and abnormal.all() and local_now.hour in SHADOW_HOURS:
            alerts.append({"severity": "critical", "code": "consecutive_abnormal_runs"})

    missing_cutoff_hours: list[int] = []
    if not history.empty and "generated_at_utc" in history:
        attempt_times = pd.to_datetime(history["generated_at_utc"], utc=True, errors="coerce")
        local_attempts = attempt_times.dt.tz_convert(LOCAL_TZ)
        attempted_hours = set(
            local_attempts.loc[
                local_attempts.dt.date.eq(local_now.date())
                & local_attempts.dt.hour.isin(SHADOW_HOURS)
            ].dt.hour.astype(int)
        )
        if local_now.hour <= SHADOW_HOURS[0]:
            expected_hours: tuple[int, ...] = ()
        elif local_now.hour <= SHADOW_HOURS[-1]:
            expected_hours = tuple(range(SHADOW_HOURS[0], local_now.hour))
        else:
            expected_hours = SHADOW_HOURS
        missing_cutoff_hours = [hour for hour in expected_hours if hour not in attempted_hours]
        if missing_cutoff_hours:
            alerts.append({"severity": "critical", "code": "missing_shadow_cutoff_attempt"})

    candidate = forecasts.copy()
    if not candidate.empty and "status" in candidate:
        candidate = candidate.loc[candidate["status"].eq("shadow_only")].copy()
    quarantined_drift_count = 0
    if not candidate.empty and "model_fingerprint" in candidate:
        ordered = candidate.sort_values("generated_at_utc").copy()
        group = ["model_version", "training_end"]
        if "model_fingerprint_version" in ordered:
            group.append("model_fingerprint_version")
        reference = (
            ordered.loc[ordered["model_fingerprint"].notna()]
            .groupby(group, dropna=False)["model_fingerprint"]
            .first()
            .rename("reference_fingerprint")
        )
        ordered = ordered.merge(reference, on=group, how="left")
        drifted = (
            ordered["model_fingerprint"].notna()
            & ordered["reference_fingerprint"].notna()
            & ordered["model_fingerprint"].ne(ordered["reference_fingerprint"])
        )
        quarantined_drift_count = int(drifted.sum())
        candidate = ordered.loc[~drifted].drop(columns="reference_fingerprint")
        if quarantined_drift_count:
            alerts.append({"severity": "warning", "code": "quarantined_model_fingerprint_drift"})
    duplicate_count = 0
    invalid_count = 0
    if not candidate.empty:
        keys = ["model_version", "forecast_day", "cutoff_hour"]
        duplicate_count = int(candidate.duplicated(keys).sum())
        numeric = candidate[["observed_arrivals", "predicted_total", "p10_total", "p90_total"]].apply(
            pd.to_numeric, errors="coerce"
        )
        valid = (
            np.isfinite(numeric).all(axis=1)
            & numeric["predicted_total"].ge(numeric["observed_arrivals"])
            & numeric["p10_total"].le(numeric["predicted_total"])
            & numeric["predicted_total"].le(numeric["p90_total"])
        )
        invalid_count = int((~valid).sum())
        if duplicate_count:
            alerts.append({"severity": "critical", "code": "duplicate_forecast_keys"})
        if invalid_count:
            alerts.append({"severity": "critical", "code": "forecast_invariant_failure"})

        latest = candidate.sort_values("generated_at_utc").iloc[-1]
        manifest_hash = manifest.get("artifact_sha256")
        status_hash = status.get("artifact_sha256")
        if pd.notna(status_hash) and status_hash != manifest_hash:
            alerts.append({"severity": "critical", "code": "artifact_hash_mismatch"})

        fingerprint = status.get("model_fingerprint")
        manifest_fingerprint = manifest.get("model_fingerprint")
        if pd.notna(fingerprint) and fingerprint != manifest_fingerprint:
            alerts.append({"severity": "critical", "code": "model_fingerprint_mismatch"})

    severities = {item["severity"] for item in alerts}
    health = "critical" if "critical" in severities else "warning" if "warning" in severities else "healthy"
    if scheduled_idle and health == "healthy":
        health = "healthy_idle"
    return {
        "generated_at_utc": now.tz_convert("UTC").isoformat(),
        "health": health,
        "latest_status": state,
        "status_age_minutes": None if age_minutes is None else round(age_minutes, 1),
        "candidate_forecasts": int(len(candidate)),
        "duplicate_forecast_keys": duplicate_count,
        "invalid_candidate_forecasts": invalid_count,
        "quarantined_model_drift_forecasts": quarantined_drift_count,
        "missing_shadow_cutoff_hours": missing_cutoff_hours,
        "prospective_days": int(readiness.get("prospective_days", 0)),
        "required_prospective_days": 28,
        "production_ready": bool(readiness.get("production_ready", False)),
        "alerts": alerts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--status-history-csv", type=Path, required=True)
    parser.add_argument("--forecasts-csv", type=Path, required=True)
    parser.add_argument("--artifact-manifest-json", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--now")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    result = evaluate_shadow_health(
        json.loads(args.status_json.read_text()),
        pd.read_csv(args.status_history_csv) if args.status_history_csv.exists() else pd.DataFrame(),
        pd.read_csv(args.forecasts_csv) if args.forecasts_csv.exists() else pd.DataFrame(),
        json.loads(args.artifact_manifest_json.read_text())
        if args.artifact_manifest_json.exists()
        else {},
        json.loads(args.readiness_json.read_text()) if args.readiness_json.exists() else {},
        now=now,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["health"] == "critical":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
