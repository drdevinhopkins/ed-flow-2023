#!/usr/bin/env python3
"""Combine intraday retrospective, prospective, artifact, and monitor evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_ARTIFACT_FIELDS = {
    "model_version",
    "artifact_sha256",
    "training_start",
    "training_end",
    "training_days",
    "model_fingerprint",
    "model_fingerprint_version",
    "training_input_fingerprint_version",
    "state_training_fingerprint",
    "weather_training_fingerprint",
}


def build_assessment(
    retrospective: dict[str, object],
    prospective: dict[str, object],
    health: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    retrospective_gates = retrospective.get("retrospective_gates", {})
    prospective_gates = prospective.get("gates", {})
    critical_alerts = [
        item.get("code", "unknown")
        for item in health.get("alerts", [])
        if item.get("severity") == "critical"
    ]
    missing_artifact_fields = sorted(
        field for field in REQUIRED_ARTIFACT_FIELDS if not manifest.get(field)
    )
    objective_gates = {
        "retrospective_ready": bool(
            retrospective.get("retrospective_ready")
            and retrospective_gates
            and all(retrospective_gates.values())
        ),
        "prospective_ready": bool(
            prospective.get("prospective_ready")
            and prospective_gates
            and all(prospective_gates.values())
        ),
        "artifact_and_input_provenance_complete": not missing_artifact_fields,
        "no_critical_monitor_alerts": not critical_alerts,
    }
    evidence_ready = all(objective_gates.values())
    blockers = [name for name, passed in objective_gates.items() if not passed]
    blockers.extend(
        f"retrospective:{name}" for name, passed in retrospective_gates.items() if not passed
    )
    blockers.extend(
        f"prospective:{name}" for name, passed in prospective_gates.items() if not passed
    )
    if missing_artifact_fields:
        blockers.append("missing_artifact_fields:" + ",".join(missing_artifact_fields))
    if critical_alerts:
        blockers.append("critical_monitor_alerts:" + ",".join(critical_alerts))
    blockers.append("explicit_manual_go_no_go_authorization")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_model": retrospective.get("candidate_model"),
        "model_version": manifest.get("model_version"),
        "objective_gates": objective_gates,
        "retrospective_gates": retrospective_gates,
        "prospective_gates": prospective_gates,
        "prospective_days": prospective.get("prospective_days", 0),
        "operational_hour_counts": prospective.get("operational_hour_counts", {}),
        "collection_reliability": prospective.get("collection_reliability"),
        "monitor_health": health.get("health", "missing"),
        "critical_monitor_alerts": critical_alerts,
        "missing_artifact_fields": missing_artifact_fields,
        "objective_evidence_ready": evidence_ready,
        "manual_authorization_recorded": False,
        "production_ready": False,
        "recommendation": "pending_manual_go_no_go" if evidence_ready else "no_go",
        "blockers": blockers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrospective-json", type=Path, required=True)
    parser.add_argument("--prospective-json", type=Path, required=True)
    parser.add_argument("--health-json", type=Path, required=True)
    parser.add_argument("--artifact-manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    args = parse_args()
    result = build_assessment(
        _read(args.retrospective_json),
        _read(args.prospective_json),
        _read(args.health_json),
        _read(args.artifact_manifest_json),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
