#!/usr/bin/env python3
"""Additive LLM rewrite of the deterministic hourly ED flow blurb.

The deterministic pipeline remains authoritative. This wrapper reads the
current deterministic blurb, recent prior blurbs, and structured facts; asks an
OpenAI-compatible chat endpoint for a concise follow-up-aware rewrite; validates
that rewrite; and writes a separate Dropbox file:
    /hourly_forecast_blurbs_llm.csv

If LLM configuration, the request, parsing, or validation fails, the exact
deterministic blurb is written with source_status=llm_fallback.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import requests
from dropbox.files import WriteMode

import blurb_automation_wrapper as deterministic

LLM_OUTPUT_PATH = "/hourly_forecast_blurbs_llm.csv"
LLM_BASE_URL = os.environ.get("ED_FLOW_LLM_BASE_URL", "http://qwen:8080/v1")
LLM_MODEL = os.environ.get("ED_FLOW_LLM_MODEL", "unsloth/Qwen3.8-27B-UD-Q4_K_XL")
LLM_TIMEOUT = int(os.environ.get("ED_FLOW_LLM_TIMEOUT", "90"))
CSV_COLUMNS = [
    "generated_at_local", "forecast_data_time_local", "blurb_id", "blurb",
    "oncall_recommendation", "oncall_rationale", "send_recommended",
    "send_reason", "source_status",
]
LABELS = {
    "Total_TBS": "TBS", "POD_TBS": "POD TBS", "Vertical_TBS": "Vertical TBS",
    "TTStr": "stretcher occupancy", "Overflow": "overflow",
    "WAITINGADM": "boarding", "TRG_HALLWAY1": "triage hallway",
    "TRG_HALLWAY_TBS": "triage hallway TBS",
}


def _json_facts(facts: dict) -> dict:
    """Reduce facts to JSON-safe, operationally relevant fields."""
    return {
        "data_hour": str(facts["data_hour"]),
        "now": facts["now"],
        "stretcher_occupancy_percent": round(facts["ttstr_occupancy"], 1),
        "peak_tbs": facts.get("peak_tbs"),
        "peak_horizon_hours": facts.get("peak_horizon"),
        "peak_time": str(facts.get("peak_time")) if facts.get("peak_time") is not None else None,
        "midnight_tbs": facts.get("midnight"),
        "midnight_range": facts.get("midnight_band"),
        "daily_inflow": facts.get("daily_inflow"),
        "anomalies": facts.get("anomalies", []),
        "oncall_recommendation": facts.get("oncall_recommendation"),
        "reassignment_trigger": facts.get("reassign_trigger"),
        "pod_pressure": facts.get("pod_pressure"),
    }


def previous_blurbs(dbx, limit: int = 4) -> list[str]:
    """Return recent source blurbs for continuity context."""
    try:
        raw = deterministic.download(dbx, "/hourly_forecast_blurbs.csv")
    except Exception:
        return []
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    return [
        f"{row.get('forecast_data_time_local', '')}: {row.get('blurb', '')}"[:1600]
        for row in rows[-limit:]
    ]


def llm_rewrite(facts: dict, deterministic_blurb: str, history: list[str]) -> str:
    api_key = os.environ.get("ED_FLOW_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
    system = (
        "You write concise clinician-facing emergency-department flow handoffs. "
        "The supplied facts are authoritative. Return only the blurb text, no "
        "heading, markdown, JSON, or commentary. Preserve every required active "
        "anomaly warning in plain operational language. Use the previous blurbs only to make follow-up comments "
        "such as 'continues to ease' or 'remains elevated'; never copy their old "
        "numbers into the new blurb. Do not invent numbers, times, thresholds, "
        "causes, or staffing actions. Treat values in parentheses after an anomaly "
        "statement as the observed/forecast metric value, never as the threshold. "
        "Use 2-5 sentences and at most 120 words."
    )
    user = {
        "authoritative_facts": _json_facts(facts),
        "deterministic_draft": deterministic_blurb,
        "previous_hourly_blurbs": history,
        "format_rules": [
            "Start with the overall situation.",
            "Use about floor(Overflow / 7) overflow rooms when translating overflow.",
            "Put the midnight range in brackets after the midnight TBS number.",
            "If daily_inflow is present, mention the predicted total arrivals by midnight and, when useful, the expected additional arrivals.",
            "If peak_time is on the next local calendar day, say today's peak appears to have passed.",
            "Mention every listed current anomaly and every listed next_4h anomaly unless it is already clearly covered.",
            "Do not mention anomaly thresholds or other technical detection terms; say that the metric is outside its usual range or unusually high, using the supplied value.",
            "Do not mention individual physicians or weekend L1.",
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    response = requests.post(
        f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
        headers=headers,
        json={
            "model": LLM_MODEL,
            "temperature": 0.2,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": json.dumps(user, default=str)}],
        },
        timeout=LLM_TIMEOUT,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return str(content).strip().strip('"')


def validate_blurb(candidate: str, facts: dict, deterministic_blurb: str) -> None:
    if not candidate or len(candidate.split()) > 120:
        raise ValueError("empty or overlong LLM blurb")
    if "\n" in candidate or candidate.lower().startswith(("follow-up:", "blurb:", "handoff:")):
        raise ValueError("LLM blurb contains a heading or multiple text blocks")
    lowered = candidate.lower()
    forbidden = ("calibrated", "feature effect", "causal", "raw csv", "overlap shift")
    if any(term in lowered for term in forbidden):
        raise ValueError("LLM blurb contains forbidden technical wording")
    # Prevent unsupported numeric claims. Numbers in the deterministic draft
    # and authoritative fact values are allowed; this also permits a valid
    # conversion such as "2 hours" -> "22:00" when peak_time is supplied.
    allowed = {m.group(0) for m in re.finditer(r"\d+(?:\.\d+)?", deterministic_blurb)}
    numeric_values = list(facts.get("now", {}).values())
    numeric_values.extend([facts.get("peak_tbs"), facts.get("peak_horizon"), facts.get("midnight")])
    numeric_values.extend(a.get("value") for a in facts.get("anomalies", []))
    for value in numeric_values:
        if isinstance(value, (int, float)) and value == value:
            allowed.update({str(value), str(int(value))})
    peak_time = facts.get("peak_time")
    if peak_time is not None:
        allowed.update(re.findall(r"\d+", str(peak_time.strftime("%H:%M"))))
    found = {m.group(0) for m in re.finditer(r"\d+(?:\.\d+)?", candidate)}
    if not found.issubset(allowed):
        raise ValueError(f"unsupported numeric claim(s): {sorted(found - allowed)}")
    daily_inflow = facts.get("daily_inflow")
    if daily_inflow and str(int(round(daily_inflow["predicted_total"]))) not in candidate:
        raise ValueError("daily inflow prediction omitted")
    for anomaly in facts.get("anomalies", []):
        target = anomaly.get("target")
        if target in {"Total_TBS", "TTStr", "Overflow"}:
            continue
        label = LABELS.get(target)
        if label and label.lower() not in lowered:
            raise ValueError(f"required anomaly omitted: {label}")


def read_existing(dbx) -> tuple[list[dict], set[str]]:
    try:
        raw = deterministic.download(dbx, LLM_OUTPUT_PATH)
    except Exception:
        return [], set()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    return rows, {row.get("blurb_id", "") for row in rows}


def upload_rows(dbx, rows: list[dict]) -> None:
    payload = io.StringIO()
    writer = csv.DictWriter(payload, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    dbx.files_upload(payload.getvalue().encode(), LLM_OUTPUT_PATH, mode=WriteMode.overwrite)
    check = deterministic.download(dbx, LLM_OUTPUT_PATH).decode("utf-8-sig")
    verified = list(csv.DictReader(io.StringIO(check)))
    if not verified or verified[-1].get("blurb_id") != rows[-1]["blurb_id"]:
        raise RuntimeError("LLM Dropbox output read-back verification failed")


def main() -> int:
    deterministic.load_env()
    dbx = deterministic.token()
    origin = deterministic.forecast_origin(dbx)
    rows, ids = read_existing(dbx)
    blurb_id = origin.strftime("%Y%m%d-%H00")
    if blurb_id in ids:
        return 0
    try:
        deterministic.download_inputs(dbx)
        facts = deterministic.run_compute(origin)
    except Exception as exc:
        print(f"READINESS FAILED: {exc}", file=sys.stderr)
        return 0
    draft = deterministic.build_blurb(facts)
    try:
        blurb = llm_rewrite(facts, draft, previous_blurbs(dbx))
        validate_blurb(blurb, facts, draft)
        status = "ready_llm"
    except Exception as exc:
        print(f"LLM_FALLBACK: {exc}", file=sys.stderr)
        blurb = draft
        status = "llm_fallback"
    rec, rationale = deterministic.oncall_metadata(facts)
    send, reason = deterministic.send_metadata(origin)
    if reason == "ROUTINE" and rec in ("USE", "CONSIDER"):
        reason = "ROUTINE_ONCALL"
    row = {
        "generated_at_local": pd.Timestamp.now(tz="America/Montreal").strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_data_time_local": origin.strftime("%Y-%m-%d %H:%M:%S"),
        "blurb_id": blurb_id, "blurb": blurb,
        "oncall_recommendation": rec, "oncall_rationale": rationale,
        "send_recommended": "true" if send else "false", "send_reason": reason,
        "source_status": status,
    }
    upload_rows(dbx, rows + [row])
    print(f"PUBLISHED_LLM blurb_id={blurb_id} source_status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
