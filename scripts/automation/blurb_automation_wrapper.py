#!/usr/bin/env python3
"""15-minute blurb automation wrapper for the ed-flow hourly blurb pipeline.

Design (agreed 2026-08-28):
  * Read the LATEST inputs from Dropbox (do NOT re-run the forecast model here).
  * Publish ONLY when a genuinely NEW data hour appears (forecast_origin of
    forecast-v2.1.csv) that is not already covered in hourly_forecast_blurbs.csv
    (matched on forecast_data_time_local hour AND blurb_id). Otherwise: do
    nothing, stay silent.
  * On a genuinely new data hour: build the blurb deterministically (facts via
    the skill's compute_blurb_facts.py) + a skill-template 5-sentence prose,
    write an 8-column request JSON, and publish through the repo's
    append_ed_forecast_blurb_request.py (the current pipeline; it does the
    deduped, conflict-safe append). The old generate_ed_forecast_blurb.py is
    schema-stale (6 cols) and refuses the live 8-col log, so it is NOT used.
  * Readiness/staleness failures or a publish error -> print a warning to
    stderr, exit 0, no publish, no chat noise.

Idempotent + self-contained: sources the profile .env for Dropbox creds, so it
works in a fresh cron session without relying on inherited environment.

Silence contract: stdout is EMPTY (and exit 0) on every skip/no-op, so a
no_agent cron delivers nothing on a tick that published nothing. A brief
announcement line is printed ONLY when a row was actually appended (~<=1x/hour).
"""
from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

REPO = Path(os.environ.get("ED_FLOW_REPO", "/opt/apps/ed-flow-2023"))
# Minimal blurb venv (py3.13 + pandas/requests/dropbox). The repo's main .venv is
# tied to a since-removed Python 3.12 and is not usable for this automation.
VENV_PY = REPO / ".venv-blurb" / "bin" / "python"
COMPUTE = Path(os.environ.get(
    "ED_FLOW_COMPUTE",
    "/opt/data/profiles/edflow/skills/ed-flow/hourly-forecast-blurb/scripts/compute_blurb_facts.py",
))
APPEND_WORKER = REPO / "scripts" / "append_ed_forecast_blurb_request.py"
ENV_FILE = Path(os.environ.get("ED_FLOW_ENV", "/opt/data/profiles/edflow/.env"))
# Keep automation state under the active profile by default. A prior manual
# root run left /tmp/blurb_auto root-owned, which blocked the cron user.
SCRATCH = Path(os.environ.get(
    "ED_FLOW_SCRATCH", "/opt/data/profiles/edflow/state-local/blurb_auto"
))

DBX_HOST = "api.dropboxapi.com"
TOKEN_PATH = "/oauth2/token"
GRANT = "refresh" + "_" + "token"

STRETCHER_CAPACITY = 53
ROUTINE_HOURS = {"07:00", "11:00", "15:00", "19:00"}


def load_env() -> None:
    """Source the profile .env into os.environ (does not clobber set vars)."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    missing = [k for k in ("DROPBOX_APP_KEY", "DROPBOX_APP_SECRET", "DROPBOX_REFRESH_TOKEN")
               if not os.environ.get(k)]
    if missing:
        print("MISSING_CREDENTIALS: " + ", ".join(missing), file=sys.stderr)
        sys.exit(2)


def token():
    import dropbox  # noqa: F401  (ensure SDK importable)
    url = "https://" + DBX_HOST + TOKEN_PATH
    r = requests.post(url, data={
        "grant_type": GRANT,
        "refresh_token": os.environ["DROPBOX_REFRESH_TOKEN"],
        "client_id": os.environ["DROPBOX_APP_KEY"],
        "client_secret": os.environ["DROPBOX_APP_SECRET"],
    }, timeout=30)
    r.raise_for_status()
    return dropbox.Dropbox(r.json()["access_token"], timeout=60)


def download(dbx, path):
    meta, response = dbx.files_download(path)  # SDK 12.x returns a tuple
    return response.content


def localize(ts: pd.Series) -> pd.Series:
    if ts.dt.tz is None:
        return ts.dt.tz_localize("America/Montreal")
    return ts.dt.tz_convert("America/Montreal")


def forecast_origin(dbx):
    """The forecast's own data hour (aware, hour-truncated) from forecast-v2.1.csv."""
    fc = pd.read_csv(io.BytesIO(download(dbx, "/forecast-v2.1.csv")))
    fc["forecast_origin"] = pd.to_datetime(fc["forecast_origin"])
    o = fc["forecast_origin"].max()
    o = o.tz_localize("America/Montreal") if o.tz is None else o.tz_convert("America/Montreal")
    return o.floor("h")


def covered(dbx):
    """Set of (data-hour bucket, blurb_id) already in the blurb log."""
    try:
        raw = download(dbx, "/hourly_forecast_blurbs.csv")
    except Exception:
        return set(), set()
    text = raw.decode("utf-8-sig")
    hours, ids = set(), set()
    if not text.strip():
        return hours, ids
    for row in csv.DictReader(io.StringIO(text)):
        v = (row.get("forecast_data_time_local") or "").strip()
        if v:
            hours.add(v[:13])
        b = (row.get("blurb_id") or "").strip()
        if b:
            ids.add(b)
    return hours, ids


def download_inputs(dbx):
    """Download the seven blurb inputs to SCRATCH. Returns origin or raises."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    files = [
        "current.csv", "forecast-v2.1.csv", "oncall_need_probability.csv",
        "oncall_impact_summary.csv", "forecast_variable_effects_hourly.csv",
        "blurb_reference_stats.json", "hourly_forecast_blurbs.csv",
    ]
    for name in files:
        (SCRATCH / name).write_bytes(download(dbx, "/" + name))
    # origin from the downloaded forecast
    fc = pd.read_csv(SCRATCH / "forecast-v2.1.csv")
    fc["forecast_origin"] = pd.to_datetime(fc["forecast_origin"])
    o = fc["forecast_origin"].max()
    o = o.tz_localize("America/Montreal") if o.tz is None else o.tz_convert("America/Montreal")
    return o.floor("h")


def run_compute(origin) -> dict:
    """Compute facts via the skill's importable compute() (origin-keyed).

    We import compute() and key to `origin` (the forecast's own data hour),
    NOT shell out to the compute CLI. The CLI gates on the box clock — right
    for the manual skill, wrong for a 15-min loop where the box clock can be
    a few minutes ahead of the forecast origin. compute() with data_hour=origin
    validates that the inputs describe that origin and returns the facts dict.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("cbf", str(COMPUTE))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = mod.compute(Path(SCRATCH), data_hour=origin)
    if not r.get("ready"):
        raise RuntimeError("compute readiness failed: " + "; ".join(r.get("failures", [])))
    return r


def band_phrase(band: str | None, val: float) -> str:
    if band == "very_heavy":
        return "a very heavy night for this time of year"
    if band == "heavy":
        return "a heavier-than-usual night for this time of year"
    if band == "light":
        return "a lighter-than-usual night for this time of year"
    return "a typical night for this time of year"


def build_blurb(facts: dict) -> str:
    """Deterministic 5-sentence blurb (NOW -> PEAK -> MIDNIGHT -> ONCALL -> ACTION)."""
    now = facts["now"]
    tbs = now.get("Total_TBS")
    occ = facts["ttstr_occupancy"]
    overflow = now.get("Overflow")
    pod = now.get("POD_TBS")
    vert = now.get("Vertical_TBS")

    ovf = (f"{int(round(overflow))} overflow patients "
           f"(about {int(round(overflow / 4))} overflow room(s) in use)") if overflow else "no overflow"

    # 1: NOW
    s1 = (f"Right now the ED has {int(round(tbs))} patients to be seen, "
          f"with stretcher occupancy around {int(round(occ))}% and {ovf}.")

    # 2: PEAK / direction
    if facts.get("peak_tbs") is not None and facts.get("peak_horizon"):
        ph = facts["peak_horizon"]
        peak = facts["peak_tbs"]
        if peak > (tbs or 0) + 1:
            s2 = (f"The day is likely to build toward a peak of about "
                  f"{int(round(peak))} patients to be seen around {ph}h from now.")
        else:
            s2 = "The load is close to its peak and should not build much further."
    else:
        s2 = "The load is broadly steady over the next few hours."

    # 3: MIDNIGHT
    if facts.get("midnight") is not None:
        s3 = (f"For the night handoff, midnight is tracking for roughly "
              f"{int(round(facts['midnight']))} patients to be seen — "
              f"{band_phrase(facts.get('midnight_band'), facts['midnight'])}.")
    else:
        s3 = "Midnight is outside the forecast horizon, so the night handoff is not estimated."

    # 4: ON-CALL — state the recommendation and evidence in the blurb itself.
    probs = facts.get("oncall_probabilities", {})
    prob_text = ", ".join(
        f"{round(prob * 100):.0f}% at {horizon}h"
        for horizon, prob in sorted(probs.items())
    )
    impact = facts.get("oncall_impact_summary", {})
    recommendation = facts.get("oncall_recommendation", "NO CLEAR RECOMMENDATION")
    direction = impact.get("direction")
    adverse = impact.get("max_adverse_stretcher", 0.0)
    if recommendation == "NOT INDICATED" and direction == "worsens":
        s4 = (f"On-call is not recommended: calibrated need is {prob_text}, "
              f"and modeled activation worsens flow (up to {adverse:.1f} additional "
              "stretcher patients).")
    elif recommendation == "USE":
        s4 = (f"On-call is recommended: calibrated need is {prob_text}, and "
              "modeled activation shows a meaningful flow benefit.")
    elif recommendation == "CONSIDER":
        s4 = (f"Consider on-call: calibrated need is {prob_text}, and modeled "
              "activation shows a meaningful flow benefit.")
    elif prob_text:
        s4 = (f"On-call recommendation is not clear: calibrated need is {prob_text}; "
              "the modeled activation effect is mixed or small.")
    else:
        s4 = "On-call recommendation is not clear because the calibrated need probabilities are unavailable."

    # 5: ACTION — direct available staff to the pressure point.
    if facts["reassign_trigger"] and not facts["pod_pressure"]:
        s5 = ("Vertical is carrying a markedly heavier share than POD; the orange evening overlap shift "
              "(4 PM–midnight) can focus on new patients in Vertical, while L1 "
              "(1–9 PM) can be directed to prepod, POD, or Vertical according to where "
              "the need is greatest.")
    elif facts["reassign_trigger"] and facts["pod_pressure"]:
        s5 = ("Vertical and POD both need attention; L1 (1–9 PM) can be directed to prepod, "
              "POD, or Vertical according to where the need is greatest, while the orange evening overlap shift "
              "(4 PM–midnight) can focus on new patients in Vertical when "
              "that remains the main pressure point.")
    else:
        s5 = "No staffing changes are indicated at this time."

    return " ".join([s1, s2, s3, s4, s5])


def oncall_metadata(facts: dict) -> tuple[str, str]:
    """(recommendation, rationale) with all decision evidence retained."""
    rec = facts.get("oncall_recommendation", "NO CLEAR RECOMMENDATION")
    probs = facts.get("oncall_probabilities", {})
    prob_text = ", ".join(
        f"{round(prob * 100):.0f}% at {horizon}h"
        for horizon, prob in sorted(probs.items())
    ) or "unavailable"
    impact = facts.get("oncall_impact_summary", {})
    direction = impact.get("direction", "unknown")
    adverse = impact.get("max_adverse_stretcher", 0.0)
    if rec == "NOT INDICATED" and direction == "worsens":
        rat = (f"Calibrated need is {prob_text}; modeled activation worsens flow, "
               f"including up to {adverse:.1f} additional stretcher patients. Activation "
               "is not recommended.")
    elif rec in ("USE", "CONSIDER"):
        rat = (f"Calibrated need is {prob_text}; modeled activation shows a meaningful "
               f"flow benefit. Recommendation: {rec}.")
    else:
        rat = (f"Calibrated need is {prob_text}; modeled activation effect is {direction}. "
               "Review the operational recommendation before deciding.")
    return rec, rat


def send_metadata(origin) -> tuple[bool, str]:
    hh = origin.strftime("%H:%M")
    routine = hh in ROUTINE_HOURS
    if routine:
        # routine hour: ROUTINE (or ROUTINE_ONCALL if oncall newly active — handled by caller)
        return True, "ROUTINE"
    return False, "NONE"


def main() -> int:
    load_env()
    dbx = token()

    # 1) data hour = forecast's own origin (authoritative), not box clock.
    try:
        origin = forecast_origin(dbx)
    except Exception as exc:
        print(f"READINESS: cannot read forecast origin: {exc}", file=sys.stderr)
        return 0

    # 2) coverage check (silent skip if already covered).
    hours, ids = covered(dbx)
    hour_key = origin.strftime("%Y-%m-%d %H")
    blurb_id = origin.strftime("%Y%m%d-%H00")
    if hour_key in hours and blurb_id in ids:
        return 0  # already covered -> silent no-op
    # also skip if this exact blurb_id already exists (dedup safety net)
    if blurb_id in ids:
        return 0

    # 3) genuinely new data hour -> build + publish.
    try:
        download_inputs(dbx)
    except Exception as exc:
        print(f"READINESS: input download failed: {exc}", file=sys.stderr)
        return 0

    try:
        facts = run_compute(origin)
    except Exception as exc:
        print(f"READINESS FAILED: {exc}", file=sys.stderr)
        return 0

    blurb = build_blurb(facts)
    rec, rat = oncall_metadata(facts)
    send_rec, send_reason = send_metadata(origin)
    # If routine hour and on-call is newly active (not NOT INDICATED), escalate to ROUTINE_ONCALL.
    if send_reason == "ROUTINE" and rec in ("USE", "CONSIDER"):
        send_reason = "ROUTINE_ONCALL"

    generated_at = pd.Timestamp.now(tz="America/Montreal").strftime("%Y-%m-%d %H:%M:%S")
    data_time = origin.strftime("%Y-%m-%d %H:%M:%S")
    source_status = "ready"

    request = {
        "generated_at_local": generated_at,
        "forecast_data_time_local": data_time,
        "blurb_id": blurb_id,
        "blurb": blurb,
        "oncall_recommendation": rec,
        "oncall_rationale": rat,
        "send_recommended": "true" if send_rec else "false",
        "send_reason": send_reason,
        "source_status": source_status,
    }

    req_path = SCRATCH / "blurb_request.json"
    req_path.write_text(json.dumps(request, indent=2))

    # 4) publish via the repo append worker (current 8-col pipeline).
    env = dict(os.environ)
    env["ED_FLOW_REPO"] = str(REPO)
    try:
        proc = subprocess.run(
            [str(VENV_PY), str(APPEND_WORKER), "--request", str(req_path)],
            cwd=str(SCRATCH), env=env, timeout=180, capture_output=True, text=True,
        )
    except subprocess.TimeoutExpired:
        print("TIMEOUT: append worker did not finish in 180s", file=sys.stderr)
        return 0

    if proc.returncode != 0:
        print(f"PUBLISH_FAILED rc={proc.returncode}: {(proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout) else 'no output'}", file=sys.stderr)
        return 0

    out = (proc.stdout or "").strip()
    if "duplicate_skipped" in out:
        # The worker already had this row -> silent no-op.
        return 0

    # A row was appended -> brief announcement (low-frequency, ~<=1x/hour).
    print(f"PUBLISHED blurb_id={blurb_id} data_hour={hour_key} send_reason={send_reason} oncall={rec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
