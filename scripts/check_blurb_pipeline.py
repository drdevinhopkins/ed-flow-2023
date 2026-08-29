#!/usr/bin/env python3
"""Read-only health check for the Dropbox hourly blurb pipeline."""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/opt/apps/ed-flow-2023/scripts/automation")
import blurb_automation_wrapper as wrapper


def main() -> int:
    wrapper.load_env()
    dbx = wrapper.token()
    failures: list[str] = []

    fc = pd.read_csv(io.BytesIO(wrapper.download(dbx, "/forecast-v2.1.csv")))
    origin = pd.to_datetime(fc["forecast_origin"]).max()
    origin = origin.tz_localize("America/Montreal") if origin.tzinfo is None else origin.tz_convert("America/Montreal")
    current = pd.read_csv(io.BytesIO(wrapper.download(dbx, "/current.csv")))
    current_hour = pd.to_datetime(current["ds"]).max().tz_localize("America/Montreal")

    raw_log = wrapper.download(dbx, "/hourly_forecast_blurbs.csv").decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(raw_log)))
    ids = [r.get("blurb_id", "").strip() for r in rows]
    duplicates = sorted({x for x in ids if x and ids.count(x) > 1})
    if duplicates:
        failures.append(f"duplicate blurb_ids: {duplicates}")
    if current_hour != origin:
        failures.append(f"current hour {current_hour} != forecast origin {origin}")
    blurb_id = origin.strftime("%Y%m%d-%H00")
    if blurb_id not in ids:
        failures.append(f"latest forecast {blurb_id} has no published blurb")

    print(f"forecast_origin={origin}")
    print(f"current_hour={current_hour}")
    print(f"blurb_rows={len(rows)}")
    print(f"latest_blurb_present={blurb_id in ids}")
    if failures:
        print("HEALTH=DEGRADED")
        for failure in failures:
            print(f"FAILURE={failure}")
        return 1
    print("HEALTH=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
