#!/usr/bin/env python3
"""Export VED1/VED2 ShiftAdmin shifts for a requested date range."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

API_URL = "https://www.shiftadmin.com/vjgh/org_scheduled_shifts"
VED_TYPES = {"VED1", "VED2"}
DEFAULT_START = date(2026, 1, 1)
DEFAULT_END = date(2027, 1, 31)


def parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}; expected YYYY-MM-DD"
        ) from exc


def fetch_shifts(start_date: date, end_date: date) -> list[dict[str, object]]:
    # Support both the repository runtime and the shared Hermes environment.
    load_dotenv()
    load_dotenv("/opt/data/.env")
    username = os.getenv("SHIFTADMIN_USER")
    password = os.getenv("SHIFTADMIN_PASS")
    if not username or not password:
        raise RuntimeError(
            "SHIFTADMIN_USER and SHIFTADMIN_PASS must be available in the environment"
        )

    response = requests.post(
        API_URL,
        json={"start_date": str(start_date), "end_date": str(end_date)},
        auth=(username, password),
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("ShiftAdmin API response was not a list")
    return payload


def shift_datetime(row: dict[str, object], field: str) -> datetime:
    value = row.get(field)
    if not isinstance(value, str):
        raise RuntimeError(f"shift row is missing {field}")
    for format_string in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(value[:19], format_string)
        except ValueError:
            continue
    raise RuntimeError(f"unsupported {field} format: {value!r}")


def export_ved_shifts(
    rows: list[dict[str, object]], output_path: Path
) -> tuple[int, dict[str, int]]:
    ved_rows = [
        row for row in rows if row.get("shift_short_name") in VED_TYPES
    ]
    ved_rows.sort(
        key=lambda row: (
            shift_datetime(row, "shift_start"),
            shift_datetime(row, "shift_end"),
            str(row.get("shift_short_name", "")),
            str(row.get("scheduled_shift_id", "")),
        )
    )
    if not ved_rows:
        raise RuntimeError("No VED1/VED2 shifts were returned for the requested range")

    fields = list(ved_rows[0].keys())
    if not fields:
        raise RuntimeError("Shift rows did not contain any fields")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ved_rows)

    counts = {shift_type: sum(row.get("shift_short_name") == shift_type for row in ved_rows)
              for shift_type in sorted(VED_TYPES)}
    return len(ved_rows), counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=parse_date, default=DEFAULT_END)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ved1_ved2_shifts_2026_2027.csv"),
    )
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must not be earlier than --start")

    rows = fetch_shifts(args.start, args.end)
    count, by_type = export_ved_shifts(rows, args.output)
    print(f"wrote {count} VED shifts to {args.output}")
    print(f"counts: {by_type}")
    print(f"requested range: {args.start} through {args.end} inclusive")


if __name__ == "__main__":
    main()
