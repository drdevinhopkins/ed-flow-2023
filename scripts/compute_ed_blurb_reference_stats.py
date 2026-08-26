#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone

import dropbox
import pandas as pd
import requests
from dropbox.files import WriteMode

SOURCE_PATH = os.getenv("DROPBOX_BLURB_HISTORY_PATH", "/allDataWithCalculatedColumns.csv")
OUTPUT_PATH = os.getenv("DROPBOX_BLURB_REFERENCE_STATS_PATH", "/blurb_reference_stats.json")
LOOKBACK_DAYS = int(os.getenv("BLURB_REFERENCE_LOOKBACK_DAYS", "730"))


def get_dbx() -> dropbox.Dropbox:
    key = os.environ.get("DROPBOX_APP_KEY")
    secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([key, secret, refresh]):
        raise RuntimeError("DROPBOX_APP_KEY, DROPBOX_APP_SECRET and DROPBOX_REFRESH_TOKEN are required")
    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": key,
            "client_secret": secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return dropbox.Dropbox(response.json()["access_token"], timeout=120)


def find_col(frame: pd.DataFrame, *names: str) -> str | None:
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        hit = lookup.get(name.lower())
        if hit is not None:
            return hit
    return None


def numeric(frame: pd.DataFrame, name: str) -> pd.Series | None:
    col = find_col(frame, name)
    if col is None:
        return None
    return pd.to_numeric(frame[col], errors="coerce")


def sum_components(frame: pd.DataFrame, names: list[str]) -> pd.Series | None:
    pieces: list[pd.Series] = []
    for name in names:
        series = numeric(frame, name)
        if series is None:
            return None
        pieces.append(series)
    return pd.concat(pieces, axis=1).sum(axis=1, min_count=len(pieces))


def add_flow_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    total = numeric(out, "Total_TBS")
    if total is None:
        total = numeric(out, "total_tbs")
    if total is None:
        total = sum_components(
            out,
            [
                "TRG_HALLWAY_TBS",
                "POD_GREEN_TBS",
                "POD_YELLOW_TBS",
                "POD_ORANGE_TBS",
                "RAZ_TBS",
                "AMBVERTTBS",
                "QTrack_TBS",
                "Garage_TBS",
            ],
        )
    if total is None:
        raise RuntimeError("Could not derive Total_TBS from historical data")
    out["__Total_TBS"] = total

    pod = numeric(out, "POD_TBS")
    if pod is None:
        pod = sum_components(
            out,
            ["TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS"],
        )
    if pod is None:
        raise RuntimeError("Could not derive POD_TBS from historical data")
    out["__POD_TBS"] = pod

    vertical = numeric(out, "Vertical_TBS")
    if vertical is None:
        vertical = sum_components(out, ["RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"])
    if vertical is None:
        raise RuntimeError("Could not derive Vertical_TBS from historical data")
    out["__Vertical_TBS"] = vertical
    return out


def quantiles(series: pd.Series) -> dict[str, float | int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    q = s.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        "n": int(len(s)),
        "p10": round(float(q.loc[0.10]), 1),
        "p25": round(float(q.loc[0.25]), 1),
        "p50": round(float(q.loc[0.50]), 1),
        "p75": round(float(q.loc[0.75]), 1),
        "p90": round(float(q.loc[0.90]), 1),
        "p95": round(float(q.loc[0.95]), 1),
    }


def usable_group_stats(grouped, value_col: str, min_n: int = 30) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label, group in grouped:
        stats = quantiles(group[value_col])
        if int(stats.get("n", 0)) >= min_n:
            result[str(label)] = stats
    return result


def build_stats(frame: pd.DataFrame) -> dict:
    ds_col = find_col(frame, "ds")
    if ds_col is None:
        raise RuntimeError("Historical data has no ds timestamp column")

    data = frame.copy()
    data["__ds"] = pd.to_datetime(data[ds_col], errors="coerce")
    data = data.dropna(subset=["__ds"]).sort_values("__ds").drop_duplicates("__ds", keep="last")
    data = add_flow_metrics(data)

    max_ds = pd.Timestamp(data["__ds"].max()).floor("h")
    start = max_ds - pd.Timedelta(days=LOOKBACK_DAYS)
    recent = data[data["__ds"].between(start, max_ds)].copy()

    midnight = recent[recent["__ds"].dt.hour.eq(0)].copy()
    midnight = midnight.dropna(subset=["__Total_TBS"])
    midnight["prior_evening_day"] = (midnight["__ds"] - pd.Timedelta(days=1)).dt.day_name()
    midnight["prior_evening_day_type"] = midnight["prior_evening_day"].isin(["Saturday", "Sunday"]).map(
        {True: "weekend", False: "weekday"}
    )

    evening = recent[recent["__ds"].dt.hour.between(15, 23)].copy()
    evening["vertical_minus_pod"] = evening["__Vertical_TBS"] - evening["__POD_TBS"]
    evening["vertical_to_pod_ratio"] = (evening["__Vertical_TBS"] + 1.0) / (evening["__POD_TBS"] + 1.0)

    stats = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": SOURCE_PATH,
        "reference_window": {
            "lookback_days": LOOKBACK_DAYS,
            "start": start.isoformat(),
            "end": max_ds.isoformat(),
        },
        "midnight_total_tbs": {
            "overall": quantiles(midnight["__Total_TBS"]),
            "by_prior_evening_day_type": usable_group_stats(
                midnight.groupby("prior_evening_day_type"), "__Total_TBS", min_n=30
            ),
            "by_prior_evening_day": usable_group_stats(
                midnight.groupby("prior_evening_day"), "__Total_TBS", min_n=30
            ),
            "plain_language_bands": {
                "light": "below p25",
                "typical": "p25 through p75",
                "heavy": "above p75 through p90",
                "very_heavy": "above p90",
            },
            "grouping_note": "For an upcoming midnight, prefer the prior-evening day-of-week group when available; otherwise use weekday/weekend, then overall.",
        },
        "evening_vertical_vs_pod": {
            "hours_local": "15:00-23:00",
            "vertical_minus_pod": quantiles(evening["vertical_minus_pod"]),
            "vertical_to_pod_ratio": quantiles(evening["vertical_to_pod_ratio"]),
            "suggested_reassignment_gate": {
                "minimum_absolute_difference": 5,
                "minimum_ratio": 1.5,
                "historical_context": "Prefer a difference at or above the historical p75; call it a marked imbalance at or above p90.",
                "persistence": "Prefer current plus at least one near-term forecast hour, unless the current imbalance is extreme.",
                "guardrail": "Do not recommend moving a POD/overlap physician if POD itself is under substantial pressure or if the relevant physician is not available.",
            },
        },
    }
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dbx = get_dbx()
    metadata, response = dbx.files_download(SOURCE_PATH)
    frame = pd.read_csv(io.BytesIO(response.content), low_memory=False)
    stats = build_stats(frame)

    payload = json.dumps(stats, indent=2, sort_keys=True) + "\n"
    with open("blurb_reference_stats.json", "w", encoding="utf-8") as handle:
        handle.write(payload)

    print(payload)
    if not args.dry_run:
        result = dbx.files_upload(payload.encode("utf-8"), OUTPUT_PATH, mode=WriteMode.overwrite, mute=True)
        print(f"Updated Dropbox reference stats: {OUTPUT_PATH} rev={result.rev}")
    else:
        print("Dry run: Dropbox reference stats were not modified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
