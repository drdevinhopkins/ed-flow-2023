#!/usr/bin/env python3
"""Archive leakage-safe Montréal peer-ED pressure features to Dropbox.

The MSSS public source is a rolling seven-day file.  Each run re-downloads that window,
rebuilds the peer-only features, appends/revises overlapping hours in the canonical
Dropbox history, and overwrites the canonical CSV.  Frequent runs therefore tolerate
missed jobs while preserving a growing historical series for future backtests.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import dropbox
import pandas as pd
import requests

from regional_ed_pressure import (
    DEFAULT_JGH_PATTERNS,
    DEFAULT_REGION_CODE,
    DEFAULT_SOURCE_URL,
    build_regional_peer_pressure,
    load_public_feed,
)
from utils import upload

DEFAULT_DROPBOX_PATH = "/regional_ed_pressure/regional_ed_pressure_history.csv"
DEFAULT_OUTPUT = Path("regional_ed_pressure_history.csv")


def dropbox_client() -> dropbox.Dropbox:
    key = os.environ.get("DROPBOX_APP_KEY")
    secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([key, secret, refresh]):
        raise RuntimeError("Dropbox refresh-token credentials are required")
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
    return dropbox.Dropbox(response.json()["access_token"])


def download_existing(dbx: dropbox.Dropbox, path: str) -> pd.DataFrame | None:
    try:
        _, response = dbx.files_download(path)
    except dropbox.exceptions.ApiError as exc:
        error = exc.error
        is_missing = (
            hasattr(error, "is_path")
            and error.is_path()
            and error.get_path().is_not_found()
        )
        if is_missing:
            return None
        raise
    frame = pd.read_csv(io.BytesIO(response.content))
    if "ds" not in frame:
        raise ValueError(f"Existing Dropbox archive {path} has no ds column")
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    return frame.dropna(subset=["ds"])


def merge_history(existing: pd.DataFrame | None, recent: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        combined = recent.copy()
    else:
        combined = pd.concat([existing, recent], ignore_index=True, sort=False)
    combined["ds"] = pd.to_datetime(combined["ds"], errors="coerce")
    combined = combined.dropna(subset=["ds"])
    return (
        combined.sort_values("ds")
        .drop_duplicates(subset=["ds"], keep="last")
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--region-code", default=DEFAULT_REGION_CODE)
    parser.add_argument("--dropbox-path", default=DEFAULT_DROPBOX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jgh-pattern",
        action="append",
        default=None,
        help="Installation-name substring identifying JGH; repeat to supply several.",
    )
    parser.add_argument(
        "--max-source-age-hours",
        type=float,
        default=4.0,
        help="Fail if the newest public-feed hour is older than this threshold.",
    )
    parser.add_argument("--no-dropbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patterns = tuple(args.jgh_pattern) if args.jgh_pattern else DEFAULT_JGH_PATTERNS
    raw = load_public_feed(args.source)
    recent = build_regional_peer_pressure(
        raw,
        region_code=args.region_code,
        jgh_patterns=patterns,
    )
    if recent.empty:
        raise ValueError("Regional pressure collector produced no rows")

    newest = pd.Timestamp(recent["ds"].max())
    now_local = pd.Timestamp.now(tz="America/Montreal").tz_localize(None)
    source_age_hours = (now_local - newest).total_seconds() / 3600.0
    if source_age_hours > args.max_source_age_hours:
        raise ValueError(
            f"MSSS regional ED source is stale: newest={newest}, age={source_age_hours:.1f}h"
        )

    existing = None
    dbx = None
    if not args.no_dropbox:
        dbx = dropbox_client()
        existing = download_existing(dbx, args.dropbox_path)

    combined = merge_history(existing, recent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(
        f"Regional ED pressure: source_rows={len(raw)} recent_hours={len(recent)} "
        f"archive_hours={len(combined)} newest={newest} age_hours={source_age_hours:.2f}",
        flush=True,
    )

    if dbx is not None:
        normalized = args.dropbox_path.strip("/")
        folder, _, name = normalized.rpartition("/")
        result = upload(dbx, str(args.output), folder, "", name, overwrite=True)
        if result is None:
            raise RuntimeError(f"Dropbox upload failed for {args.dropbox_path}")
        print(f"Uploaded canonical archive to {args.dropbox_path}", flush=True)


if __name__ == "__main__":
    main()
