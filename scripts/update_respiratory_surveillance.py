#!/usr/bin/env python3
"""Build/update the Montréal INSPQ respiratory-surveillance history in Dropbox."""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import dropbox
import pandas as pd
import requests

from respiratory_surveillance import INSPQ_INDEX_URL, fetch_report, fetch_report_index
from utils import upload

DEFAULT_DROPBOX_PATH = "/respiratory_surveillance/inspq_montreal_weekly.csv"
DEFAULT_OUTPUT = Path("inspq_montreal_weekly.csv")


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
    if "available_date" not in frame:
        raise ValueError(f"Existing respiratory archive {path} has no available_date")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    return frame.dropna(subset=["available_date"]).copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-url", default=INSPQ_INDEX_URL)
    parser.add_argument("--dropbox-path", default=DEFAULT_DROPBOX_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--refresh-latest",
        type=int,
        default=2,
        help="Re-fetch this many newest reports to capture provisional revisions.",
    )
    parser.add_argument(
        "--max-report-age-days",
        type=int,
        default=21,
        help="Fail if the latest available report is older than this threshold.",
    )
    parser.add_argument("--no-dropbox", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.refresh_latest < 0:
        raise ValueError("refresh-latest must be >= 0")

    dbx = None if args.no_dropbox else dropbox_client()
    existing = None if dbx is None else download_existing(dbx, args.dropbox_path)
    if existing is None:
        existing = pd.DataFrame()

    reports = fetch_report_index(args.index_url)
    existing_urls = set(existing.get("source_url", pd.Series(dtype=str)).dropna().astype(str))
    refresh_urls = {
        report.url for report in reports[-args.refresh_latest :]
    } if args.refresh_latest else set()
    pending = [
        report for report in reports if report.url not in existing_urls or report.url in refresh_urls
    ]

    records: list[dict[str, object]] = []
    errors: list[str] = []
    for report in pending:
        try:
            print(f"Fetching INSPQ respiratory report: {report.label} {report.url}", flush=True)
            records.append(fetch_report(report))
        except Exception as exc:  # Keep other weeks usable if one archived PDF is transiently bad.
            message = f"{report.url}: {type(exc).__name__}: {exc}"
            print(f"WARNING: {message}", flush=True)
            errors.append(message)

    refreshed = pd.DataFrame.from_records(records)
    if existing.empty:
        combined = refreshed
    elif refreshed.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, refreshed], ignore_index=True, sort=False)

    if combined.empty:
        raise ValueError("No respiratory surveillance reports were available")
    combined["available_date"] = pd.to_datetime(combined["available_date"], errors="coerce")
    combined = (
        combined.dropna(subset=["available_date"])
        .sort_values("available_date")
        .drop_duplicates(subset=["available_date"], keep="last")
        .reset_index(drop=True)
    )

    newest = pd.Timestamp(combined["available_date"].max()).normalize()
    today = pd.Timestamp.now(tz="America/Montreal").tz_localize(None).normalize()
    age_days = int((today - newest).days)
    if age_days > args.max_report_age_days:
        raise ValueError(
            f"Respiratory surveillance archive is stale: newest={newest.date()}, age={age_days}d"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(
        f"Respiratory surveillance: discovered={len(reports)} fetched={len(refreshed)} "
        f"archive_weeks={len(combined)} newest={newest.date()} age_days={age_days} "
        f"fetch_errors={len(errors)}",
        flush=True,
    )

    if dbx is not None:
        normalized = args.dropbox_path.strip("/")
        folder, _, name = normalized.rpartition("/")
        result = upload(dbx, str(args.output), folder, "", name, overwrite=True)
        if result is None:
            raise RuntimeError(f"Dropbox upload failed for {args.dropbox_path}")
        print(f"Uploaded canonical archive to {args.dropbox_path}", flush=True)

    # Do not fail a successful archive refresh solely because one historical link failed.
    # The warning list remains visible in Actions logs and that report will be retried later.


if __name__ == "__main__":
    main()
