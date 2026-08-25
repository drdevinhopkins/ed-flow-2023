#!/usr/bin/env python3
"""Generate paired prospective hourly forecasts with and without weather routing.

This runner is deliberately non-production. It never writes ``forecast-v2.csv`` and never
uses the production Dropbox filename. Instead it archives a paired forecast table plus the
exact weather-source slice used by the shadow forecast so weather routes can be scored later
without hindsight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import dropbox
import pandas as pd
import requests
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import backtest_covariate_ablation as base  # noqa: E402
import hourly_forecast_v2 as forecast_v2  # noqa: E402

DEFAULT_OUTPUT_DIR = Path("validation/prospective-weather/latest-run")
DROPBOX_ROOT = "/validation/prospective-weather"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-dropbox", action="store_true")
    return parser.parse_args()


def _future_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    future = frame.loc[frame["row_type"].eq("forecast")].copy()
    keep = [
        "ds",
        "target_name",
        "forecast",
        "forecast_lower",
        "forecast_upper",
        "horizon_hour",
        "horizon_band",
        "scenario",
        "feature_family",
        "forecast_origin",
        "generated_at_utc",
        "routing_version",
        "weather_routing_enabled",
    ]
    future = future[keep].rename(
        columns={
            "forecast": f"{label}_prediction",
            "forecast_lower": f"{label}_lower",
            "forecast_upper": f"{label}_upper",
            "scenario": f"{label}_scenario",
            "feature_family": f"{label}_feature_family",
            "weather_routing_enabled": f"{label}_weather_routing_enabled",
        }
    )
    return future


def _capture_weather_run(
    pipeline: Chronos2Pipeline,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run weather routing while retaining the exact raw weather dataframe it consumed."""
    captured: dict[str, pd.DataFrame] = {}
    original = forecast_v2.weather_bt.load_weather

    def capture(source: str) -> pd.DataFrame:
        frame = original(source)
        captured["weather"] = frame.copy(deep=True)
        return frame

    forecast_v2.weather_bt.load_weather = capture
    try:
        forecast = forecast_v2.build_forecast_v2(pipeline, allow_weather=True)
    finally:
        forecast_v2.weather_bt.load_weather = original

    if "weather" not in captured:
        raise RuntimeError("Weather input capture failed")
    return forecast, captured["weather"]


def _weather_snapshot(weather: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    start = cutoff - pd.Timedelta(days=7)
    end = cutoff + pd.Timedelta(hours=forecast_v2.HORIZON)
    snapshot = weather.loc[weather["ds"].between(start, end)].copy()
    expected = pd.date_range(
        cutoff + pd.Timedelta(hours=1), periods=forecast_v2.HORIZON, freq="h"
    )
    available = set(snapshot["ds"])
    missing = [stamp for stamp in expected if stamp not in available]
    if missing:
        raise RuntimeError(
            f"Prospective weather snapshot is missing future hours: {missing[:6]}"
        )
    return snapshot.reset_index(drop=True)


def _snapshot_id(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _dropbox_client() -> dropbox.Dropbox:
    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([app_key, app_secret, refresh_token]):
        raise RuntimeError("Dropbox credentials are required unless --no-dropbox is used")

    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return dropbox.Dropbox(response.json()["access_token"])


def _upload_file(dbx: dropbox.Dropbox, local_path: Path, remote_path: str) -> None:
    with local_path.open("rb") as handle:
        dbx.files_upload(
            handle.read(),
            remote_path,
            mode=dropbox.files.WriteMode.overwrite,
            mute=True,
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {base.MODEL_ID} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        base.MODEL_ID, device_map=device
    )

    # Paired predictions from the same code revision. The safe run never uses weather;
    # the shadow run allows the existing weather-winning routes and captures its inputs.
    safe = forecast_v2.build_forecast_v2(pipeline, allow_weather=False)
    weather, weather_used = _capture_weather_run(pipeline)

    safe_future = _future_rows(safe, "baseline")
    weather_future = _future_rows(weather, "weather")
    join_keys = ["ds", "target_name", "horizon_hour", "horizon_band", "forecast_origin"]
    paired = safe_future.merge(
        weather_future,
        on=join_keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_baseline", "_weather"),
    )

    expected_rows = len(forecast_v2.FLOW_TARGETS) * forecast_v2.HORIZON
    if len(paired) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} paired rows; got {len(paired)}")

    cutoff = pd.Timestamp(paired["forecast_origin"].iloc[0]).floor("h")
    if not paired["forecast_origin"].eq(cutoff).all():
        raise RuntimeError("Paired runs did not use one common forecast origin")

    snapshot = _weather_snapshot(weather_used, cutoff)
    snapshot_id = _snapshot_id(snapshot)
    issued_at = pd.Timestamp.now(tz="UTC")
    run_id = f"{issued_at.strftime('%Y%m%dT%H%M%SZ')}-{snapshot_id}"

    paired.insert(0, "forecast_run_id", run_id)
    paired["forecast_issued_at"] = issued_at.isoformat()
    paired["weather_snapshot_id"] = snapshot_id
    paired["weather_snapshot_is_prospective"] = True
    paired["model_family"] = "chronos2"
    paired["model_version"] = base.MODEL_ID
    paired["git_sha"] = os.environ.get("GITHUB_SHA", "unknown")
    paired["data_cutoff_ds"] = cutoff

    snapshot.insert(0, "forecast_run_id", run_id)
    snapshot.insert(1, "weather_snapshot_id", snapshot_id)
    snapshot.insert(2, "forecast_issued_at", issued_at.isoformat())
    snapshot.insert(3, "forecast_origin", cutoff)
    snapshot.insert(4, "weather_snapshot_is_prospective", True)

    forecast_path = args.output_dir / "shadow-weather-forecast.csv"
    weather_path = args.output_dir / "weather-snapshot.csv"
    metadata_path = args.output_dir / "metadata.json"
    paired.to_csv(forecast_path, index=False)
    snapshot.to_csv(weather_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "forecast_run_id": run_id,
                "forecast_origin": str(cutoff),
                "forecast_issued_at": issued_at.isoformat(),
                "weather_snapshot_id": snapshot_id,
                "weather_snapshot_is_prospective": True,
                "paired_rows": len(paired),
                "targets": sorted(paired["target_name"].unique().tolist()),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    if not args.no_dropbox:
        dbx = _dropbox_client()
        remote_dir = f"{DROPBOX_ROOT}/{run_id}"
        _upload_file(dbx, forecast_path, f"{remote_dir}/{forecast_path.name}")
        _upload_file(dbx, weather_path, f"{remote_dir}/{weather_path.name}")
        _upload_file(dbx, metadata_path, f"{remote_dir}/{metadata_path.name}")
        # Convenience pointers; historical run folders remain immutable by convention.
        _upload_file(dbx, forecast_path, f"{DROPBOX_ROOT}/latest-shadow-weather-forecast.csv")
        _upload_file(dbx, metadata_path, f"{DROPBOX_ROOT}/latest-metadata.json")

    print(f"Shadow weather run {run_id}: {len(paired)} paired forecast rows")
    print(paired.groupby(["target_name", "weather_scenario"]).size().rename("rows"))


if __name__ == "__main__":
    main()
