#!/usr/bin/env python3
"""Generate append-only prospective shadow forecasts for candidate ED flow metrics.

This is deliberately non-production. It forecasts the five candidate targets with both
history-only baseline and the pre-registered robustness-aware candidate route, while
keeping the full 13-target Chronos-2 bundle used during validation. It never writes a
production forecast filename or changes production routing.
"""

from __future__ import annotations

import argparse
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
import backtest_hourly_final_features as final_bt  # noqa: E402
import backtest_hourly_weather_features as weather_bt  # noqa: E402
import backtest_staffing_features as staffing_bt  # noqa: E402
from candidate_flow_metrics import CANDIDATE_TARGETS  # noqa: E402
from evaluation.backtests import backtest_candidate_metrics_cutoff as candidate_bt  # noqa: E402
from evaluation.prospective.candidate_metric_routes import (  # noqa: E402
    horizon_band,
    scenario_for,
    scenarios_needed,
)
from staffing_features import build_schedule_feature_frames  # noqa: E402

HORIZON = 24
MAX_HISTORY_DAYS = 365
EFFECT_MIN_HOURS = 24
EFFECT_SHRINKAGE_HOURS = 72.0
DEFAULT_OUTPUT_DIR = Path("validation/prospective-candidates/latest-run")
DROPBOX_ROOT = "/validation/prospective-candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-dropbox", action="store_true")
    return parser.parse_args()


def _predict(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
) -> pd.DataFrame:
    kwargs: dict[str, object] = {
        "prediction_length": HORIZON,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": list(candidate_bt.TARGETS),
        "quantile_levels": [0.2, 0.5, 0.8],
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs).copy()
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    for column in ["0.2", "0.8"]:
        if column not in result.columns:
            result[column] = result["predictions"]
    result["ds"] = pd.to_datetime(result["ds"], format="mixed", errors="coerce")
    return result[["ds", "target_name", "predictions", "0.2", "0.8"]]


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


def _upload(dbx: dropbox.Dropbox, local: Path, remote: str) -> None:
    with local.open("rb") as handle:
        dbx.files_upload(
            handle.read(), remote, mode=dropbox.files.WriteMode.overwrite, mute=True
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Freeze all live inputs once so baseline and routed forecasts are exactly paired.
    flow = candidate_bt.load_flow()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)
    cutoff = pd.Timestamp(flow["ds"].max()).floor("h")
    calendar = final_bt.build_calendar_frame(flow, [cutoff], HORIZON)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        final_bt.MODEL_ID, device_map=device
    )

    needed = tuple(sorted(set(scenarios_needed()) | {"baseline"}))
    forecasts: dict[str, pd.DataFrame] = {}
    for scenario in needed:
        print(f"Candidate shadow forecast: scenario={scenario}")
        history, future = candidate_bt.scenario_frames(
            scenario,
            flow=flow,
            staffing=staffing,
            weather=weather,
            shifts=shifts,
            schedule_frames=schedule_frames,
            calendar=calendar,
            cutoff=cutoff,
            horizon=HORIZON,
            max_history_days=MAX_HISTORY_DAYS,
            effect_min_hours=EFFECT_MIN_HOURS,
            effect_shrinkage_hours=EFFECT_SHRINKAGE_HOURS,
        )
        forecasts[scenario] = _predict(pipeline, history, future)

    issued_at = pd.Timestamp.now(tz="UTC")
    run_id = f"{issued_at.strftime('%Y%m%dT%H%M%SZ')}-{os.environ.get('GITHUB_RUN_ID', 'local')}"
    expected_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=HORIZON, freq="h")
    rows: list[dict[str, object]] = []

    for target in CANDIDATE_TARGETS:
        for horizon_hour, stamp in enumerate(expected_hours, start=1):
            route = scenario_for(target, horizon_hour)
            baseline_match = forecasts["baseline"].loc[
                forecasts["baseline"]["target_name"].eq(target)
                & forecasts["baseline"]["ds"].eq(stamp)
            ]
            routed_match = forecasts[route].loc[
                forecasts[route]["target_name"].eq(target)
                & forecasts[route]["ds"].eq(stamp)
            ]
            if len(baseline_match) != 1 or len(routed_match) != 1:
                raise RuntimeError(
                    f"Expected one paired candidate row for {target} h={horizon_hour}; "
                    f"baseline={len(baseline_match)} routed={len(routed_match)}"
                )
            b = baseline_match.iloc[0]
            r = routed_match.iloc[0]
            rows.append(
                {
                    "forecast_run_id": run_id,
                    "forecast_issued_at": issued_at.isoformat(),
                    "forecast_origin": cutoff,
                    "target_ds": stamp,
                    "target_name": target,
                    "horizon_hour": horizon_hour,
                    "horizon_band": horizon_band(horizon_hour),
                    "baseline_prediction": float(b["predictions"]),
                    "baseline_lower": float(b["0.2"]),
                    "baseline_upper": float(b["0.8"]),
                    "candidate_prediction": float(r["predictions"]),
                    "candidate_lower": float(r["0.2"]),
                    "candidate_upper": float(r["0.8"]),
                    "candidate_scenario": route,
                    "model_family": "chronos2",
                    "model_version": final_bt.MODEL_ID,
                    "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
                    "data_cutoff_ds": cutoff,
                }
            )

    paired = pd.DataFrame(rows)
    expected_rows = len(CANDIDATE_TARGETS) * HORIZON
    if len(paired) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} candidate rows; got {len(paired)}")
    keys = ["forecast_run_id", "target_name", "target_ds", "horizon_hour"]
    if paired[keys].duplicated().any():
        raise RuntimeError("Duplicate candidate shadow forecast rows")
    if set(paired["target_name"]) != set(CANDIDATE_TARGETS):
        raise RuntimeError("Candidate target set mismatch")

    forecast_path = args.output_dir / "shadow-candidate-forecast.csv"
    metadata_path = args.output_dir / "metadata.json"
    paired.to_csv(forecast_path, index=False)
    metadata_path.write_text(
        json.dumps(
            {
                "forecast_run_id": run_id,
                "forecast_origin": str(cutoff),
                "forecast_issued_at": issued_at.isoformat(),
                "paired_rows": len(paired),
                "targets": list(CANDIDATE_TARGETS),
                "scenarios_generated": list(needed),
                "production_routing_changed": False,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    if not args.no_dropbox:
        dbx = _dropbox_client()
        remote_dir = f"{DROPBOX_ROOT}/{run_id}"
        _upload(dbx, forecast_path, f"{remote_dir}/{forecast_path.name}")
        _upload(dbx, metadata_path, f"{remote_dir}/{metadata_path.name}")
        _upload(dbx, forecast_path, f"{DROPBOX_ROOT}/latest-shadow-candidate-forecast.csv")
        _upload(dbx, metadata_path, f"{DROPBOX_ROOT}/latest-metadata.json")

    print(f"Candidate shadow run {run_id}: {len(paired)} paired rows")
    print(paired.groupby(["target_name", "candidate_scenario"]).size().rename("rows"))


if __name__ == "__main__":
    main()
