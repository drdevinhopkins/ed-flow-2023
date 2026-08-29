#!/usr/bin/env python3
"""Run the frozen intraday day-completion candidate without publishing it."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
import sklearn

BACKTEST_DIR = Path(__file__).resolve().parents[1] / "backtests"
sys.path.insert(0, str(BACKTEST_DIR))

from backtest_intraday_day_completion import (  # noqa: E402
    FLOW_URL,
    LOCAL_TZ,
    TOTAL_TBS_COMPONENTS,
    _fit_quantile_models,
    _predict_remaining_quantiles,
    add_curve_features,
    apply_quantile_corrections,
    build_snapshots,
    build_weather_features,
    expected_local_hours,
    feature_sets,
    fit_completion_curve,
    fit_prior_update,
    fit_quantile_corrections,
    load_hourly_flow,
)

MODEL_VERSION = "intraday-ensemble-v1-2026-08-28"
MODEL_FINGERPRINT_VERSION = "functional-probe-v2"
FALLBACK_VERSION = "intraday-prior-update-fallback-v1"
OPERATIONAL_HOURS = range(11, 19)


class DataQualityError(RuntimeError):
    """Expected input failure that must suppress, rather than fabricate, output."""


def validate_live_flow(
    flow: pd.DataFrame,
    *,
    now: pd.Timestamp,
    max_age_minutes: int = 90,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    latest_ds = pd.Timestamp(flow["ds"].max())
    if latest_ds.tzinfo is not None:
        latest_ds = latest_ds.tz_convert(LOCAL_TZ).tz_localize(None)
    local_now = pd.Timestamp(now)
    if local_now.tzinfo is None:
        local_now = local_now.tz_localize(LOCAL_TZ)
    local_now = local_now.tz_convert(LOCAL_TZ).tz_localize(None)
    age_minutes = (local_now - latest_ds).total_seconds() / 60.0
    if age_minutes < -5:
        raise DataQualityError(f"latest flow timestamp is {abs(age_minutes):.0f} minutes in the future")
    if age_minutes > max_age_minutes:
        raise DataQualityError(
            f"latest flow timestamp is stale ({age_minutes:.0f} minutes; limit {max_age_minutes})"
        )

    latest_day = latest_ds.normalize()
    current = flow.loc[flow["day"].eq(latest_day)].sort_values(["ds", "_source_order"]).copy()
    actual_hours = current["ds"].dt.hour.astype(int).tolist()
    expected_prefix = expected_local_hours(latest_day)[: len(actual_hours)]
    if actual_hours != expected_prefix:
        raise DataQualityError(
            f"current day has missing or out-of-order hours: expected {expected_prefix}, got {actual_hours}"
        )
    if int(latest_ds.hour) not in OPERATIONAL_HOURS:
        raise DataQualityError(f"cutoff hour {latest_ds.hour:02d}:00 is outside the shadow window")
    if current["Inflow_Total"].isna().any() or current["Inflow_Total"].lt(0).any():
        raise DataQualityError("current-day inflow contains missing or negative values")

    missing_state = [column for column in TOTAL_TBS_COMPONENTS if column not in current.columns]
    if missing_state:
        raise DataQualityError(f"current flow is missing Total_TBS components: {missing_state}")
    latest_state = current.iloc[-1][list(TOTAL_TBS_COMPONENTS)].apply(pd.to_numeric, errors="coerce")
    if latest_state.isna().any() or latest_state.lt(0).any():
        raise DataQualityError("latest Total_TBS components contain missing or negative values")
    return latest_ds, current


def _append_forecast(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        key = ["model_version", "forecast_day", "cutoff_hour"]
        duplicate = existing.merge(new[key], on=key, how="inner")
        if not duplicate.empty:
            return
        new = pd.concat([existing, new], ignore_index=True)
    new.to_csv(path, index=False)


def _append_status(path: Path, status: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([status])
    if path.exists():
        existing = pd.read_csv(path)
        new = pd.concat([existing, new], ignore_index=True)
    new.to_csv(path, index=False)


def write_model_artifact(
    bundle: dict[str, object], artifact_path: Path, manifest_path: Path
) -> tuple[dict[str, object], dict[str, object]]:
    """Persist and reload the frozen bundle, returning the verified copy and manifest."""
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact_path, compress=3)
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    verified = joblib.load(artifact_path)
    manifest = {
        "model_version": bundle["model_version"],
        "source_hash": bundle["source_hash"],
        "artifact_sha256": digest,
        "artifact_bytes": artifact_path.stat().st_size,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "serialization": "joblib",
        "training_start": bundle["training_start"],
        "training_end": bundle["training_end"],
        "training_days": bundle["training_days"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return verified, manifest


def functional_model_fingerprint(bundle: dict[str, object], probe: pd.DataFrame) -> str:
    """Hash stable predictions and calibration state, not nondeterministic pickle bytes."""
    ordered = probe.sort_values(["day", "cutoff_hour"]).groupby(
        "cutoff_hour", group_keys=False
    ).tail(3)
    state_raw = _predict_remaining_quantiles(
        bundle["state_models"], ordered, features=bundle["state_features"]
    )
    state_calibrated = apply_quantile_corrections(
        state_raw,
        ordered["cutoff_hour"].to_numpy(dtype=int),
        bundle["corrections"],
    )
    weather_raw = _predict_remaining_quantiles(
        bundle["weather_models"], ordered, features=bundle["calendar_weather_features"]
    )
    payload = {
        "fingerprint_version": MODEL_FINGERPRINT_VERSION,
        "model_version": bundle["model_version"],
        "training_start": bundle["training_start"],
        "training_end": bundle["training_end"],
        "cutoff_hours": ordered["cutoff_hour"].astype(int).tolist(),
        "state_features": list(bundle["state_features"]),
        "calendar_weather_features": list(bundle["calendar_weather_features"]),
        "state_raw": np.round(state_raw, 10).tolist(),
        "state_calibrated": np.round(state_calibrated, 10).tolist(),
        "weather_raw": np.round(weather_raw, 10).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_fingerprint_against_ledger(
    path: Path,
    *,
    model_version: str,
    fingerprint_version: str,
    training_end: str,
    fingerprint: str,
) -> None:
    """Suppress a refit that changes functionally within a frozen training window."""
    if not path.exists():
        return
    existing = pd.read_csv(path)
    required = {
        "model_version",
        "model_fingerprint_version",
        "training_end",
        "model_fingerprint",
    }
    if not required.issubset(existing.columns):
        return
    reference = existing.loc[
        existing["model_version"].eq(model_version)
        & existing["model_fingerprint_version"].eq(fingerprint_version)
        & existing["training_end"].astype(str).eq(training_end)
        & existing["model_fingerprint"].notna(),
        "model_fingerprint",
    ].astype(str)
    if not reference.empty and not reference.eq(fingerprint).all():
        raise DataQualityError(
            "functional model fingerprint drift for unchanged model and training window"
        )


def run_prior_update_fallback(
    args: argparse.Namespace, model_error: Exception
) -> dict[str, object]:
    """Record a clearly labeled deterministic fallback without candidate intervals."""
    generated_at = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    if generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("UTC")
    flow = load_hourly_flow(args.flow_csv)
    latest_ds, _ = validate_live_flow(
        flow, now=generated_at, max_age_minutes=args.max_age_minutes
    )
    live_day = latest_ds.normalize()
    prepared_flow = flow.copy()
    prepared_flow.loc[prepared_flow["day"].eq(live_day), "is_complete_day"] = True
    snapshots = build_snapshots(prepared_flow, calendar_mode="rich", weather=None)
    train = snapshots.loc[snapshots["day"].lt(live_day)].copy()
    live = snapshots.loc[
        snapshots["day"].eq(live_day) & snapshots["ds"].eq(latest_ds)
    ].copy()
    if len(live) != 1:
        raise DataQualityError(f"fallback expected one live snapshot at {latest_ds}; found {len(live)}")
    if int(train["day"].nunique()) < args.min_train_days:
        raise DataQualityError("fallback has insufficient complete training history")

    hour = int(live["cutoff_hour"].iat[0])
    prior_params = fit_prior_update(train)
    baseline_total = float(
        live["prior_total"].iat[0]
        + prior_params.loc[hour, "beta"] * live["pace_residual"].iat[0]
    )
    row = {
        "generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
        "forecast_day": live_day.date().isoformat(),
        "cutoff_ds_local": latest_ds.isoformat(),
        "cutoff_hour": hour,
        "model_version": FALLBACK_VERSION,
        "candidate_model_version": MODEL_VERSION,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16],
        "artifact_sha256": None,
        "training_start": pd.Timestamp(train["day"].min()).date().isoformat(),
        "training_end": pd.Timestamp(train["day"].max()).date().isoformat(),
        "training_days": int(train["day"].nunique()),
        "observed_arrivals": float(live["cumulative_arrivals"].iat[0]),
        "predicted_total": baseline_total,
        "p10_total": None,
        "p90_total": None,
        "prior_update_baseline": baseline_total,
        "status": "shadow_fallback",
        "fallback_reason": f"{type(model_error).__name__}: {model_error}",
    }
    _append_forecast(args.output_csv, row)
    return row


def run_shadow(args: argparse.Namespace) -> dict[str, object]:
    generated_at = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    if generated_at.tzinfo is None:
        generated_at = generated_at.tz_localize("UTC")
    flow = load_hourly_flow(args.flow_csv)
    latest_ds, _ = validate_live_flow(
        flow, now=generated_at, max_age_minutes=args.max_age_minutes
    )
    weather = build_weather_features(args.weather_csv)

    live_day = latest_ds.normalize()
    training_days = int(flow.loc[flow["is_complete_day"] & flow["day"].lt(live_day), "day"].nunique())
    if training_days < args.min_train_days:
        raise DataQualityError(
            f"only {training_days} complete training days; require {args.min_train_days}"
        )

    prepared_flow = flow.copy()
    prepared_flow.loc[prepared_flow["day"].eq(live_day), "is_complete_day"] = True
    snapshots = build_snapshots(prepared_flow, calendar_mode="rich", weather=weather)
    train = snapshots.loc[snapshots["day"].lt(live_day)].copy()
    live = snapshots.loc[
        snapshots["day"].eq(live_day) & snapshots["ds"].eq(latest_ds)
    ].copy()
    if len(live) != 1:
        raise DataQualityError(f"expected one live snapshot at {latest_ds}; found {len(live)}")
    weather_columns = [column for column in live if column.startswith("weather_current_")]
    if not weather_columns or live[weather_columns].isna().all(axis=None):
        raise DataQualityError("no cutoff-safe weather observation is available within two hours")

    curve = fit_completion_curve(train)
    train = add_curve_features(train, curve)
    live = add_curve_features(live, curve)
    train_days = np.array(sorted(pd.to_datetime(train["day"].unique())))
    calibration_days = min(args.calibration_days, len(train_days) - 28)
    if calibration_days < 7:
        raise DataQualityError("insufficient history for nested calibration")
    calibration_start = train_days[-calibration_days]
    core = train.loc[train["day"].lt(calibration_start)].copy()
    calibration = train.loc[train["day"].ge(calibration_start)].copy()
    inner_curve = fit_completion_curve(core)
    core = add_curve_features(core, inner_curve)
    calibration = add_curve_features(calibration, inner_curve)

    state_features = feature_sets(core)["boosted_state"]
    calibration_models = _fit_quantile_models(
        core, features=state_features, max_iter=args.max_iter, random_state=args.random_state
    )
    calibration_prediction = _predict_remaining_quantiles(
        calibration_models, calibration, features=state_features
    )
    corrections = fit_quantile_corrections(
        calibration["remaining_arrivals"].to_numpy(dtype=float),
        calibration_prediction,
        calibration["cutoff_hour"].to_numpy(dtype=int),
        shrinkage_days=args.calibration_shrinkage_days,
    )

    sets = feature_sets(train)
    calendar_weather_features = sets.get("boosted_calendar_weather")
    if not calendar_weather_features:
        raise DataQualityError("calendar/weather feature route is unavailable")
    state_models = _fit_quantile_models(
        train, features=sets["boosted_state"], max_iter=args.max_iter, random_state=args.random_state
    )
    weather_models = _fit_quantile_models(
        train, features=calendar_weather_features, max_iter=args.max_iter, random_state=args.random_state
    )
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    bundle = {
        "model_version": MODEL_VERSION,
        "source_hash": source_hash,
        "training_start": pd.Timestamp(train["day"].min()).date().isoformat(),
        "training_end": pd.Timestamp(train["day"].max()).date().isoformat(),
        "training_days": int(train["day"].nunique()),
        "state_features": sets["boosted_state"],
        "calendar_weather_features": calendar_weather_features,
        "state_models": state_models,
        "weather_models": weather_models,
        "corrections": corrections,
    }
    bundle, artifact_manifest = write_model_artifact(
        bundle, args.artifact_joblib, args.artifact_manifest_json
    )
    model_fingerprint = functional_model_fingerprint(bundle, train)
    validate_fingerprint_against_ledger(
        args.output_csv,
        model_version=MODEL_VERSION,
        fingerprint_version=MODEL_FINGERPRINT_VERSION,
        training_end=bundle["training_end"],
        fingerprint=model_fingerprint,
    )
    artifact_manifest["model_fingerprint"] = model_fingerprint
    artifact_manifest["model_fingerprint_version"] = MODEL_FINGERPRINT_VERSION
    args.artifact_manifest_json.write_text(json.dumps(artifact_manifest, indent=2) + "\n")
    state_raw = _predict_remaining_quantiles(
        bundle["state_models"], live, features=bundle["state_features"]
    )
    state_calibrated = apply_quantile_corrections(
        state_raw, live["cutoff_hour"].to_numpy(dtype=int), bundle["corrections"]
    )[0]
    weather_raw = _predict_remaining_quantiles(
        bundle["weather_models"], live, features=bundle["calendar_weather_features"]
    )[0]
    observed = float(live["cumulative_arrivals"].iat[0])
    predicted_total = observed + 0.5 * (state_calibrated[1] + weather_raw[1])
    p10_total = min(observed + state_calibrated[0], predicted_total)
    p90_total = max(observed + state_calibrated[2], predicted_total)

    prior_params = fit_prior_update(train)
    hour = int(live["cutoff_hour"].iat[0])
    baseline_total = float(
        live["prior_total"].iat[0]
        + prior_params.loc[hour, "beta"] * live["pace_residual"].iat[0]
    )
    row = {
        "generated_at_utc": generated_at.tz_convert("UTC").isoformat(),
        "forecast_day": live_day.date().isoformat(),
        "cutoff_ds_local": latest_ds.isoformat(),
        "cutoff_hour": hour,
        "model_version": MODEL_VERSION,
        "source_hash": source_hash,
        "artifact_sha256": artifact_manifest["artifact_sha256"],
        "model_fingerprint": model_fingerprint,
        "model_fingerprint_version": MODEL_FINGERPRINT_VERSION,
        "training_start": pd.Timestamp(train["day"].min()).date().isoformat(),
        "training_end": pd.Timestamp(train["day"].max()).date().isoformat(),
        "training_days": int(train["day"].nunique()),
        "observed_arrivals": observed,
        "predicted_total": float(predicted_total),
        "p10_total": float(p10_total),
        "p90_total": float(p90_total),
        "prior_update_baseline": baseline_total,
        "status": "shadow_only",
    }
    _append_forecast(args.output_csv, row)
    metadata = {
        "model_version": MODEL_VERSION,
        "source_hash": source_hash,
        "candidate": "50% raw calendar/weather point + 50% calibrated ED-state point",
        "interval": "calibrated ED-state P80 bounds expanded around ensemble point",
        "training_days": row["training_days"],
        "training_start": row["training_start"],
        "training_end": row["training_end"],
        "calibration_days": calibration_days,
        "calibration_shrinkage_days": args.calibration_shrinkage_days,
        "state_features": sets["boosted_state"],
        "calendar_weather_features": calendar_weather_features,
        "corrections": corrections.reset_index().to_dict(orient="records"),
        "artifact_manifest": artifact_manifest,
    }
    args.metadata_json.write_text(json.dumps(metadata, indent=2) + "\n")
    return {"status": "forecast_written", **row}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-csv", default=FLOW_URL)
    parser.add_argument("--weather-csv", required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--status-json", type=Path, required=True)
    parser.add_argument("--status-history-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--artifact-joblib", type=Path, required=True)
    parser.add_argument("--artifact-manifest-json", type=Path, required=True)
    parser.add_argument("--now")
    parser.add_argument("--max-age-minutes", type=int, default=90)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--calibration-days", type=int, default=56)
    parser.add_argument("--calibration-shrinkage-days", type=float, default=28.0)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.status_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    try:
        status = run_shadow(args)
    except DataQualityError as exc:
        status = {
            "status": "suppressed_data_quality",
            "reason": str(exc),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_version": MODEL_VERSION,
        }
    except Exception as exc:
        status = run_prior_update_fallback(args, exc)
    args.status_json.write_text(json.dumps(status, indent=2) + "\n")
    _append_status(args.status_history_csv, status)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
