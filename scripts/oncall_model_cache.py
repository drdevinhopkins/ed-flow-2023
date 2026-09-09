"""Freshness and compatibility checks for persisted on-call models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

CACHE_SCHEMA_VERSION = 1
DAILY_RETRAIN_HOUR = 4
MONTREAL_TIMEZONE = "America/Montreal"
TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CacheDecision:
    use_cache: bool
    reason: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class CachePolicy:
    force_retrain: bool


def cache_policy_from_env(environ: Mapping[str, str]) -> CachePolicy:
    force_retrain = (
        environ.get("ED_FLOW_ONCALL_FORCE_RETRAIN", "").strip().lower()
        in TRUE_VALUES
    )
    return CachePolicy(force_retrain=force_retrain)


def latest_daily_retrain_boundary(
    now_utc: datetime,
    *,
    timezone_name: str = MONTREAL_TIMEZONE,
    retrain_hour: int = DAILY_RETRAIN_HOUR,
) -> datetime:
    """Return the latest local daily retraining boundary as UTC."""
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    if retrain_hour not in range(24):
        raise ValueError("retrain_hour must be between 0 and 23")

    timezone = ZoneInfo(timezone_name)
    local_now = now_utc.astimezone(timezone)
    boundary_date = local_now.date()
    if local_now.time().replace(tzinfo=None) < time(retrain_hour):
        boundary_date -= timedelta(days=1)
    local_boundary = datetime.combine(
        boundary_date,
        time(retrain_hour),
        tzinfo=timezone,
    )
    return local_boundary.astimezone(UTC)


def _reject(reason: str) -> CacheDecision:
    return CacheDecision(use_cache=False, reason=reason)


def _artifact_paths(model_dir: Path, horizons: Sequence[int]) -> list[Path]:
    return [model_dir / f"oncall_within_{horizon}h.cbm" for horizon in horizons]


def evaluate_model_cache(
    model_dir: Path,
    *,
    expected_features: Sequence[str],
    expected_categorical: Sequence[str],
    expected_horizons: Sequence[int],
    expected_label_sha256: str,
    expected_model_training_version: str,
    now_utc: datetime,
    force_retrain: bool = False,
) -> CacheDecision:
    """Return whether all cached model artifacts are safe to reuse."""
    if force_retrain:
        return _reject("forced retraining requested")

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.is_file():
        return _reject("metadata is missing")

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return _reject(f"metadata is unreadable: {exc}")
    if not isinstance(metadata, dict):
        return _reject("metadata is not an object")

    if metadata.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
        return _reject("cache schema changed")
    if metadata.get("features") != list(expected_features):
        return _reject("feature schema changed")
    if metadata.get("categorical_features") != list(expected_categorical):
        return _reject("categorical feature schema changed")
    if metadata.get("horizons_hours") != list(expected_horizons):
        return _reject("forecast horizons changed")
    if metadata.get("label_file_sha256") != expected_label_sha256:
        return _reject("on-call labels changed")
    if metadata.get("model_training_version") != expected_model_training_version:
        return _reject("model training version changed")

    thresholds = metadata.get("calibration_thresholds")
    if not isinstance(thresholds, dict):
        return _reject("calibration metadata is incomplete")
    for horizon in expected_horizons:
        values = thresholds.get(str(horizon))
        if not isinstance(values, dict):
            return _reject("calibration metadata is incomplete")
        x_values = values.get("x")
        y_values = values.get("y")
        if (
            not isinstance(x_values, list)
            or not isinstance(y_values, list)
            or not x_values
            or len(x_values) != len(y_values)
        ):
            return _reject("calibration metadata is incomplete")

    validation_metrics = metadata.get("validation_metrics")
    if not isinstance(validation_metrics, list):
        return _reject("validation metrics are incomplete")
    try:
        metric_horizons = [metric["horizon_hours"] for metric in validation_metrics]
    except (KeyError, TypeError):
        return _reject("validation metrics are incomplete")
    if metric_horizons != list(expected_horizons):
        return _reject("validation metrics are incomplete")

    trained_at_raw = metadata.get("trained_at_utc")
    try:
        trained_at = datetime.fromisoformat(str(trained_at_raw))
    except ValueError:
        return _reject("training timestamp is invalid")
    if trained_at.tzinfo is None:
        return _reject("training timestamp has no timezone")

    now = now_utc.astimezone(UTC)
    trained_at_utc = trained_at.astimezone(UTC)
    if trained_at_utc > now:
        return _reject("training timestamp is in the future")
    if trained_at_utc < latest_daily_retrain_boundary(now):
        return _reject("cache predates daily retraining boundary")

    missing = [
        path.name
        for path in _artifact_paths(model_dir, expected_horizons)
        if not path.is_file()
    ]
    if missing:
        return _reject(f"model artifacts are missing: {', '.join(missing)}")

    model_hashes = metadata.get("model_sha256")
    if not isinstance(model_hashes, dict):
        return _reject("model artifact checksums are incomplete")
    for horizon, path in zip(expected_horizons, _artifact_paths(model_dir, expected_horizons)):
        expected_hash = model_hashes.get(str(horizon))
        if not isinstance(expected_hash, str):
            return _reject("model artifact checksums are incomplete")
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return _reject("model artifact checksum changed")
        if digest != expected_hash:
            return _reject("model artifact checksum changed")

    return CacheDecision(
        use_cache=True,
        reason="fresh compatible cache",
        metadata=metadata,
    )
