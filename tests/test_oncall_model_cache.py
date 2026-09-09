import hashlib
import json
import sys
import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from oncall_model_cache import (
    CACHE_SCHEMA_VERSION,
    cache_policy_from_env,
    evaluate_model_cache,
    latest_daily_retrain_boundary,
)


class OncallModelCacheTests(unittest.TestCase):
    def create_cache(self, model_dir: Path, **metadata_overrides):
        metadata = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "trained_at_utc": "2026-09-08T12:00:00+00:00",
            "features": ["total_tbs", "oncall_physician_id"],
            "categorical_features": ["oncall_physician_id"],
            "horizons_hours": [4, 6, 8],
            "label_file_sha256": "abc123",
            "model_training_version": "spec-v1",
            "model_sha256": {
                str(horizon): hashlib.sha256(b"").hexdigest()
                for horizon in (4, 6, 8)
            },
            "calibration_thresholds": {
                str(horizon): {"x": [0.0, 1.0], "y": [0.0, 1.0]}
                for horizon in (4, 6, 8)
            },
            "validation_metrics": [
                {"horizon_hours": horizon} for horizon in (4, 6, 8)
            ],
        }
        metadata.update(metadata_overrides)
        (model_dir / "metadata.json").write_text(json.dumps(metadata))
        for horizon in (4, 6, 8):
            (model_dir / f"oncall_within_{horizon}h.cbm").touch()
        return metadata

    def evaluate(self, model_dir: Path, **overrides):
        kwargs = {
            "expected_features": ["total_tbs", "oncall_physician_id"],
            "expected_categorical": ["oncall_physician_id"],
            "expected_horizons": (4, 6, 8),
            "expected_label_sha256": "abc123",
            "expected_model_training_version": "spec-v1",
            "now_utc": datetime(2026, 9, 8, 20, 0, tzinfo=UTC),
        }
        kwargs.update(overrides)
        return evaluate_model_cache(model_dir, **kwargs)

    def test_non_utf8_metadata_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "metadata.json").write_bytes(b"\xff\xfe{}")

            decision = self.evaluate(model_dir)

            self.assertFalse(decision.use_cache)
            self.assertTrue(
                decision.reason.startswith("metadata is unreadable: ")
            )

    def test_non_object_metadata_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "metadata.json").write_text("[]")

            decision = self.evaluate(model_dir)

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "metadata is not an object")

    def test_fresh_compatible_cache_is_reused(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            metadata = self.create_cache(model_dir)

            decision = self.evaluate(model_dir)

            self.assertTrue(decision.use_cache)
            self.assertEqual(decision.reason, "fresh compatible cache")
            self.assertEqual(decision.metadata, metadata)

    def test_stale_cache_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(
                model_dir,
                trained_at_utc="2026-09-07T08:00:00+00:00",
            )

            decision = self.evaluate(
                model_dir,
                now_utc=datetime(2026, 9, 9, 13, 0, tzinfo=UTC),
            )

            self.assertFalse(decision.use_cache)
            self.assertEqual(
                decision.reason, "cache predates daily retraining boundary"
            )

    def test_cache_is_reused_before_montreal_four_am_boundary(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(
                model_dir,
                now_utc=datetime(2026, 9, 9, 7, 30, tzinfo=UTC),
            )

            self.assertTrue(decision.use_cache)

    def test_daily_boundary_is_dst_safe(self):
        spring = latest_daily_retrain_boundary(
            datetime(2026, 3, 8, 12, 0, tzinfo=UTC)
        )
        fall = latest_daily_retrain_boundary(
            datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
        )

        self.assertEqual(spring, datetime(2026, 3, 8, 8, 0, tzinfo=UTC))
        self.assertEqual(fall, datetime(2026, 11, 1, 9, 0, tzinfo=UTC))

    def test_changed_labels_require_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(model_dir, expected_label_sha256="changed")

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "on-call labels changed")

    def test_changed_categorical_schema_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(
                model_dir, expected_categorical=["other_id"]
            )

            self.assertFalse(decision.use_cache)
            self.assertEqual(
                decision.reason, "categorical feature schema changed"
            )

    def test_changed_horizons_require_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(model_dir, expected_horizons=(4, 6))

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "forecast horizons changed")

    def test_exact_daily_boundary_is_reusable(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(
                model_dir,
                trained_at_utc="2026-09-08T08:00:00+00:00",
            )

            decision = self.evaluate(
                model_dir,
                now_utc=datetime(2026, 9, 8, 8, 0, tzinfo=UTC),
            )

            self.assertTrue(decision.use_cache)

    def test_future_training_timestamp_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(
                model_dir,
                trained_at_utc="2026-09-09T20:00:00+00:00",
            )

            decision = self.evaluate(
                model_dir,
                now_utc=datetime(2026, 9, 9, 12, 0, tzinfo=UTC),
            )

            self.assertFalse(decision.use_cache)
            self.assertEqual(
                decision.reason, "training timestamp is in the future"
            )

    def test_changed_feature_schema_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(model_dir, expected_features=["total_tbs"])

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "feature schema changed")

    def test_changed_training_version_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(
                model_dir, expected_model_training_version="spec-v2"
            )

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "model training version changed")

    def test_modified_model_artifact_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)
            (model_dir / "oncall_within_6h.cbm").write_bytes(b"changed")

            decision = self.evaluate(model_dir)

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "model artifact checksum changed")

    def test_validation_metric_horizons_must_match(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(
                model_dir,
                validation_metrics=[{"horizon_hours": 4}] * 3,
            )

            decision = self.evaluate(model_dir)

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "validation metrics are incomplete")

    def test_missing_calibration_thresholds_require_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir, calibration_thresholds={})

            decision = self.evaluate(model_dir)

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "calibration metadata is incomplete")

    def test_missing_artifact_requires_retraining(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)
            (model_dir / "oncall_within_6h.cbm").unlink()

            decision = self.evaluate(model_dir)

            self.assertFalse(decision.use_cache)
            self.assertIn("oncall_within_6h.cbm", decision.reason)

    def test_force_retrain_bypasses_fresh_cache(self):
        with TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            self.create_cache(model_dir)

            decision = self.evaluate(model_dir, force_retrain=True)

            self.assertFalse(decision.use_cache)
            self.assertEqual(decision.reason, "forced retraining requested")

    def test_cache_policy_defaults_to_scheduled_retraining(self):
        policy = cache_policy_from_env({})

        self.assertFalse(policy.force_retrain)

    def test_cache_policy_accepts_force_retrain_override(self):
        policy = cache_policy_from_env(
            {"ED_FLOW_ONCALL_FORCE_RETRAIN": "true"}
        )

        self.assertTrue(policy.force_retrain)


if __name__ == "__main__":
    unittest.main()
