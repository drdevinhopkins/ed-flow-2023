import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation" / "prospective"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation" / "backtests"))

from backtest_intraday_day_completion import load_hourly_flow
from run_shadow_intraday_day_completion import (
    DataQualityError,
    _append_forecast,
    run_prior_update_fallback,
    training_input_fingerprint,
    validate_live_flow,
    validate_fingerprint_against_ledger,
    write_model_artifact,
)
from score_shadow_intraday_day_completion import (
    evaluate_prospective_readiness,
    score_forecasts,
    summarize_scores,
)
from check_shadow_intraday_day_completion import evaluate_shadow_health


class ShadowIntradayTests(unittest.TestCase):
    def _flow(self, *, missing_hour=None):
        rows = []
        for hour in range(12):
            if hour == missing_hour:
                continue
            rows.append(
                {
                    "ds": f"2026-08-28 {hour:02d}:00:00",
                    "Inflow_Total": 2,
                    "TRG_HALLWAY_TBS": 1,
                    "POD_GREEN_TBS": 2,
                    "POD_YELLOW_TBS": 3,
                    "POD_ORANGE_TBS": 4,
                    "RAZ_TBS": 0,
                    "AMBVERTTBS": 1,
                    "QTrack_TBS": 0,
                    "Garage_TBS": 0,
                }
            )
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame(rows).to_csv(handle.name, index=False)
        return load_hourly_flow(handle.name)

    def test_fresh_contiguous_flow_passes(self):
        latest, current = validate_live_flow(
            self._flow(), now=pd.Timestamp("2026-08-28 11:30", tz="America/Montreal")
        )
        self.assertEqual(latest, pd.Timestamp("2026-08-28 11:00"))
        self.assertEqual(len(current), 12)

    def test_stale_or_missing_hour_suppresses(self):
        with self.assertRaises(DataQualityError):
            validate_live_flow(
                self._flow(), now=pd.Timestamp("2026-08-28 14:00", tz="America/Montreal")
            )
        with self.assertRaises(DataQualityError):
            validate_live_flow(
                self._flow(missing_hour=7),
                now=pd.Timestamp("2026-08-28 11:30", tz="America/Montreal"),
            )

    def test_append_is_idempotent_for_model_day_and_hour(self):
        path = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        path.unlink()
        row = {
            "model_version": "v1",
            "forecast_day": "2026-08-28",
            "cutoff_hour": 11,
            "predicted_total": 200,
        }
        _append_forecast(path, row)
        _append_forecast(path, {**row, "predicted_total": 999})
        saved = pd.read_csv(path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.loc[0, "predicted_total"], 200)

    def test_serialized_artifact_is_reloaded_and_hashed(self):
        directory = Path(tempfile.mkdtemp())
        artifact = directory / "model.joblib"
        manifest_path = directory / "manifest.json"
        bundle = {
            "model_version": "v1",
            "source_hash": "abc",
            "training_start": "2025-01-01",
            "training_end": "2025-12-31",
            "training_days": 365,
            "payload": [1, 2, 3],
        }
        loaded, manifest = write_model_artifact(bundle, artifact, manifest_path)
        self.assertEqual(loaded["payload"], [1, 2, 3])
        self.assertEqual(len(manifest["artifact_sha256"]), 64)
        self.assertEqual(json.loads(manifest_path.read_text()), manifest)

    def test_functional_drift_is_suppressed_before_append(self):
        path = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        pd.DataFrame(
            {
                "model_version": ["v1"],
                "model_fingerprint_version": ["functional-probe-v2"],
                "training_end": ["2026-08-28"],
                "model_fingerprint": ["stable"],
            }
        ).to_csv(path, index=False)
        validate_fingerprint_against_ledger(
            path,
            model_version="v1",
            fingerprint_version="functional-probe-v2",
            training_end="2026-08-28",
            fingerprint="stable",
        )
        with self.assertRaises(DataQualityError):
            validate_fingerprint_against_ledger(
                path,
                model_version="v1",
                fingerprint_version="functional-probe-v2",
                training_end="2026-08-28",
                fingerprint="drifted",
            )

    def test_training_input_fingerprint_is_order_stable_and_route_specific(self):
        frame = pd.DataFrame(
            {
                "day": ["2026-08-27", "2026-08-26"],
                "cutoff_hour": [12, 11],
                "remaining_arrivals": [10.0, 20.0],
                "state_feature": [1.0, 2.0],
                "weather_feature": [3.0, 4.0],
            }
        )
        state = training_input_fingerprint(frame, ["state_feature"])
        self.assertEqual(state, training_input_fingerprint(frame.iloc[::-1], ["state_feature"]))

        revised_weather = frame.copy()
        revised_weather.loc[0, "weather_feature"] = 99.0
        self.assertEqual(state, training_input_fingerprint(revised_weather, ["state_feature"]))
        self.assertNotEqual(
            training_input_fingerprint(frame, ["weather_feature"]),
            training_input_fingerprint(revised_weather, ["weather_feature"]),
        )

    def test_functional_drift_attributes_changed_training_route(self):
        path = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        pd.DataFrame(
            {
                "model_version": ["v1"],
                "model_fingerprint_version": ["functional-probe-v2"],
                "training_end": ["2026-08-28"],
                "model_fingerprint": ["stable"],
                "training_input_fingerprint_version": ["training-matrix-v1"],
                "state_training_fingerprint": ["state-stable"],
                "weather_training_fingerprint": ["weather-stable"],
            }
        ).to_csv(path, index=False)
        with self.assertRaisesRegex(DataQualityError, "changed calendar/weather training matrix"):
            validate_fingerprint_against_ledger(
                path,
                model_version="v1",
                fingerprint_version="functional-probe-v2",
                training_end="2026-08-28",
                fingerprint="drifted",
                training_input_fingerprint_version="training-matrix-v1",
                state_training_fingerprint="state-stable",
                weather_training_fingerprint="weather-revised",
            )

    def test_functional_drift_attributes_fit_nondeterminism(self):
        path = Path(tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name)
        pd.DataFrame(
            {
                "model_version": ["v1"],
                "model_fingerprint_version": ["functional-probe-v2"],
                "training_end": ["2026-08-28"],
                "model_fingerprint": ["stable"],
                "training_input_fingerprint_version": ["training-matrix-v1"],
                "state_training_fingerprint": ["state-stable"],
                "weather_training_fingerprint": ["weather-stable"],
            }
        ).to_csv(path, index=False)
        with self.assertRaisesRegex(DataQualityError, "possible fit or dependency nondeterminism"):
            validate_fingerprint_against_ledger(
                path,
                model_version="v1",
                fingerprint_version="functional-probe-v2",
                training_end="2026-08-28",
                fingerprint="drifted",
                training_input_fingerprint_version="training-matrix-v1",
                state_training_fingerprint="state-stable",
                weather_training_fingerprint="weather-stable",
            )

    def test_scoring_excludes_functionally_drifted_forecast(self):
        forecasts = pd.DataFrame(
            {
                "forecast_day": ["2026-08-27", "2026-08-27"],
                "cutoff_hour": [17, 18],
                "model_version": ["v1", "v1"],
                "source_hash": ["source", "source"],
                "model_fingerprint_version": ["functional-probe-v2", "functional-probe-v2"],
                "training_end": ["2026-08-26", "2026-08-26"],
                "generated_at_utc": ["2026-08-27T21:00:00Z", "2026-08-27T22:00:00Z"],
                "model_fingerprint": ["stable", "drifted"],
                "predicted_total": [49.0, 999.0],
                "p10_total": [40.0, 900.0],
                "p90_total": [60.0, 1000.0],
                "prior_update_baseline": [45.0, 45.0],
                "status": ["shadow_only", "shadow_only"],
            }
        )
        rows = [
            {"ds": f"2026-08-27 {hour:02d}:00:00", "Inflow_Total": 2.0}
            for hour in range(24)
        ]
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame(rows).to_csv(handle.name, index=False)
        scored = score_forecasts(forecasts, load_hourly_flow(handle.name))
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0]["model_fingerprint"], "stable")

    def test_scoring_waits_for_complete_day_and_calculates_metrics(self):
        forecasts = pd.DataFrame(
            {
                "forecast_day": ["2026-08-27", "2026-08-28"],
                "cutoff_hour": [15, 15],
                "model_version": ["v1", "v1"],
                "predicted_total": [49.0, 100.0],
                "p10_total": [40.0, 90.0],
                "p90_total": [60.0, 110.0],
                "prior_update_baseline": [45.0, 95.0],
            }
        )
        rows = []
        for hour in range(24):
            rows.append({"ds": f"2026-08-27 {hour:02d}:00:00", "Inflow_Total": 2.0})
        rows.append({"ds": "2026-08-28 00:00:00", "Inflow_Total": 2.0})
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame(rows).to_csv(handle.name, index=False)

        scored = score_forecasts(forecasts, load_hourly_flow(handle.name))
        summary = summarize_scores(scored)
        readiness = evaluate_prospective_readiness(summary)

        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0]["actual_total"], 48.0)
        self.assertEqual(scored.iloc[0]["error"], 1.0)
        self.assertEqual(summary["prospective_days"], 1)
        self.assertFalse(readiness["prospective_ready"])

    def test_fallback_rows_are_excluded_from_candidate_evidence(self):
        forecasts = pd.DataFrame(
            {
                "forecast_day": ["2026-08-27", "2026-08-27"],
                "cutoff_hour": [15, 15],
                "model_version": ["candidate", "fallback"],
                "predicted_total": [49.0, 48.0],
                "p10_total": [40.0, None],
                "p90_total": [60.0, None],
                "prior_update_baseline": [45.0, 48.0],
                "status": ["shadow_only", "shadow_fallback"],
            }
        )
        rows = [
            {"ds": f"2026-08-27 {hour:02d}:00:00", "Inflow_Total": 2.0}
            for hour in range(24)
        ]
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        pd.DataFrame(rows).to_csv(handle.name, index=False)
        scored = score_forecasts(forecasts, load_hourly_flow(handle.name))
        self.assertEqual(len(scored), 1)
        self.assertEqual(scored.iloc[0]["model_version"], "candidate")

    def test_model_error_records_labeled_prior_update_fallback(self):
        directory = Path(tempfile.mkdtemp())
        args = argparse.Namespace(
            now="2026-08-28T18:30:00-04:00",
            flow_csv="unused.csv",
            max_age_minutes=90,
            min_train_days=1,
            output_csv=directory / "forecasts.csv",
        )
        flow = pd.DataFrame(
            {"day": [pd.Timestamp("2026-08-27")], "is_complete_day": [True]}
        )
        snapshots = pd.DataFrame(
            {
                "day": [pd.Timestamp("2026-08-27"), pd.Timestamp("2026-08-28")],
                "ds": [pd.Timestamp("2026-08-27 18:00"), pd.Timestamp("2026-08-28 18:00")],
                "cutoff_hour": [18, 18],
                "prior_total": [250.0, 260.0],
                "pace_residual": [0.0, 5.0],
                "cumulative_arrivals": [200.0, 210.0],
            }
        )
        prior_params = pd.DataFrame({"beta": [0.5]}, index=[18])
        with (
            patch("run_shadow_intraday_day_completion.load_hourly_flow", return_value=flow),
            patch(
                "run_shadow_intraday_day_completion.validate_live_flow",
                return_value=(pd.Timestamp("2026-08-28 18:00"), pd.DataFrame()),
            ),
            patch(
                "run_shadow_intraday_day_completion.build_snapshots",
                return_value=snapshots,
            ),
            patch(
                "run_shadow_intraday_day_completion.fit_prior_update",
                return_value=prior_params,
            ),
        ):
            row = run_prior_update_fallback(args, RuntimeError("candidate failed"))

        self.assertEqual(row["status"], "shadow_fallback")
        self.assertEqual(row["predicted_total"], 262.5)
        self.assertIsNone(row["p10_total"])
        saved = pd.read_csv(args.output_csv)
        self.assertEqual(saved.iloc[0]["model_version"], "intraday-prior-update-fallback-v1")

    def test_empty_score_ledger_reports_zero_progress(self):
        summary = summarize_scores(pd.DataFrame())
        readiness = evaluate_prospective_readiness(summary)

        self.assertEqual(readiness["prospective_days"], 0)
        self.assertTrue(all(count == 0 for count in readiness["operational_hour_counts"].values()))
        self.assertFalse(readiness["prospective_ready"])

    def test_monitor_accepts_expected_idle_and_detects_bad_interval(self):
        now = pd.Timestamp("2026-08-29T00:05:00Z")
        idle = evaluate_shadow_health(
            {
                "status": "suppressed_data_quality",
                "reason": "cutoff hour 19:00 is outside the shadow window",
                "generated_at_utc": "2026-08-29T00:04:00Z",
            },
            pd.DataFrame({"status": ["shadow_only", "suppressed_data_quality"]}),
            pd.DataFrame(),
            {},
            {"prospective_days": 0, "production_ready": False},
            now=now,
        )
        self.assertEqual(idle["health"], "healthy_idle")

        lagged_idle = evaluate_shadow_health(
            {
                "status": "suppressed_data_quality",
                "reason": "cutoff hour 10:00 is outside the shadow window",
                "generated_at_utc": "2026-08-29T15:03:00Z",
            },
            pd.DataFrame(
                {
                    "status": ["suppressed_data_quality", "suppressed_data_quality"],
                    "reason": [
                        "cutoff hour 19:00 is outside the shadow window",
                        "cutoff hour 10:00 is outside the shadow window",
                    ],
                }
            ),
            pd.DataFrame(),
            {},
            {"prospective_days": 1, "production_ready": False},
            now=pd.Timestamp("2026-08-29T15:04:00Z"),
        )
        self.assertEqual(lagged_idle["health"], "healthy_idle")
        self.assertNotIn("consecutive_abnormal_runs", {item["code"] for item in lagged_idle["alerts"]})

        bad = pd.DataFrame(
            {
                "model_version": ["candidate"],
                "forecast_day": ["2026-08-28"],
                "cutoff_hour": [18],
                "generated_at_utc": ["2026-08-28T23:00:00Z"],
                "status": ["shadow_only"],
                "observed_arrivals": [222],
                "predicted_total": [280],
                "p10_total": [290],
                "p90_total": [300],
                "artifact_sha256": ["abc"],
            }
        )
        critical = evaluate_shadow_health(
            {"status": "shadow_only", "generated_at_utc": "2026-08-29T00:04:00Z"},
            pd.DataFrame({"status": ["shadow_only"]}),
            bad,
            {"artifact_sha256": "abc"},
            {"prospective_days": 0},
            now=now,
        )
        self.assertEqual(critical["health"], "critical")
        self.assertEqual(critical["invalid_candidate_forecasts"], 1)

    def test_monitor_detects_functional_model_drift_with_same_training_window(self):
        forecasts = pd.DataFrame(
            {
                "model_version": ["v1", "v1"],
                "forecast_day": ["2026-08-29", "2026-08-29"],
                "cutoff_hour": [13, 14],
                "generated_at_utc": ["2026-08-29T17:00:00Z", "2026-08-29T18:00:00Z"],
                "status": ["shadow_only", "shadow_only"],
                "training_end": ["2026-08-28", "2026-08-28"],
                "source_hash": ["source", "source"],
                "model_fingerprint_version": ["functional-probe-v2", "functional-probe-v2"],
                "model_fingerprint": ["functional-a", "functional-b"],
                "artifact_sha256": ["bytes-a", "bytes-b"],
                "observed_arrivals": [120, 140],
                "predicted_total": [260, 259],
                "p10_total": [245, 246],
                "p90_total": [275, 274],
            }
        )
        result = evaluate_shadow_health(
            {
                "status": "shadow_only",
                "generated_at_utc": "2026-08-29T18:00:00Z",
                "artifact_sha256": "bytes-b",
                "model_fingerprint": "functional-b",
            },
            pd.DataFrame({"status": ["shadow_only"]}),
            forecasts,
            {"artifact_sha256": "bytes-b", "model_fingerprint": "functional-b"},
            {"prospective_days": 1},
            now=pd.Timestamp("2026-08-29T18:01:00Z"),
        )
        self.assertEqual(result["health"], "warning")
        self.assertEqual(result["quarantined_model_drift_forecasts"], 1)
        self.assertIn(
            "quarantined_model_fingerprint_drift",
            {item["code"] for item in result["alerts"]},
        )


if __name__ == "__main__":
    unittest.main()
