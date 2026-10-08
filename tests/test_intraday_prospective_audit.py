"""Offline safeguards for original-issue scoring; no model or credentials."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "audit", Path(__file__).resolve().parents[1] / "scripts/evaluation/prospective/audit_intraday_forecasts.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def forecast(**changes):
    return dict(forecast_day="2026-09-05", cutoff_ds_local="2026-09-05T11:00:00",
                cutoff_hour=11, generated_at_utc="2026-09-05T15:20:00Z",
                observed_arrivals=10, predicted_total=48, p10_total=40, p90_total=60,
                expected_additional_arrivals=38, prior_update_baseline=50,
                model_version="frozen-v1", status="experimental_forecast",
                within_prospective_window=True) | changes


class AuditTests(unittest.TestCase):
    def artifact(self, root, number, row=None, status="forecast_written", bad_hash=False):
        row = row or forecast()
        payload = {"status": status, "forecast": row, "generated_at_utc": row["generated_at_utc"]}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("intraday-daily-inflow-status.json", json.dumps(payload))
            archive.writestr("intraday-daily-inflow-forecast.csv", pd.DataFrame([row]).to_csv(index=False))
        data = buffer.getvalue()
        (root / f"{number}.zip").write_bytes(data)
        return {"id": number, "workflow_run": {"id": number, "head_branch": "main", "head_sha": "abc"},
                "created_at": "2026-09-05T16:00:00Z", "expired": False,
                "digest": "sha256:" + ("bad" if bad_hash else hashlib.sha256(data).hexdigest())}

    def test_suppression_ignores_leftover_csv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = [self.artifact(root, 1), self.artifact(root, 2, status="suppressed_data_quality")]
            rows, inventory, excluded = audit.read_artifacts(artifacts, [{"id": 1}, {"id": 2}], root)
            self.assertEqual(rows.artifact_id.tolist(), [1])
            self.assertEqual(inventory.status.tolist(), ["forecast_written", "suppressed_data_quality"])
            self.assertEqual(excluded, [])

    def test_digest_mismatch_quarantined(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = [self.artifact(root, 1), self.artifact(root, 2, bad_hash=True)]
            rows, inventory, excluded = audit.read_artifacts(artifacts, [{"id": 1}, {"id": 2}], root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(inventory.iloc[1].status, "quarantined_artifact")
            self.assertIn("digest mismatch", excluded[0]["reason"])

    def test_interval_model_is_a_separate_target_collection(self):
        from arrival_day_policy import INTERVAL_TARGET_VERSION, INTERVAL_QUALITY_VERSION
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = forecast(target_definition_version=INTERVAL_TARGET_VERSION,
                           target_quality_version=INTERVAL_QUALITY_VERSION, model_version="interval-v2")
            artifacts = [self.artifact(root, 1), self.artifact(root, 2, row)]
            runs = [{"id": 1}, {"id": 2}]
            legacy, _, _ = audit.read_artifacts(artifacts, runs, root)
            new, _, _ = audit.read_artifacts(artifacts, runs, root, target_definition=INTERVAL_TARGET_VERSION)
            self.assertEqual(legacy.artifact_id.tolist(), [1])
            self.assertEqual(new.artifact_id.tolist(), [2])
            self.assertEqual(new.target_quality_version.tolist(), [INTERVAL_QUALITY_VERSION])

    def test_earliest_issue_wins_not_best_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            late = forecast(generated_at_utc="2026-09-05T15:30:00Z", predicted_total=49,
                            expected_additional_arrivals=39)
            artifacts = [self.artifact(root, 2, late), self.artifact(root, 1)]
            rows, _, excluded = audit.read_artifacts(artifacts, [{"id": 1}, {"id": 2}], root)
            self.assertEqual(rows.predicted_total.tolist(), [48])
            self.assertEqual(excluded, [{"artifact_id": 2, "reason": "later repeated issue"}])

    def test_forecast_invariants_and_prospective_flag(self):
        for changes in [{"within_prospective_window": "False"}, {"p10_total": 9},
                        {"expected_additional_arrivals": 0}, {"generated_at_utc": "2026-09-05T17:00:00Z"},
                        {"generated_at_utc": "2026-09-05T15:20:00"}, {"predicted_total": float("nan")},
                        {"cutoff_hour": 12}, {"model_version": ""}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                audit.validate_forecast(forecast(**changes), "2026-09-05T18:00:00Z")

    def test_artifact_must_precede_end_of_day(self):
        with self.assertRaises(ValueError):
            audit.validate_forecast(forecast(), "2026-09-06T04:00:00Z")

    def test_only_verified_matured_actuals_scored_without_mutation(self):
        hourly = pd.DataFrame({"ds": pd.date_range("2026-09-05", periods=72, freq="h"), "Inflow_Total": 2})
        hourly = hourly.loc[~hourly.ds.eq(pd.Timestamp("2026-09-06T03:00"))
                            & hourly.ds.le(pd.Timestamp("2026-09-07T14:00"))].copy()
        original = hourly.copy(deep=True)
        rows = []
        for index, day in enumerate(["2026-09-05", "2026-09-06", "2026-09-07"]):
            rows.append(audit.validate_forecast(forecast(forecast_day=day, cutoff_ds_local=day+"T11:00:00",
                        generated_at_utc=day+"T15:20:00Z"), day+"T16:00:00Z") | {"artifact_id": index})
        scores, _, exclusions = audit.score_forecasts(pd.DataFrame(rows), hourly, now="2026-09-07T18:00:00Z")
        self.assertEqual(scores.actual.tolist(), [48])
        self.assertEqual([r["reason"] for r in exclusions], ["unverified actual", "not yet matured"])
        pd.testing.assert_frame_equal(hourly, original)

    def test_missing_date_breaks_collection_streak(self):
        rows = pd.DataFrame([{"model_version": "v1", "forecast_day": pd.Timestamp(day), "cutoff_hour": h}
                             for day in ["2026-09-05", "2026-09-07"] for h in range(11, 19)])
        calendar = audit.coverage_calendar(rows, "2026-09-05", "2026-09-07")
        self.assertEqual(calendar.cutoffs_present.tolist(), [8, 0, 8])
        self.assertEqual(audit.longest_streak(calendar.complete), (1, 1))

    def test_boundary_comparison_is_diagnostic_not_target_relabeling(self):
        hourly = pd.DataFrame({"ds": pd.date_range("2026-09-05", periods=49, freq="h"), "Inflow_Total": 2})
        hourly.loc[0, "Inflow_Total"] = 5
        hourly["Inflow_Cum_Total"] = [48 if stamp.hour == 0 else stamp.hour * 2 for stamp in hourly.ds]
        _, quality = audit.build_daily_inflow_outputs(hourly, now="2026-09-07T18:00:00Z")
        boundary = audit.day_boundary_audit(hourly, quality, pd.Timestamp("2026-09-05"), pd.Timestamp("2026-09-05"))
        self.assertEqual(boundary.stored_calendar_sum.tolist(), [51])
        self.assertEqual(boundary.shifted_hour_sum.tolist(), [48])
        self.assertTrue(boundary.entire_counter_chain_matches_shifted_sum.all())


if __name__ == "__main__":
    unittest.main()
