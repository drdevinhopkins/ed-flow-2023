import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "backtests"))

from backtest_intraday_day_completion import (
    add_curve_features,
    apply_quantile_corrections,
    attach_weather,
    build_expanding_folds,
    build_snapshots,
    build_weather_features,
    evaluate_readiness,
    expected_local_hours,
    fit_completion_curve,
    fit_quantile_corrections,
    load_hourly_flow,
    predict_completion_curve,
    run_backtest,
)


class IntradayDayCompletionTests(unittest.TestCase):
    def _flow_rows(self, start: str, days: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for day_number, day in enumerate(pd.date_range(start, periods=days, freq="D")):
            for sequence, hour in enumerate(expected_local_hours(day)):
                inflow = float(2 + (hour >= 8) + (day_number % 5 == 0))
                rows.append(
                    {
                        "ds": day + pd.Timedelta(hours=hour),
                        "Inflow_Total": inflow,
                        "INFLOW_STRETCHER": float(hour % 2 == 0),
                        "INFLOW_AMBULATORY": inflow - float(hour % 2 == 0),
                        "INFLOW_AMBULANCES": float(hour % 6 == 0),
                        "TRG_HALLWAY_TBS": 1 + hour % 3,
                        "POD_GREEN_TBS": 2 + hour % 4,
                        "POD_YELLOW_TBS": 3,
                        "POD_ORANGE_TBS": 4,
                        "RAZ_TBS": 2,
                        "AMBVERTTBS": 1,
                        "QTrack_TBS": 1,
                        "Garage_TBS": 0,
                        "POST_POD1": 2,
                        "TRG_HALLWAY1": 3,
                        "WAITINGADM": 5 + day_number % 3,
                        "TTStr": 40 + hour % 5,
                        "RESUS": 2,
                        "_sequence": sequence,
                    }
                )
        return rows

    def _write_csv(self, rows: list[dict[str, object]]) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        pd.DataFrame(rows).drop(columns="_sequence", errors="ignore").to_csv(handle.name, index=False)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_snapshots_predict_remaining_and_use_canonical_total_tbs(self):
        flow = load_hourly_flow(self._write_csv(self._flow_rows("2026-01-01", 20)))
        snapshots = build_snapshots(flow, calendar_mode="basic")
        row = snapshots.loc[
            snapshots["day"].eq(pd.Timestamp("2026-01-10"))
            & snapshots["cutoff_hour"].eq(11)
        ].iloc[0]

        expected_so_far = sum(2 + (hour >= 8) for hour in range(12))
        expected_total = sum(2 + (hour >= 8) for hour in range(24))
        self.assertEqual(row["cumulative_arrivals"], expected_so_far)
        self.assertEqual(row["final_total"], expected_total)
        self.assertEqual(row["remaining_arrivals"], expected_total - expected_so_far)
        self.assertEqual(row["state_Total_TBS"], 1 + 11 % 3 + 2 + 11 % 4 + 3 + 4 + 2 + 1 + 1)

    def test_missing_ordinary_hour_excludes_entire_day(self):
        rows = self._flow_rows("2026-01-01", 5)
        rows = [
            row
            for row in rows
            if not (pd.Timestamp(row["ds"]) == pd.Timestamp("2026-01-03 12:00:00"))
        ]
        flow = load_hourly_flow(self._write_csv(rows))
        snapshots = build_snapshots(flow, calendar_mode="basic")

        self.assertFalse(bool(flow.loc[flow["day"].eq(pd.Timestamp("2026-01-03")), "is_complete_day"].any()))
        self.assertNotIn(pd.Timestamp("2026-01-03"), set(snapshots["day"]))

    def test_prior_uses_only_earlier_daily_totals(self):
        original = self._flow_rows("2026-01-01", 40)
        changed = [dict(row) for row in original]
        for row in changed:
            if pd.Timestamp(row["ds"]).normalize() >= pd.Timestamp("2026-02-05"):
                row["Inflow_Total"] = 1000.0

        first = build_snapshots(load_hourly_flow(self._write_csv(original)), calendar_mode="basic")
        second = build_snapshots(load_hourly_flow(self._write_csv(changed)), calendar_mode="basic")
        columns = ["day", "cutoff_hour", "prior_total"]
        before = pd.Timestamp("2026-02-05")
        pd.testing.assert_frame_equal(
            first.loc[first["day"].le(before), columns].reset_index(drop=True),
            second.loc[second["day"].le(before), columns].reset_index(drop=True),
        )

    def test_weather_merge_never_uses_a_future_observation(self):
        weather_rows = [
            {"ds": "2026-01-01 10:00:00", "temperature_2m": -5.0, "snowfall": 1.0},
            {"ds": "2026-01-01 12:00:00", "temperature_2m": 20.0, "snowfall": 100.0},
        ]
        weather = build_weather_features(self._write_csv(weather_rows))
        snapshots = pd.DataFrame({"ds": pd.to_datetime(["2026-01-01 11:00:00"])})
        merged = attach_weather(snapshots, weather)

        self.assertEqual(merged.loc[0, "weather_current_temperature_2m"], -5.0)
        self.assertEqual(merged.loc[0, "weather_current_snowfall"], 1.0)

    def test_expanding_folds_and_backtest_are_time_ordered_and_bounded(self):
        flow = load_hourly_flow(self._write_csv(self._flow_rows("2025-01-01", 90)))
        snapshots = build_snapshots(flow, calendar_mode="basic")
        eligible = snapshots.loc[snapshots["cutoff_hour"].eq(11)].copy()
        folds = build_expanding_folds(eligible, n_folds=1, test_days=10, min_train_days=40)
        _, train_mask, test_mask = folds[0]
        self.assertLess(eligible.loc[train_mask, "day"].max(), eligible.loc[test_mask, "day"].min())

        predictions, summary, features = run_backtest(
            snapshots,
            cutoff_hours=[11],
            n_folds=1,
            test_days=10,
            min_train_days=40,
            max_iter=10,
            random_state=7,
        )
        self.assertTrue((predictions["predicted_total"] >= predictions["observed_so_far"]).all())
        self.assertTrue({"completion_curve", "prior_update", "boosted_full"}.issubset(predictions["model"]))
        self.assertIn("boosted_full_calibrated", set(predictions["model"]))
        self.assertFalse(summary.empty)
        self.assertIn("state_Total_TBS", set(features.loc[features["model"].eq("boosted_full"), "feature"]))
        readiness = evaluate_readiness(predictions)
        self.assertTrue(readiness["retrospective_gates"]["forecast_and_interval_invariants"])
        self.assertFalse(readiness["production_ready"])

    def test_completion_curve_is_fit_from_training_rows(self):
        flow = load_hourly_flow(self._write_csv(self._flow_rows("2026-01-01", 30)))
        snapshots = build_snapshots(flow, calendar_mode="basic")
        train = snapshots.loc[snapshots["day"].lt(pd.Timestamp("2026-01-25"))]
        test = snapshots.loc[
            snapshots["day"].eq(pd.Timestamp("2026-01-25"))
            & snapshots["cutoff_hour"].eq(11)
        ]
        curve = fit_completion_curve(train)
        featured = add_curve_features(test, curve)
        prediction = predict_completion_curve(featured, curve, fold_id=0)

        self.assertGreater(featured.iloc[0]["expected_fraction"], 0)
        self.assertGreaterEqual(prediction.iloc[0]["predicted_total"], prediction.iloc[0]["observed_so_far"])

    def test_quantile_calibration_learns_only_residual_adjustments(self):
        actual = np.array([30.0, 32.0, 34.0, 36.0])
        predicted = np.column_stack([actual - 7.0, actual - 5.0, actual - 3.0])
        hours = np.array([11, 11, 12, 12])

        corrections = fit_quantile_corrections(
            actual,
            predicted,
            hours,
            shrinkage_days=0.0,
        )
        calibrated = apply_quantile_corrections(predicted, hours, corrections)

        np.testing.assert_allclose(calibrated, np.column_stack([actual, actual, actual]))
        self.assertTrue((np.diff(calibrated, axis=1) >= 0).all())

        skewed_actual = np.array([0.0, 0.0, 0.0, 8.0])
        skewed_prediction = np.zeros((4, 3))
        mean_bias_correction = fit_quantile_corrections(
            skewed_actual,
            skewed_prediction,
            np.repeat(11, 4),
            shrinkage_days=0.0,
        )
        self.assertEqual(mean_bias_correction.loc[11, "q50_correction"], 2.0)


if __name__ == "__main__":
    unittest.main()
