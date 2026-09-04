#!/usr/bin/env python3

import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_daily_arrival_outlook import OUTLOOK_COLUMNS, build_daily_outlook


class DailyArrivalOutlookTests(unittest.TestCase):
    def _daily(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "ds": "2026-09-04",
                    "daily_visits_prediction": 295.0,
                    "0.1": 273.0,
                    "0.9": 317.0,
                    "data_cutoff": "2026-09-03",
                    "forecast_generated_at_utc": "2026-09-04T10:15:00+00:00",
                    "seasonal_weekday_baseline": 293.5,
                    "explainability_method": "chronos2_group_counterfactual_v1",
                    "top_driver_1": "calendar_closure",
                    "top_driver_1_effect": 4.7,
                    "top_driver_2": "wind",
                    "top_driver_2_effect": -0.2,
                    "top_driver_3": "temperature",
                    "top_driver_3_effect": 0.1,
                    "explanation_text": "Daily explanation for today.",
                },
                {
                    "ds": "2026-09-05",
                    "daily_visits_prediction": 245.0,
                    "0.1": 227.0,
                    "0.9": 262.0,
                    "data_cutoff": "2026-09-03",
                    "forecast_generated_at_utc": "2026-09-04T10:15:00+00:00",
                    "seasonal_weekday_baseline": 251.25,
                    "explainability_method": "chronos2_group_counterfactual_v1",
                    "top_driver_1": "calendar_closure",
                    "top_driver_1_effect": -2.5,
                    "top_driver_2": "wind",
                    "top_driver_2_effect": -0.8,
                    "top_driver_3": "temperature",
                    "top_driver_3_effect": -0.3,
                    "explanation_text": "Daily explanation for tomorrow.",
                },
                {
                    "ds": "2026-09-03",
                    "daily_visits_prediction": 280.0,
                    "0.1": 260.0,
                    "0.9": 300.0,
                    "data_cutoff": "2026-09-02",
                    "forecast_generated_at_utc": "2026-09-03T10:15:00+00:00",
                    "seasonal_weekday_baseline": 290.0,
                    "explainability_method": "chronos2_group_counterfactual_v1",
                    "top_driver_1": "wind",
                    "top_driver_1_effect": -1.0,
                    "top_driver_2": "temperature",
                    "top_driver_2_effect": 0.5,
                    "top_driver_3": "atmosphere",
                    "top_driver_3_effect": 0.1,
                    "explanation_text": "Stale prior-day row.",
                },
            ]
        )

    def _intraday(self, forecast_day: str = "2026-09-04") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "generated_at_utc": "2026-09-04T15:05:00+00:00",
                    "generated_at_local": "2026-09-04T11:05:00-04:00",
                    "forecast_day": forecast_day,
                    "cutoff_ds_local": "2026-09-04T11:00:00",
                    "observed_arrivals": 112.0,
                    "predicted_total": 287.0,
                    "p10_total": 268.0,
                    "p90_total": 306.0,
                    "expected_additional_arrivals": 175.0,
                    "model_version": "intraday-ensemble-v1-2026-08-28",
                    "method": "50pct_calendar_weather_plus_50pct_calibrated_ed_state",
                    "forecast_text": "112 arrivals through 11:00. Forecast: 287 total by midnight.",
                }
            ]
        )

    def test_daily_only_maps_to_power_bi_schema(self) -> None:
        outlook = build_daily_outlook(self._daily(), today="2026-09-04")
        self.assertEqual(list(outlook.columns), OUTLOOK_COLUMNS)
        self.assertEqual(outlook["target_date"].tolist(), ["2026-09-04", "2026-09-05"])
        self.assertEqual(outlook["forecast_stage"].tolist(), ["day_ahead", "daily"])
        self.assertEqual(outlook["horizon_day"].tolist(), [0, 1])
        self.assertEqual(outlook.loc[0, "predicted_arrivals"], 295.0)
        self.assertEqual(outlook.loc[0, "delta_vs_baseline"], 1.5)
        self.assertEqual(outlook.loc[0, "source_model"], "daily_chronos2")
        self.assertTrue(str(outlook.loc[0, "generated_at_local"]).endswith("-04:00"))

    def test_intraday_supersedes_today_only(self) -> None:
        outlook = build_daily_outlook(
            self._daily(), intraday=self._intraday(), today="2026-09-04"
        )
        today = outlook.loc[outlook["target_date"].eq("2026-09-04")].iloc[0]
        tomorrow = outlook.loc[outlook["target_date"].eq("2026-09-05")].iloc[0]

        self.assertEqual(today["forecast_stage"], "intraday")
        self.assertEqual(today["predicted_arrivals"], 287.0)
        self.assertEqual(today["lower_80"], 268.0)
        self.assertEqual(today["upper_80"], 306.0)
        self.assertEqual(today["observed_arrivals"], 112.0)
        self.assertEqual(today["expected_remaining"], 175.0)
        self.assertEqual(today["seasonal_weekday_baseline"], 293.5)
        self.assertEqual(today["delta_vs_baseline"], -6.5)
        self.assertEqual(today["source_model"], "intraday_day_completion")
        self.assertTrue(pd.isna(today["top_driver_1"]))
        self.assertTrue(pd.isna(today["explainability_method"]))
        self.assertIn("287 total by midnight", today["explanation_text"])

        self.assertEqual(tomorrow["forecast_stage"], "daily")
        self.assertEqual(tomorrow["predicted_arrivals"], 245.0)
        self.assertEqual(tomorrow["top_driver_1"], "calendar_closure")

    def test_stale_intraday_does_not_overwrite_today(self) -> None:
        outlook = build_daily_outlook(
            self._daily(),
            intraday=self._intraday(forecast_day="2026-09-03"),
            today="2026-09-04",
        )
        today = outlook.loc[outlook["target_date"].eq("2026-09-04")].iloc[0]
        self.assertEqual(today["forecast_stage"], "day_ahead")
        self.assertEqual(today["predicted_arrivals"], 295.0)

    def test_intraday_can_create_today_when_daily_file_starts_tomorrow(self) -> None:
        daily = self._daily().loc[self._daily()["ds"].eq("2026-09-05")].copy()
        outlook = build_daily_outlook(daily, intraday=self._intraday(), today="2026-09-04")
        self.assertEqual(outlook["target_date"].tolist(), ["2026-09-04", "2026-09-05"])
        today = outlook.iloc[0]
        self.assertEqual(today["forecast_stage"], "intraday")
        self.assertTrue(pd.isna(today["seasonal_weekday_baseline"]))

    def test_interval_invariant_is_enforced(self) -> None:
        daily = self._daily()
        daily.loc[daily["ds"].eq("2026-09-04"), "0.1"] = 310.0
        with self.assertRaisesRegex(ValueError, "interval invariant"):
            build_daily_outlook(daily, today="2026-09-04")


if __name__ == "__main__":
    unittest.main()
