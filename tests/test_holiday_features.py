import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from holiday_features import (
    add_holiday_features,
    build_ramq_calendar,
    build_ramq_nominal_calendar,
    load_jgh_ramq_calendar,
)


class HolidayFeatureTests(unittest.TestCase):
    def test_ramq_nominal_calendar_has_13_families(self):
        calendar = build_ramq_nominal_calendar([2026])
        dates_2026 = {day: name for day, name in calendar.items() if day.year == 2026}
        self.assertEqual(len(dates_2026), 13)
        self.assertIn(pd.Timestamp("2026-05-18").date(), dates_2026)
        self.assertIn(pd.Timestamp("2026-09-07").date(), dates_2026)
        self.assertIn(pd.Timestamp("2026-10-12").date(), dates_2026)
        self.assertIn(pd.Timestamp("2026-12-24").date(), dates_2026)

    def test_jgh_calendar_has_13_dates_per_reference_year(self):
        calendar, coverage_start, coverage_end = load_jgh_ramq_calendar()
        self.assertEqual(len(calendar), 78)
        self.assertEqual(coverage_start, pd.Timestamp("2021-07-01").date())
        self.assertEqual(coverage_end, pd.Timestamp("2027-06-24").date())

    def test_jgh_calendar_replaces_nominal_dates_inside_coverage(self):
        exact = build_ramq_calendar([2025, 2026], ramq_calendar="jgh")
        nominal = build_ramq_calendar([2025, 2026], ramq_calendar="nominal")

        rosh_2025 = pd.Timestamp("2025-09-23").date()
        jan2_2026 = pd.Timestamp("2026-01-02").date()
        passover_2026 = pd.Timestamp("2026-04-02").date()
        easter_monday_2026 = pd.Timestamp("2026-04-06").date()

        self.assertIn(rosh_2025, exact)
        self.assertNotIn(rosh_2025, nominal)
        self.assertNotIn(jan2_2026, exact)
        self.assertIn(jan2_2026, nominal)
        self.assertIn(passover_2026, exact)
        self.assertNotIn(passover_2026, nominal)
        self.assertNotIn(easter_monday_2026, exact)
        self.assertIn(easter_monday_2026, nominal)

    def test_feature_builder_uses_jgh_ramq_calendar_by_default(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    ["2025-09-23", "2026-01-02", "2026-04-02", "2026-04-06"]
                )
            }
        )
        exact = add_holiday_features(frame, feature_set="closures")
        nominal = add_holiday_features(
            frame, feature_set="closures", ramq_calendar="nominal"
        )
        self.assertEqual(exact["is_ramq_holiday"].tolist(), [1, 0, 1, 0])
        self.assertEqual(nominal["is_ramq_holiday"].tolist(), [0, 1, 0, 1])

    def test_long_weekend_shoulders(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    [
                        "2026-05-15 12:00",
                        "2026-05-18 12:00",
                        "2026-05-19 12:00",
                    ]
                )
            }
        )
        featured = add_holiday_features(frame, feature_set="closures")
        self.assertEqual(int(featured.loc[0, "is_friday_before_monday_holiday"]), 1)
        self.assertEqual(int(featured.loc[1, "is_ramq_holiday"]), 1)
        self.assertEqual(int(featured.loc[2, "is_tuesday_after_monday_holiday"]), 1)
        self.assertEqual(int(featured.loc[0, "closed_days_immediately_ahead"]), 3)
        self.assertEqual(int(featured.loc[0, "is_pre_long_closure"]), 1)
        self.assertEqual(int(featured.loc[2, "closed_days_immediately_before"]), 3)
        self.assertEqual(int(featured.loc[2, "is_rebound_after_long_closure"]), 1)

    def test_regular_monday_is_not_long_closure_rebound(self):
        frame = pd.DataFrame({"ds": pd.to_datetime(["2026-05-11 12:00"])})
        featured = add_holiday_features(frame, feature_set="closures")
        self.assertEqual(int(featured.loc[0, "closed_days_immediately_before"]), 2)
        self.assertEqual(int(featured.loc[0, "is_first_business_day_after_closure"]), 1)
        self.assertEqual(int(featured.loc[0, "is_rebound_after_long_closure"]), 0)

    def test_christmas_new_year_cluster(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    ["2026-12-23", "2026-12-28", "2027-01-03", "2027-01-04"]
                )
            }
        )
        featured = add_holiday_features(frame, feature_set="rich")
        self.assertEqual(featured["is_christmas_newyear_period"].tolist(), [1, 1, 1, 0])

    def test_major_jewish_features_exist_in_year(self):
        frame = pd.DataFrame({"ds": pd.date_range("2026-01-01", "2026-12-31", freq="D")})
        featured = add_holiday_features(frame, feature_set="rich")
        self.assertGreater(int(featured["is_major_jewish_holiday"].sum()), 0)
        self.assertGreater(int(featured["is_major_jewish_holiday_eve"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
