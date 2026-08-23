import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from holiday_features import add_holiday_features, build_ramq_nominal_calendar


class HolidayFeatureTests(unittest.TestCase):
    def test_ramq_nominal_calendar_has_13_families(self):
        calendar = build_ramq_nominal_calendar([2026])
        dates_2026 = {day: name for day, name in calendar.items() if day.year == 2026}
        self.assertEqual(len(dates_2026), 13)
        self.assertIn(pd.Timestamp("2026-05-18").date(), dates_2026)
        self.assertIn(pd.Timestamp("2026-09-07").date(), dates_2026)
        self.assertIn(pd.Timestamp("2026-10-12").date(), dates_2026)
        self.assertIn(pd.Timestamp("2026-12-24").date(), dates_2026)

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
        featured = add_holiday_features(frame, feature_set="rich")
        self.assertEqual(int(featured.loc[0, "is_friday_before_monday_holiday"]), 1)
        self.assertEqual(int(featured.loc[1, "is_ramq_holiday"]), 1)
        self.assertEqual(int(featured.loc[2, "is_tuesday_after_monday_holiday"]), 1)

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
