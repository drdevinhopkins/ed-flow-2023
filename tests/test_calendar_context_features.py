import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from calendar_context_features import add_calendar_context_features


class CalendarContextFeatureTests(unittest.TestCase):
    def test_published_construction_vacation_boundaries(self):
        frame = pd.DataFrame(
            {
                "ds": pd.to_datetime(
                    [
                        "2021-07-17",
                        "2021-07-18",
                        "2021-07-31",
                        "2021-08-01",
                        "2024-07-21",
                        "2024-08-03",
                        "2026-07-19",
                        "2026-08-01",
                    ]
                )
            }
        )
        featured = add_calendar_context_features(frame)
        self.assertEqual(
            featured["is_construction_holiday"].tolist(), [0, 1, 1, 0, 1, 1, 1, 1]
        )
        self.assertEqual(int(featured.loc[1, "is_construction_holiday_start"]), 1)
        self.assertEqual(int(featured.loc[2, "is_construction_holiday_end"]), 1)

    def test_construction_shoulders(self):
        frame = pd.DataFrame(
            {"ds": pd.to_datetime(["2026-07-12", "2026-07-18", "2026-08-02", "2026-08-08"])}
        )
        featured = add_calendar_context_features(frame)
        self.assertEqual(featured["is_week_before_construction_holiday"].tolist(), [1, 1, 0, 0])
        self.assertEqual(featured["is_week_after_construction_holiday"].tolist(), [0, 0, 1, 1])

    def test_2024_school_starts_are_separate(self):
        frame = pd.DataFrame(
            {"ds": pd.to_datetime(["2024-08-27", "2024-08-29", "2024-09-03"])}
        )
        featured = add_calendar_context_features(frame)
        self.assertEqual(featured["is_french_school_start"].tolist(), [1, 0, 0])
        self.assertEqual(featured["is_english_school_start"].tolist(), [0, 1, 0])
        self.assertEqual(featured["is_jewish_school_start_proxy"].tolist(), [0, 0, 1])

    def test_2025_english_return_is_after_labour_day(self):
        frame = pd.DataFrame({"ds": pd.to_datetime(["2025-08-27", "2025-09-02"])})
        featured = add_calendar_context_features(frame)
        self.assertEqual(int(featured.loc[0, "is_french_school_start"]), 1)
        self.assertEqual(int(featured.loc[1, "is_english_school_start"]), 1)

    def test_first_full_week_of_march_is_break_proxy(self):
        frame = pd.DataFrame(
            {"ds": pd.to_datetime(["2025-03-02", "2025-03-03", "2025-03-07", "2025-03-08"])}
        )
        featured = add_calendar_context_features(frame)
        self.assertEqual(featured["is_french_spring_break_proxy"].tolist(), [0, 1, 1, 0])
        self.assertEqual(featured["is_english_spring_break_proxy"].tolist(), [0, 1, 1, 0])

    def test_major_jewish_dates_extend_jewish_school_proxy(self):
        frame = pd.DataFrame({"ds": pd.date_range("2024-09-25", "2024-10-31", freq="D")})
        featured = add_calendar_context_features(frame)
        self.assertGreater(int(featured["is_jewish_school_religious_break_proxy"].sum()), 0)
        religious = featured["is_jewish_school_religious_break_proxy"].astype(bool)
        self.assertTrue(
            (featured.loc[religious, "is_jewish_school_break_proxy"] == 1).all()
        )

    def test_system_aggregates_are_consistent(self):
        frame = pd.DataFrame({"ds": pd.to_datetime(["2026-07-10", "2026-08-27"])})
        featured = add_calendar_context_features(frame)
        expected_count = featured[
            [
                "is_french_school_break_proxy",
                "is_english_school_break_proxy",
                "is_jewish_school_break_proxy",
            ]
        ].sum(axis=1)
        self.assertEqual(featured["school_systems_closed_count"].tolist(), expected_count.tolist())
        self.assertTrue(featured["school_transition_intensity"].between(0, 3).all())


if __name__ == "__main__":
    unittest.main()
