import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hourly_calendar_features import add_hourly_calendar_features, scenario_columns


class HourlyCalendarFeatureTests(unittest.TestCase):
    def test_jgh_only_ramq_day_is_separate_from_nominal(self):
        frame = pd.DataFrame({"ds": pd.to_datetime(["2025-09-23 10:00"])})
        out = add_hourly_calendar_features(frame)
        self.assertEqual(int(out.loc[0, "is_jgh_ramq_holiday"]), 1)
        self.assertEqual(int(out.loc[0, "is_ramq_holiday"]), 0)
        self.assertEqual(int(out.loc[0, "is_jgh_only_ramq_holiday"]), 1)
        self.assertEqual(int(out.loc[0, "is_jgh_only_daytime"]), 1)

    def test_nominal_only_ramq_day_is_separate_from_jgh(self):
        frame = pd.DataFrame({"ds": pd.to_datetime(["2026-01-02 10:00"])})
        out = add_hourly_calendar_features(frame)
        self.assertEqual(int(out.loc[0, "is_ramq_holiday"]), 1)
        self.assertEqual(int(out.loc[0, "is_jgh_ramq_holiday"]), 0)
        self.assertEqual(int(out.loc[0, "is_nominal_only_ramq_holiday"]), 1)
        self.assertEqual(int(out.loc[0, "is_nominal_only_daytime"]), 1)

    def test_daypart_interactions(self):
        frame = pd.DataFrame(
            {"ds": pd.to_datetime(["2025-09-23 03:00", "2025-09-23 12:00", "2025-09-23 20:00"])}
        )
        out = add_hourly_calendar_features(frame)
        self.assertEqual(out["is_jgh_only_overnight"].tolist(), [1, 0, 0])
        self.assertEqual(out["is_jgh_only_daytime"].tolist(), [0, 1, 0])
        self.assertEqual(out["is_jgh_only_evening"].tolist(), [0, 0, 1])

    def test_scenarios_are_nested(self):
        demand = set(scenario_columns("demand_calendar"))
        mismatch = set(scenario_columns("demand_plus_jgh_mismatch"))
        interactions = set(scenario_columns("demand_plus_jgh_interactions"))
        self.assertTrue(demand < mismatch)
        self.assertTrue(mismatch < interactions)


if __name__ == "__main__":
    unittest.main()
