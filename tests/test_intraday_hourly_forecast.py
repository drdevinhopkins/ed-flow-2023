import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from forecast_intraday_daily_inflow import (  # noqa: E402
    DataQualityError,
    write_forecast,
    validate_live_flow,
)


class IntradayHourlyForecastTests(unittest.TestCase):
    def _live_flow(self, *, latest_hour: int = 16) -> pd.DataFrame:
        rows = []
        for source_order, hour in enumerate(range(latest_hour + 1)):
            rows.append(
                {
                    "ds": pd.Timestamp("2026-09-03") + pd.Timedelta(hours=hour),
                    "day": pd.Timestamp("2026-09-03"),
                    "_source_order": source_order,
                    "Inflow_Total": 10.0,
                    "TRG_HALLWAY_TBS": 1.0,
                    "POD_GREEN_TBS": 2.0,
                    "POD_YELLOW_TBS": 3.0,
                    "POD_ORANGE_TBS": 4.0,
                    "RAZ_TBS": 5.0,
                    "AMBVERTTBS": 6.0,
                    "QTrack_TBS": 7.0,
                    "Garage_TBS": 8.0,
                }
            )
        return pd.DataFrame(rows)

    def test_accepts_fresh_contiguous_current_day(self):
        flow = self._live_flow()

        latest, current = validate_live_flow(
            flow,
            now=pd.Timestamp("2026-09-03T20:30:00Z"),
        )

        self.assertEqual(latest, pd.Timestamp("2026-09-03 16:00:00"))
        self.assertEqual(current["Inflow_Total"].sum(), 170.0)

    def test_missing_hour_suppresses_forecast(self):
        flow = self._live_flow().loc[lambda frame: frame["ds"].dt.hour.ne(8)]

        with self.assertRaisesRegex(DataQualityError, "missing or out-of-order"):
            validate_live_flow(
                flow,
                now=pd.Timestamp("2026-09-03T20:30:00Z"),
            )

    def test_stale_input_suppresses_forecast(self):
        with self.assertRaisesRegex(DataQualityError, "stale"):
            validate_live_flow(
                self._live_flow(),
                now=pd.Timestamp("2026-09-03T23:00:00Z"),
            )

    def test_invalid_inflow_suppresses_forecast(self):
        flow = self._live_flow()
        flow.loc[3, "Inflow_Total"] = -1.0

        with self.assertRaisesRegex(DataQualityError, "negative"):
            validate_live_flow(
                flow,
                now=pd.Timestamp("2026-09-03T20:30:00Z"),
            )

    def test_output_is_one_replaceable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.csv"
            first = {
                "observed_arrivals": 100.0,
                "predicted_total": 200.0,
                "p10_total": 180.0,
                "p90_total": 220.0,
            }
            second = {
                "observed_arrivals": 120.0,
                "predicted_total": 205.0,
                "p10_total": 190.0,
                "p90_total": 225.0,
            }

            write_forecast(path, first)
            write_forecast(path, second)
            output = pd.read_csv(path)

            self.assertEqual(len(output), 1)
            self.assertEqual(output.loc[0, "observed_arrivals"], 120.0)
            self.assertLessEqual(output.loc[0, "p10_total"], output.loc[0, "predicted_total"])
            self.assertLessEqual(output.loc[0, "predicted_total"], output.loc[0, "p90_total"])


if __name__ == "__main__":
    unittest.main()
