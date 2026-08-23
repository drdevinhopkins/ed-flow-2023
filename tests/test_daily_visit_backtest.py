import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_holiday_features import load_daily_visits


class DailyVisitAggregationTests(unittest.TestCase):
    @staticmethod
    def _rows(day: str, hours: int, value: float) -> list[dict[str, object]]:
        start = pd.Timestamp(day)
        return [
            {"ds": start + pd.Timedelta(hours=hour), "Inflow_Total": value}
            for hour in range(hours)
        ]

    def _write_csv(self, rows: list[dict[str, object]]) -> str:
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        pd.DataFrame(rows).to_csv(handle.name, index=False)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_sums_complete_days_and_drops_partial_trailing_day(self):
        rows = []
        rows += self._rows("2026-01-01", 24, 1)
        rows += self._rows("2026-01-02", 24, 2)
        rows += self._rows("2026-01-03", 12, 3)

        daily = load_daily_visits(self._write_csv(rows))

        self.assertEqual(daily["ds"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-01", "2026-01-02"])
        self.assertEqual(daily["daily_visits"].tolist(), [24.0, 48.0])

    def test_rejects_incomplete_day_inside_history(self):
        rows = []
        rows += self._rows("2026-01-01", 24, 1)
        rows += self._rows("2026-01-02", 12, 2)
        rows += self._rows("2026-01-03", 24, 1)

        with self.assertRaisesRegex(ValueError, "Incomplete daily inflow observations"):
            load_daily_visits(self._write_csv(rows))


if __name__ == "__main__":
    unittest.main()
