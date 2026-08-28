import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "evaluation" / "prospective"))
sys.path.insert(0, str(ROOT / "scripts" / "evaluation" / "backtests"))

from backtest_intraday_day_completion import load_hourly_flow
from run_shadow_intraday_day_completion import (
    DataQualityError,
    _append_forecast,
    validate_live_flow,
)


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


if __name__ == "__main__":
    unittest.main()
