"""Check forecast-time availability, paired windows, and scoring alignment."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/evaluation/backtests"))
import backtest_prophet_chronos_daily as bt


def calendar_stub(frame):
    frame = frame.copy()
    for name in bt.REGRESSORS:
        frame[name] = 0.0
    return frame


class FakeProphet:
    fits = []

    def __init__(self, **kwargs):
        self.yearly = kwargs["yearly_seasonality"]

    def add_regressor(self, *args, **kwargs):
        pass

    def fit(self, frame, **kwargs):
        self.frame = frame.copy()
        self.fits.append(frame.copy())

    def predict(self, frame):
        return pd.DataFrame({"yhat": np.repeat(self.frame.y.mean(), len(frame)),
                             "trend": np.repeat(self.frame.y.mean(), len(frame)),
                             "weekly": 0.0, "ds": frame.ds})


class ProphetChronosTests(unittest.TestCase):
    def setUp(self):
        FakeProphet.fits = []
        self.daily = pd.DataFrame({"ds": pd.date_range("2020-01-01", periods=1500),
                                   bt.TARGET: 200 + np.arange(1500) % 17})
        self.patch = patch.object(bt, "calendar", calendar_stub)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_future_target_changes_do_not_change_origin_forecast(self):
        origin = self.daily.ds.iloc[800]
        altered = self.daily.copy()
        altered.loc[altered.ds.gt(origin), bt.TARGET] = 99999
        first = bt.ProphetVintages(self.daily, model_factory=FakeProphet).predict(origin)
        second = bt.ProphetVintages(altered, model_factory=FakeProphet).predict(origin)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(all(frame.ds.max() <= origin for frame in FakeProphet.fits))

    def test_historical_features_are_forecasts_and_omit_missing_targets(self):
        self.daily.loc[500, bt.TARGET] = np.nan
        vintages = bt.ProphetVintages(self.daily, model_factory=FakeProphet)
        dates = self.daily.ds.iloc[800:830]
        features = vintages.historical(dates)
        self.assertTrue((features.prophet_origin < features.ds).all())
        self.assertTrue(((features.ds - features.prophet_origin).dt.days <= 7).all())
        self.assertTrue(all(not f.y.isna().any() for f in FakeProphet.fits))
        for audit in vintages.audit:
            self.assertLessEqual(audit["train_end"], audit["origin"])
        count = len(FakeProphet.fits)
        vintages.historical(dates)
        self.assertEqual(count, len(FakeProphet.fits))

    def test_gap_is_hard_chronos_boundary(self):
        self.daily.loc[900, bt.TARGET] = np.nan
        history = bt.history_at(self.daily, self.daily.ds.iloc[1050], 365, 120)
        self.assertEqual(history.ds.min(), self.daily.ds.iloc[901])
        with self.assertRaises(ValueError):
            bt.history_at(self.daily, self.daily.ds.iloc[1000], 365, 120)

    def test_selection_covers_seasons_and_nonoverlapping_complete_horizons(self):
        self.daily.loc[900, bt.TARGET] = np.nan
        cutoffs = bt.select_cutoffs(self.daily, 7, 365, 120, 365, 1095, 24)
        self.assertEqual(set(cutoffs.season), {"winter", "spring", "summer", "fall"})
        self.assertTrue(cutoffs.cutoff.diff().dropna().dt.days.ge(7).all())
        for row in cutoffs.itertuples():
            future = self.daily.set_index("ds")[bt.TARGET].reindex(pd.date_range(row.cutoff + pd.Timedelta(days=1), periods=7))
            self.assertFalse(future.isna().any())
            self.assertGreaterEqual(row.history_days, 120)

    def test_chronos_predictions_align_dates_and_exclude_audit_columns(self):
        class Pipeline:
            def predict_df(inner, history, **kwargs):
                self.assertNotIn("prophet_origin", history)
                self.assertNotIn("prophet_origin", kwargs["future_df"])
                return pd.DataFrame({"ds": pd.to_datetime(["2024-01-03", "2024-01-02"]), "predictions": [203., 202.]})
        history = pd.DataFrame({"ds": pd.to_datetime(["2024-01-01"]), bt.TARGET: [200.], "prophet_yhat": [199.], "prophet_origin": pd.to_datetime(["2023-12-31"])})
        future = pd.DataFrame({"ds": pd.to_datetime(["2024-01-02", "2024-01-03"]), "prophet_yhat": [201., 202.], "prophet_origin": pd.to_datetime(["2024-01-01"] * 2)})
        np.testing.assert_array_equal(bt.chronos_predict(Pipeline(), history, future, ["prophet_yhat"], 2), [202., 203.])

    def test_missing_dates_are_restored_as_gaps(self):
        prepared = bt.prepare_daily(self.daily.drop(index=50))
        self.assertTrue(pd.isna(prepared.loc[50, bt.TARGET]))
        with self.assertRaises(ValueError):
            bt.prepare_daily(pd.concat([self.daily, self.daily.head(1)]))


if __name__ == "__main__":
    unittest.main()
