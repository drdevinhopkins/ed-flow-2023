import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from canadiens_features import schedule_features, download_schedule
from backtest_canadiens_features import hourly_frame, run, score_predictions
from intraday_day_completion_model import load_hourly_flow


def schedule(start='2026-01-02T00:00:00Z'):
    return dict(coverage_start='2025-07-01', coverage_end='2026-07-01',
                fetched_at='2026-07-01T00:00:00Z', schedule_vintage='retrospective_latest',
                games=[dict(start_utc=start, home=1, playoffs=0)])


class CanadiensTests(unittest.TestCase):
    def test_window_boundaries_and_remaining_exposure(self):
        # 19:00 Montreal start; fixed 22:00 end, no realized overtime.
        stamps = pd.date_range('2026-01-01 17:00', periods=9, freq='h')
        features = schedule_features(stamps, schedule())
        self.assertEqual(features.sports_pre_2h.tolist(), [1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(features.sports_scheduled_game.tolist(), [0, 0, 1, 1, 1, 0, 0, 0, 0])
        self.assertEqual(features.sports_post_3h.tolist(), [0, 0, 0, 0, 0, 1, 1, 1, 0])
        self.assertEqual(features.sports_scheduled_hours_remaining_today.tolist(), [3,3,3,2,1,0,0,0,0])
        self.assertEqual(features.iloc[7].sports_game_day, 0)

    def test_late_game_clips_exposure_at_midnight(self):
        features = schedule_features(pd.to_datetime(['2026-01-01 22:00', '2026-01-02 00:00']),
                                     schedule('2026-01-02T03:30:00Z'))
        self.assertEqual(features.sports_scheduled_hours_remaining_today.tolist(), [1.5, 1.5])

    def test_timezone_conversion_and_dst(self):
        features = schedule_features(pd.to_datetime(['2026-03-09T00:00:00Z']),
                                     schedule('2026-03-09T00:00:00Z'))
        self.assertEqual(features.iloc[0].sports_scheduled_game, 1)
        with self.assertRaises(Exception):
            schedule_features(pd.to_datetime(['2025-11-02 01:30']), schedule())

    def test_missing_schedule_is_not_a_no_game_day(self):
        with self.assertRaisesRegex(ValueError, 'coverage'):
            schedule_features(pd.to_datetime(['2024-01-01']), schedule())
        with self.assertRaises(ValueError):
            schedule_features(pd.to_datetime(['2026-01-01']), schedule(), duration_hours=-1)

    def test_outcomes_cannot_change_features(self):
        original = schedule()
        changed = json.loads(json.dumps(original))
        changed['games'][0].update(score=999, winner='MTL', actual_end='2026-01-02T05:00Z')
        times = pd.date_range('2026-01-01 17:00', periods=9, freq='h')
        pd.testing.assert_frame_equal(schedule_features(times, original), schedule_features(times, changed))

    def test_hourly_lags_do_not_cross_missing_hours(self):
        ds = pd.to_datetime(['2026-01-01 00:00', '2026-01-01 02:00'])
        frame = pd.DataFrame(dict(ds=ds, day=ds.normalize(), Inflow_Total=[10,20],
                                  is_complete_day=[True, True]))
        hourly, _ = hourly_frame(frame, calendar_mode='basic')
        self.assertTrue(np.isnan(hourly.iloc[1].lag_total_1))
        self.assertEqual(hourly.iloc[1].lag_total_2, 10)

    def test_end_to_end_synthetic_folds(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'flow.csv'
            ds = pd.date_range('2026-01-01', periods=24*70, freq='h')
            data = pd.DataFrame(dict(ds=ds, Inflow_Total=5 + (ds.hour % 5),
                                     INFLOW_AMBULANCES=np.ones(len(ds))))
            data.to_csv(path, index=False)
            flow = load_hourly_flow(path)
            with threadpool_limits(limits=1):
                predictions = run(flow, schedule(), n_folds=2, test_days=10,
                                  min_train_days=30, max_iter=3, calendar_mode='basic')
            self.assertEqual(predictions.fold.nunique(), 2)
            self.assertEqual(set(predictions.target), {'total','ambulance','nonambulance'})
            self.assertTrue(np.isfinite(predictions[['baseline','with_sports']]).all().all())
            self.assertEqual(set(predictions.task), {'day_completion','hourly'})
            self.assertFalse(score_predictions(predictions).empty)
            for _, block in predictions.groupby('fold'):
                self.assertEqual(block.day.nunique(), 10)
                self.assertTrue((block.train_end < block.day).all())


if __name__ == '__main__':
    unittest.main()
