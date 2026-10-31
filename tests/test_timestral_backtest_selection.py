"""Seasonal evaluation must not silently score gaps or mix season boundaries."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/evaluation/backtests'))
from backtest_timestral import seasonal_cutoffs


class SeasonalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(1.0, index=pd.date_range('2025-08-01', '2026-09-01', freq='h'),
                                 columns=['a', 'b'])

    def select(self, data):
        return seasonal_cutoffs(data, year=2026, context=672, spacing=49, count=24)

    def test_balanced_nonoverlapping_windows_stay_inside_season(self):
        selected, audit = self.select(self.data)
        self.assertEqual(len(set(selected)), 96)
        self.assertTrue(all(b-a >= 24 for a, b in zip(selected, selected[1:])))
        self.assertTrue(audit[audit.selected].groupby('season').size().eq(24).all())
        seasons = {12:'winter', 1:'winter', 2:'winter', 3:'spring', 4:'spring', 5:'spring',
                   6:'summer', 7:'summer', 8:'summer', 9:'autumn', 10:'autumn', 11:'autumn'}
        for end in selected:
            months = self.data.index[end+1:end+25].month
            self.assertEqual(len({seasons[m] for m in months}), 1)

    def test_missing_future_is_rejected_even_when_history_can_be_filled(self):
        self.data.loc['2025-09-01 12:00', 'a'] = np.nan
        _, audit = self.select(self.data)
        row = audit[audit.cutoff == '2025-08-31 23:00:00'].iloc[0]
        self.assertEqual(row.eligibility, 'missing_or_negative_actual')
        self.assertFalse(row.selected)

    def test_cannot_replace_missing_season_with_other_seasons(self):
        self.data.loc['2025-12-01':'2026-02-28', :] = np.nan
        with self.assertRaisesRegex(ValueError, 'winter: only 0 eligible cutoffs'):
            self.select(self.data)

    def test_selection_does_not_depend_on_error_or_target_magnitude(self):
        before, _ = self.select(self.data)
        self.data.loc['2026-01-01':'2026-02-01', :] = 1000
        after, _ = self.select(self.data)
        self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
