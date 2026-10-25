import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from anomaly_refresh import publish_file, validate_ranges, validate_source
from validate_forecast_inputs import ANOMALY_TARGET_ALIASES

NOW = pd.Timestamp('2026-10-25 07:25')


def ranges(periods=72):
    df = pd.DataFrame({'ds': pd.date_range('2026-10-25 08:00', periods=periods, freq='h')})
    for alias in ANOMALY_TARGET_ALIASES.values():
        for suffix, value in [('yhat', 10), ('yhat_lower', 5), ('yhat_upper', 15)]:
            df[f'{alias}_{suffix}'] = value
    return df


class RefreshTests(unittest.TestCase):
    def test_healthy_ranges(self):
        validate_ranges(ranges(), now=NOW)

    def test_reported_four_hour_shortfall(self):
        with self.assertRaises(ValueError):
            validate_ranges(ranges(20), now=NOW)

    def test_daily_refresh_requires_headroom(self):
        with self.assertRaises(ValueError):
            validate_ranges(ranges(24), now=NOW)

    def test_missing_target_or_null_interval(self):
        df = ranges().drop(columns=['total_tbs_yhat'])
        with self.assertRaises(ValueError):
            validate_ranges(df, now=NOW)
        df = ranges()
        df.loc[0, 'total_tbs_yhat_lower'] = float('nan')
        with self.assertRaises(ValueError):
            validate_ranges(df, now=NOW)

    def test_stale_source(self):
        df = pd.DataFrame({'ds': pd.date_range('2026-10-24', periods=24, freq='h')})
        with self.assertRaises(ValueError):
            validate_source(df, now=NOW)

    def test_publish_and_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ranges.csv'
            ranges().to_csv(path, index=False)
            dbx = Mock()
            response = Mock(content=path.read_bytes())
            dbx.files_download.return_value = (None, response)
            with patch('anomaly_refresh.now_local', return_value=NOW):
                publish_file(dbx, path, verify_ranges=True)
            response.close.assert_called_once()
            dbx.files_upload.assert_called_once()
            response.content = b'stale data'
            with self.assertRaises(RuntimeError):
                publish_file(dbx, path)
            dbx.files_upload.side_effect = RuntimeError('upload failed')
            with self.assertRaises(RuntimeError):
                publish_file(dbx, path)

    def test_invalid_ranges_never_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ranges.csv'
            ranges(20).to_csv(path, index=False)
            dbx = Mock()
            with patch('anomaly_refresh.now_local', return_value=NOW):
                with self.assertRaises(ValueError):
                    publish_file(dbx, path, verify_ranges=True)
            dbx.files_upload.assert_not_called()


if __name__ == '__main__':
    unittest.main()
