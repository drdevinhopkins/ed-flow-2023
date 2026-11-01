import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from hourly_report_rows import collapse_report_rows, merge_report_history
from daily_arrival_quality import build_daily_inflow_outputs


def report():
    # Values from the 2026-11-01 report, newest first.
    return pd.DataFrame({
        'ds': ['2026-11-01 03:00', '2026-11-01 02:00',
               '2026-11-01 02:00', '2026-11-01 01:00'],
        'Inflow_Total': [6, 0, 8, 7], 'TTStr': [105, 98, 101, 98],
        'Inflow_Cum_Total': [21, 15, 15, 7],
        'CUM_ADMREQ': [6, 4, 0, 2],
    })


def test_actual_dst_report_uses_maximum_of_every_column():
    result = collapse_report_rows(report())
    assert result.ds.is_unique
    assert len(result) == 3
    row = result.set_index('ds').loc['2026-11-01 02:00']
    assert row.to_dict() == {'Inflow_Total': 8, 'TTStr': 101,
                              'Inflow_Cum_Total': 15, 'CUM_ADMREQ': 4}
    assert result.columns.tolist() == report().columns.tolist()


def test_input_order_does_not_change_result():
    pd.testing.assert_frame_equal(collapse_report_rows(report()),
                                  collapse_report_rows(report().iloc[::-1]))


def test_ordinary_report_is_unchanged_except_for_sorting_and_parsing():
    raw = report().drop(index=2)
    expected = raw.assign(ds=pd.to_datetime(raw.ds)).sort_values('ds').reset_index(drop=True)
    pd.testing.assert_frame_equal(collapse_report_rows(raw), expected)


def test_repair_restores_eight_arrivals_and_maximum_occupancy():
    history = pd.DataFrame({'ds': ['2026-10-31 23:00', '2026-11-01 02:00'],
                            'Inflow_Total': [6, 0], 'TTStr': [102, 98]})
    result = merge_report_history(history, report()).set_index('ds')
    assert result.loc['2026-11-01 02:00', 'Inflow_Total'] == 8
    assert result.loc['2026-11-01 02:00', 'TTStr'] == 101
    assert result.loc['2026-10-31 23:00', 'Inflow_Total'] == 6
    assert result.index.is_unique


def test_repeated_refresh_is_idempotent():
    first = merge_report_history(report().iloc[:0], report())
    second = merge_report_history(first, report())
    pd.testing.assert_frame_equal(first, second)


def test_later_downward_revision_replaces_history_rather_than_taking_maximum():
    history = collapse_report_rows(report())
    fresh = report().drop(index=2).copy()
    fresh.loc[fresh.ds.eq('2026-11-01 03:00'), 'TTStr'] = 99
    result = merge_report_history(history, fresh).set_index('ds')
    assert result.loc['2026-11-01 03:00', 'TTStr'] == 99


def test_unverified_dst_day_remains_excluded_from_daily_training():
    raw = pd.DataFrame({'ds': pd.date_range('2026-11-01', periods=24, freq='h'),
                        'Inflow_Total': 2})
    daily, quality = build_daily_inflow_outputs(collapse_report_rows(raw),
                                               now=pd.Timestamp('2026-11-02T18:00Z'))
    assert pd.isna(daily.iloc[0].Daily_Inflow_Total)
    assert quality.iloc[0].quality_status == 'dst_requires_verification'
    assert not quality.iloc[0].audit_eligible


def test_invalid_timestamp_is_rejected():
    with pytest.raises(ValueError):
        collapse_report_rows(pd.DataFrame({'ds': [None], 'Inflow_Total': [1]}))
