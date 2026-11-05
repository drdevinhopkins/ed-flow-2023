"""Availability and outcome boundaries for the isolated next-day experiment."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/evaluation/backtests'))
import backtest_next_day_intraday as experiment

NOW = pd.Timestamp('2026-08-10T18:00Z')
ISSUE_DAY = pd.Timestamp('2026-07-20')


def source():
    ds = pd.date_range('2026-05-01', '2026-08-08T23:00', freq='h')
    return pd.DataFrame({'ds': ds, 'Inflow_Total': 2.0 + ds.weekday * 0.1 + ds.hour * 0.02})


def issue(examples, slot='evening'):
    return next(examples.loc[examples.issue_day.eq(ISSUE_DAY)
                              & examples.forecast_issue_slot.eq(slot)].itertuples(index=False))


def test_uses_only_prefix_and_completed_training_labels():
    hourly = source()
    examples = experiment.build_examples(hourly, now=NOW)
    row = issue(examples)
    observed = hourly.loc[hourly.ds.between(ISSUE_DAY, ISSUE_DAY + pd.Timedelta(hours=18)), 'Inflow_Total'].sum()
    assert np.isclose(row.observed_today, observed)
    assert row.target_date == ISSUE_DAY + pd.Timedelta(days=1)
    prediction, n, end = experiment.predict_issue(examples, row, intraday=True, min_train_days=28)
    assert np.isfinite(prediction) and prediction >= 0
    assert n >= 28 and end < ISSUE_DAY


def test_later_hours_today_and_future_outcomes_cannot_change_issue_prediction():
    original = experiment.build_examples(source(), now=NOW)
    changed = source()
    changed.loc[changed.ds.gt(ISSUE_DAY + pd.Timedelta(hours=18)), 'Inflow_Total'] = 9999
    rebuilt = experiment.build_examples(changed, now=NOW)
    before, after = issue(original), issue(rebuilt)
    for name in experiment.HISTORY_FEATURES + experiment.INTRADAY_FEATURES:
        assert np.isclose(getattr(before, name), getattr(after, name)), name
    a = experiment.predict_issue(original, before, intraday=True, min_train_days=28)[0]
    b = experiment.predict_issue(rebuilt, after, intraday=True, min_train_days=28)[0]
    assert np.isclose(a, b)
    assert before.target_actual != after.target_actual


def test_missing_or_duplicate_prefix_never_becomes_a_zero_arrival_hour():
    for problem in ['missing', 'duplicate', 'negative']:
        hourly = source()
        stamp = ISSUE_DAY + pd.Timedelta(hours=8)
        if problem == 'missing':
            hourly = hourly.loc[hourly.ds.ne(stamp)]
        elif problem == 'duplicate':
            hourly = pd.concat([hourly, hourly.loc[hourly.ds.eq(stamp)]])
        else:
            hourly.loc[hourly.ds.eq(stamp), 'Inflow_Total'] = -1
        examples = experiment.build_examples(hourly, now=NOW)
        rows = examples.loc[examples.issue_day.eq(ISSUE_DAY)]
        assert rows.forecast_issue_slot.tolist() == ['morning']


def test_same_day_target_is_not_a_completed_training_label():
    examples = experiment.build_examples(source(), now=NOW)
    row = issue(examples)
    before = experiment.predict_issue(examples, row, intraday=True, min_train_days=28)
    changed = examples.copy()
    changed.loc[changed.target_date.ge(ISSUE_DAY), 'target_actual'] = 1e9
    after = experiment.predict_issue(changed, row, intraday=True, min_train_days=28)
    assert np.isclose(before[0], after[0])
    assert before[1:] == after[1:]


def test_test_days_is_a_calendar_window_rather_than_last_eligible_rows():
    examples = experiment.build_examples(source(), now=NOW)
    scored = experiment.backtest(examples, test_days=7, min_train_days=28)
    assert not scored.empty
    last_matured = examples.loc[examples.target_actual.notna(), 'issue_day'].max()
    assert scored.issue_day.min() >= last_matured - pd.Timedelta(days=6)
    assert scored.issue_day.nunique() <= 7
