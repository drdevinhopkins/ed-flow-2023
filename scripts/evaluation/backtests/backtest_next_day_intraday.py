#!/usr/bin/env python3
"""Isolated next-day arrivals experiment using only the observed same-day prefix.

Compare the same calendar/history ridge model with and without intraday features.
Fit each issue using only target days completed before that issue's local date.
No operational forecast, weather route, daily target or published file is modified.
The completion estimate is a predictor, never an observed training target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from daily_arrival_quality import build_daily_inflow_outputs, QUALITY_VERSION

SLOTS = {'morning': 6, 'evening': 18}
HISTORY_FEATURES = ['weekday_baseline', 'recent_7_mean', 'recent_28_mean',
                    'target_lag_7', 'target_lag_14', 'weekday_sin', 'weekday_cos',
                    'year_sin', 'year_cos']
INTRADAY_FEATURES = ['observed_today', 'inflow_last_3h', 'inflow_last_6h',
                     'pace_ratio', 'estimated_today_total']
MODEL_VERSION = 'next-day-prefix-ridge-v1'


def _prefixes(hourly, hour):
    values = {}
    for day, group in hourly.groupby(hourly.ds.dt.normalize()):
        prefix = group.loc[group.ds.dt.hour.le(hour)].sort_values('ds')
        if prefix.ds.tolist() != list(pd.date_range(day, periods=hour + 1, freq='h')):
            continue
        inflow = prefix.Inflow_Total.to_numpy(dtype=float)
        if not np.isfinite(inflow).all() or (inflow < 0).any():
            continue
        values[day] = (inflow.sum(), inflow[-3:].sum(), inflow[-6:].sum())
    return values


def _weekday_mean(history, day):
    same = history.loc[history.index.weekday == day.weekday()].tail(8)
    return float((same if not same.empty else history.tail(28)).mean())


def build_examples(hourly, *, now):
    """Construct local-time issue examples; never use later hours as predictors."""
    hourly = hourly[['ds', 'Inflow_Total']].copy()
    hourly['ds'] = pd.to_datetime(hourly.ds, format='mixed', errors='raise')
    hourly['Inflow_Total'] = pd.to_numeric(hourly.Inflow_Total, errors='coerce')
    # Shared validation rejects aware/off-hour/future timestamps and invalid targets.
    daily, _ = build_daily_inflow_outputs(hourly, now=now)
    actual = daily.set_index('ds').Daily_Inflow_Total
    rows = []
    for slot, hour in SLOTS.items():
        prefixes = _prefixes(hourly, hour)
        prefix_totals = pd.Series({day: values[0] for day, values in prefixes.items()}, dtype=float)
        for day, (observed, last3, last6) in sorted(prefixes.items()):
            history = actual.loc[actual.index < day].dropna()
            if len(history) < 28 or not np.isfinite(actual.get(day - pd.Timedelta(days=1), np.nan)):
                continue
            target = day + pd.Timedelta(days=1)
            recent7 = history.reindex(pd.date_range(day - pd.Timedelta(days=7), periods=7))
            recent28 = history.reindex(pd.date_range(day - pd.Timedelta(days=28), periods=28))
            if recent7.isna().any() or recent28.isna().any():
                continue
            prior_prefix = prefix_totals.loc[(prefix_totals.index < day)
                                              & prefix_totals.index.isin(history.index)]
            same = prior_prefix.loc[prior_prefix.index.weekday == day.weekday()].tail(8)
            expected_prefix = float((same if len(same) >= 4 else prior_prefix.tail(28)).mean())
            if not np.isfinite(expected_prefix) or expected_prefix <= 0:
                continue
            today_baseline = _weekday_mean(history, day)
            row = dict(issue_day=day, target_date=target, forecast_issue_slot=slot,
                       cutoff_hour=hour, target_actual=actual.get(target, np.nan),
                       weekday_baseline=_weekday_mean(history, target),
                       recent_7_mean=recent7.mean(), recent_28_mean=recent28.mean(),
                       target_lag_7=actual.get(target - pd.Timedelta(days=7), np.nan),
                       target_lag_14=actual.get(target - pd.Timedelta(days=14), np.nan),
                       weekday_sin=np.sin(2 * np.pi * target.weekday() / 7),
                       weekday_cos=np.cos(2 * np.pi * target.weekday() / 7),
                       year_sin=np.sin(2 * np.pi * target.dayofyear / 365.25),
                       year_cos=np.cos(2 * np.pi * target.dayofyear / 365.25),
                       observed_today=observed, inflow_last_3h=last3, inflow_last_6h=last6,
                       pace_ratio=observed / expected_prefix,
                       estimated_today_total=today_baseline * observed / expected_prefix)
            if np.isfinite([row[key] for key in HISTORY_FEATURES + INTRADAY_FEATURES]).all():
                rows.append(row)
    columns = ['issue_day', 'target_date', 'forecast_issue_slot', 'cutoff_hour', 'target_actual',
               *HISTORY_FEATURES, *INTRADAY_FEATURES]
    return pd.DataFrame(rows, columns=columns).sort_values(['issue_day', 'cutoff_hour']).reset_index(drop=True)


def predict_issue(examples, row, *, intraday, min_train_days=365, context_days=1095, alpha=20.0):
    """Training labels stop yesterday; centering/scaling also use training rows only."""
    train = examples.loc[
        examples.forecast_issue_slot.eq(row.forecast_issue_slot)
        & examples.target_date.lt(row.issue_day)
        & examples.issue_day.ge(row.issue_day - pd.Timedelta(days=context_days))
        & examples.target_actual.notna()
    ]
    if len(train) < min_train_days:
        raise ValueError(f'Only {len(train)} completed training outcomes; need {min_train_days}')
    features = HISTORY_FEATURES + (INTRADAY_FEATURES if intraday else [])
    x = train[features].to_numpy(dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-9] = 1
    x = (x - center) / scale
    residual = (train.target_actual - train.weekday_baseline).to_numpy(dtype=float)
    intercept = residual.mean()
    coefficients = np.linalg.solve(x.T @ x + alpha * np.eye(len(features)), x.T @ (residual - intercept))
    query = np.array([getattr(row, name) for name in features], dtype=float)
    prediction = max(0.0, float(row.weekday_baseline + intercept + ((query - center) / scale) @ coefficients))
    return prediction, len(train), train.target_date.max()


def backtest(examples, *, test_days=730, min_train_days=365):
    matured = examples.loc[examples.target_actual.notna()]
    if matured.empty:
        return pd.DataFrame()
    first_issue = matured.issue_day.max() - pd.Timedelta(days=test_days - 1)
    outputs = []
    for row in matured.loc[matured.issue_day.ge(first_issue)].itertuples(index=False):
        try:
            control, n, train_end = predict_issue(examples, row, intraday=False, min_train_days=min_train_days)
            candidate, _, _ = predict_issue(examples, row, intraday=True, min_train_days=min_train_days)
        except ValueError:
            continue
        outputs.append(dict(issue_day=row.issue_day, target_date=row.target_date,
                            forecast_issue_slot=row.forecast_issue_slot, actual=row.target_actual,
                            weekday_baseline=row.weekday_baseline, history_control=control,
                            intraday_candidate=candidate, training_days=n, training_target_end=train_end,
                            observed_today=row.observed_today, estimated_today_total=row.estimated_today_total))
    return pd.DataFrame(outputs)


def summarize(detail):
    rows = []
    if detail.empty:
        return pd.DataFrame(columns=['forecast_issue_slot', 'period', 'n', 'model', 'mae', 'rmse', 'bias'])
    scored = detail.copy()
    scored['quarter'] = pd.to_datetime(scored.issue_day).dt.to_period('Q').astype(str)
    for slot, slot_rows in scored.groupby('forecast_issue_slot'):
        for period, group in [('all', slot_rows), *list(slot_rows.groupby('quarter'))]:
            for model in ['weekday_baseline', 'history_control', 'intraday_candidate']:
                error = group[model] - group.actual
                rows.append(dict(forecast_issue_slot=slot, period=period, n=len(group), model=model,
                                 mae=error.abs().mean(), rmse=np.sqrt((error ** 2).mean()), bias=error.mean()))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hourly-csv', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--test-days', type=int, default=730, help='Calendar days ending at latest matured issue')
    parser.add_argument('--min-train-days', type=int, default=365)
    parser.add_argument('--now', help='Aware timestamp for reproducible source-quality checks')
    args = parser.parse_args()
    if args.test_days < 1 or args.min_train_days < 28:
        raise ValueError('Require positive test-days and at least 28 training days')
    outputs = [args.output_dir / name for name in ['detail.csv', 'summary.csv', 'provenance.json']]
    if args.hourly_csv.resolve() in {path.resolve() for path in outputs}:
        raise ValueError('Experiment outputs must not overwrite hourly source')
    now = pd.Timestamp(args.now) if args.now else pd.Timestamp.now(tz='UTC')
    examples = build_examples(pd.read_csv(args.hourly_csv), now=now)
    detail = backtest(examples, test_days=args.test_days, min_train_days=args.min_train_days)
    summary = summarize(detail)
    if detail.empty:
        raise ValueError('No eligible time-ordered test issues')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(outputs[0], index=False)
    summary.to_csv(outputs[1], index=False)
    provenance = dict(model_version=MODEL_VERSION, target_quality_version=QUALITY_VERSION,
                      source_sha256=hashlib.sha256(args.hourly_csv.read_bytes()).hexdigest(),
                      audit_at_utc=now.isoformat(), test_days=args.test_days, min_train_days=args.min_train_days,
                      slots=SLOTS, ridge_alpha=20.0, examples=len(examples), scored_issues=len(detail),
                      first_scored_issue=str(detail.issue_day.min().date()),
                      last_scored_issue=str(detail.issue_day.max().date()),
                      distinct_scored_dates=int(detail.issue_day.nunique()),
                      limitation='Historical hourly snapshots may contain later revisions. No archived Chronos '
                      'or forecast-time weather comparator; retrospective experiment only. No production publication.')
    outputs[2].write_text(json.dumps(provenance, indent=2) + '\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
