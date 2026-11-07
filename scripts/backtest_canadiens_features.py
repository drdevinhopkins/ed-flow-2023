#!/usr/bin/env python3
"""Isolated Canadiens ablation: rolling one-hour arrivals and day completion.

Requires an explicitly supplied local ED CSV. Never downloads ED data or writes
production forecasts. This is a screening model, not the locked production ensemble.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from canadiens_features import schedule_features
from intraday_day_completion_model import (
    load_hourly_flow, build_snapshots, feature_sets, build_weather_features,
    build_expanding_folds, fit_completion_curve, add_curve_features, add_calendar_features,
)


def hourly_frame(flow: pd.DataFrame, weather=None, calendar_mode='rich') -> tuple[pd.DataFrame, list[str]]:
    """Strict clock-lag joins; missing hours never become previous-hour observations."""
    frame = flow[['ds', 'day', 'Inflow_Total', 'is_complete_day']].copy()
    frame = frame.loc[frame.is_complete_day].copy()
    lookup = flow.loc[flow.is_complete_day].set_index('ds')
    if not lookup.index.is_unique:
        raise ValueError('Duplicate local timestamps; resolve the source DST contract first')
    targets = {'total': 'Inflow_Total'}
    if 'INFLOW_AMBULANCES' in lookup:
        targets['ambulance'] = 'INFLOW_AMBULANCES'
    features = []
    for target, column in targets.items():
        values = pd.to_numeric(lookup[column], errors='coerce')
        frame['actual_' + target] = frame.ds.map(values)
        for hours in (1, 2, 24, 168):
            name = f'lag_{target}_{hours}'
            frame[name] = (frame.ds - pd.Timedelta(hours=hours)).map(values)
            features.append(name)
    if 'ambulance' in targets:
        frame['actual_nonambulance'] = frame.actual_total - frame.actual_ambulance
        valid = frame.actual_ambulance.ge(0) & frame.actual_nonambulance.ge(0)
        frame.loc[~valid, ['actual_ambulance', 'actual_nonambulance']] = np.nan
    for name, values in [('hour', frame.ds.dt.hour), ('weekday', frame.ds.dt.dayofweek),
                         ('month', frame.ds.dt.month),
                         ('trend', (frame.day - frame.day.min()).dt.days),
                         ('year_sin', np.sin(2 * np.pi * frame.ds.dt.dayofyear / 365.25)),
                         ('year_cos', np.cos(2 * np.pi * frame.ds.dt.dayofyear / 365.25))]:
        frame['calendar_' + name] = values
        features.append('calendar_' + name)
    calendar = add_calendar_features(frame.day.drop_duplicates(), calendar_mode)
    extra = [c for c in calendar if c != 'day' and c not in frame]
    frame = frame.merge(calendar[['day', *extra]], on='day', how='left', validate='many_to_one')
    features += extra
    if weather is not None:
        # One-hour-ahead: weather observations may only be used up to origin ds-1h.
        shifted = weather.copy()
        shifted['ds'] += pd.Timedelta(hours=1)
        frame = frame.merge(shifted, on='ds', how='left', validate='one_to_one')
        features += [c for c in frame if c.startswith('weather_')]
    return frame.reset_index(drop=True), features


def score_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = list(predictions.groupby(['task', 'target', 'fold']))
    groups += [((task, target, 'all'), group) for (task, target), group
               in predictions.groupby(['task', 'target'])]
    for (task, target, fold), group in groups:
        subsets = {'all': np.ones(len(group), dtype=bool),
                   'game_day': group.sports_game_day.eq(1),
                   'during': group.sports_scheduled_game.eq(1),
                   'after': group.sports_post_3h.eq(1),
                   'non_game_day': group.sports_game_day.eq(0)}
        for subset, mask in subsets.items():
            block = group.loc[mask]
            if block.empty:
                continue
            base = np.abs(block.actual - block.baseline)
            sports = np.abs(block.actual - block.with_sports)
            paired = pd.DataFrame({'day': block.day, 'delta': sports - base})
            daily = paired.groupby('day').delta.agg(['sum', 'count'])
            rng = np.random.default_rng(42)
            draws = rng.integers(0, len(daily), size=(1000, len(daily)))
            boot = daily['sum'].to_numpy()[draws].sum(axis=1) / daily['count'].to_numpy()[draws].sum(axis=1)
            low, high = np.quantile(boot, [.025, .975])
            rows.append(dict(task=task, target=target, fold=fold, subset=subset,
                             n=len(block), days=block.day.nunique(),
                             baseline_mae=float(base.mean()), sports_mae=float(sports.mean()),
                             mae_change=float(sports.mean() - base.mean()),
                             day_bootstrap_delta_low=float(low), day_bootstrap_delta_high=float(high),
                             baseline_bias=float((block.baseline - block.actual).mean()),
                             sports_bias=float((block.with_sports - block.actual).mean())))
    return pd.DataFrame(rows)


def run(flow, schedule, *, weather=None, n_folds=4, test_days=180,
        min_train_days=365, max_iter=150, duration_hours=3, calendar_mode='rich',
        alignment='interval-end'):
    flow = flow.copy()
    valid = np.isfinite(flow.Inflow_Total) & flow.Inflow_Total.ge(0)
    flow['is_complete_day'] &= valid.groupby(flow.day).transform('all')
    snapshots = build_snapshots(flow, calendar_mode=calendar_mode, weather=weather)
    snapshots = snapshots.loc[snapshots.cutoff_hour.between(6, 22)].reset_index(drop=True)
    hourly, hourly_features = hourly_frame(flow, weather, calendar_mode)
    # Existing day-completion target/date convention remains untouched. Exposure is
    # evaluated at report cutoff; hourly interval ending ds uses its midpoint ds-30m.
    for frame, stamps in [(snapshots, snapshots.ds),
                          (hourly, hourly.ds - pd.Timedelta(minutes=30)
                           if alignment == 'interval-end' else hourly.ds + pd.Timedelta(minutes=30))]:
        covariates = schedule_features(stamps, schedule, duration_hours)
        for column in covariates:
            frame[column] = covariates[column].to_numpy()
    sports_features = [c for c in snapshots if c.startswith('sports_')]
    # Shared complete-day folds, so every task uses the same held-out calendar days.
    folds = build_expanding_folds(snapshots, n_folds=n_folds, test_days=test_days,
                                  min_train_days=min_train_days)
    predictions = []
    for fold, train_mask, test_mask in folds:
        train_days = snapshots.loc[train_mask, 'day'].unique()
        test_dates = snapshots.loc[test_mask, 'day'].unique()
        train = snapshots.loc[train_mask].copy()
        test = snapshots.loc[test_mask].copy()
        curve = fit_completion_curve(train)
        train = add_curve_features(train, curve)
        test = add_curve_features(test, curve)
        tasks = [('day_completion', 'total', train, test,
                  feature_sets(train)['boosted_full'], 'remaining_arrivals')]
        htrain = hourly.loc[hourly.day.isin(train_days)]
        htest = hourly.loc[hourly.day.isin(test_dates)]
        for column in [c for c in hourly if c.startswith('actual_')]:
            target = column.removeprefix('actual_')
            tasks.append(('hourly', target, htrain, htest, hourly_features, column))
        for task, target, train, test, features, label in tasks:
            train = train.loc[np.isfinite(train[label]) & train[label].ge(0)]
            test = test.loc[np.isfinite(test[label]) & test[label].ge(0)].copy()
            if train.empty or test.empty:
                continue
            result = test[['ds', 'day', *sports_features]].copy()
            result['actual'] = test[label]
            for name, columns in [('baseline', features),
                                  ('with_sports', features + sports_features)]:
                model = HistGradientBoostingRegressor(
                    max_iter=max_iter, learning_rate=.05, max_leaf_nodes=15,
                    l2_regularization=1, early_stopping=False, random_state=42)
                model.fit(train[columns].replace([np.inf, -np.inf], np.nan), train[label])
                result[name] = np.maximum(0, model.predict(
                    test[columns].replace([np.inf, -np.inf], np.nan)))
            if task == 'day_completion':
                for column in ['actual', 'baseline', 'with_sports']:
                    result[column] += test.cumulative_arrivals
            result['train_end'] = pd.Timestamp(max(train_days))
            result['task'], result['target'], result['fold'] = task, target, fold
            predictions.append(result)
            print(f'fold={fold} task={task} target={target} n={len(result)}', flush=True)
    return pd.concat(predictions, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flow-csv', type=Path, required=True)
    parser.add_argument('--schedule-json', type=Path, required=True)
    parser.add_argument('--weather-csv', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--test-days', type=int, default=180)
    parser.add_argument('--min-train-days', type=int, default=365)
    parser.add_argument('--max-iter', type=int, default=150)
    parser.add_argument('--duration-hours', type=float, default=3)
    parser.add_argument('--calendar-mode', choices=['basic', 'rich'], default='rich')
    parser.add_argument('--alignment', choices=['interval-end', 'interval-start'], default='interval-end')
    args = parser.parse_args()
    if min(args.folds, args.test_days, args.min_train_days, args.max_iter) <= 0:
        parser.error('folds, test-days, min-train-days and max-iter must be positive')
    flow = load_hourly_flow(args.flow_csv)
    schedule = json.loads(args.schedule_json.read_text())
    # Do not interpret dates outside retrieved seasons as verified no-game dates.
    start, end = pd.Timestamp(schedule['coverage_start']), pd.Timestamp(schedule['coverage_end'])
    flow = flow.loc[flow.day.ge(start + pd.Timedelta(days=1)) & flow.day.lt(end - pd.Timedelta(days=1))].copy()
    weather = build_weather_features(args.weather_csv) if args.weather_csv else None
    with threadpool_limits(limits=2):
        predictions = run(flow, schedule, weather=weather, n_folds=args.folds,
                          test_days=args.test_days, min_train_days=args.min_train_days,
                          max_iter=args.max_iter, duration_hours=args.duration_hours,
                          calendar_mode=args.calendar_mode, alignment=args.alignment)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / 'predictions.csv', index=False)
    score_predictions(predictions).to_csv(args.output_dir / 'metrics.csv', index=False)
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    schedule_fetched_at=schedule['fetched_at'],
                    schedule_vintage=schedule['schedule_vintage'],
                    production_changed=False, promotion_status='screening_only',
                    limitations=['Retrospective schedule revisions can leak old reschedules.',
                                 'Hourly timing convention requires sensitivity analysis.',
                                 'Day completion is a boosted ablation, not the production ensemble.',
                                 'Weather confounding remains when weather-csv is omitted.',
                                 'Multiple cutoffs per day are correlated; use day-block uncertainty.',
                                 'No causal or staffing claims; no automatic promotion.'])
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
