"""Offline, paired CPU comparison; never publishes operational forecasts.

Install Timestral from mistralai/timestral and chronos-forecasting separately.
Input is the historical allDataWithCalculatedColumns CSV used by ED backtests.
Only genuinely observed future hours are scored. Historical gaps are forward
filled up to six hours, without reading beyond the forecast cutoff.
"""
import argparse
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import shutil

import numpy as np
import pandas as pd

TARGETS = ['Total_TBS', 'POD_TBS', 'Vertical_TBS', 'TTStr', 'Overflow', 'WAITINGADM']
SOURCE = ['total_tbs', 'pod_tbs', 'vert_tbs', 'TTStr', 'overflow', 'WAITINGADM']
LEVELS = [0.1, 0.5, 0.9]


def seasonal_cutoffs(data, *, year, context, spacing, count):
    """Select fixed-grid, nonoverlapping windows before looking at model errors.

    A seasonal year includes autumn of year-1 and winter/spring/summer of year.
    Entire 24-hour scoring windows must lie inside their designated season.
    """
    boundaries = [
        ('autumn', f'{year-1}-09-01', f'{year-1}-12-01'),
        ('winter', f'{year-1}-12-01', f'{year}-03-01'),
        ('spring', f'{year}-03-01', f'{year}-06-01'),
        ('summer', f'{year}-06-01', f'{year}-09-01'),
    ]
    selected, audit = [], []
    for season, start, stop in boundaries:
        eligible = []
        grid = pd.date_range(pd.Timestamp(start)-pd.Timedelta(hours=1),
                             pd.Timestamp(stop)-pd.Timedelta(hours=25), freq=f'{spacing}h')
        for cutoff in grid:
            end = data.index.get_indexer([cutoff])[0]
            reason = 'eligible'
            if end < context-1 or end+24 >= len(data):
                reason = 'outside_available_history'
            else:
                history = data.iloc[end-context+1:end+1].ffill(limit=6).to_numpy()
                actual = data.iloc[end+1:end+25].to_numpy()
                if not np.isfinite(actual).all() or (actual < 0).any():
                    reason = 'missing_or_negative_actual'
                elif not np.isfinite(history).all() or (history < 0).any():
                    reason = 'unusable_history'
                else:
                    eligible.append(end)
            audit.append(dict(season=season, cutoff=str(cutoff), eligibility=reason))
        if len(eligible) < count:
            raise ValueError(f'{season}: only {len(eligible)} eligible cutoffs, requested {count}')
        selected.extend(eligible[i] for i in np.linspace(0, len(eligible)-1, count, dtype=int))
    chosen = {str(data.index[i]) for i in selected}
    for row in audit:
        row['selected'] = row['cutoff'] in chosen
    return sorted(selected), pd.DataFrame(audit)


def main():
    import torch
    from chronos import BaseChronosPipeline
    from timestral import TimestralForecaster

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--timestral', default='mistralai/Timestral-123M')
    p.add_argument('--chronos', default='amazon/chronos-2')
    p.add_argument('--cutoffs', type=int, default=28)
    p.add_argument('--context', type=int, default=672)
    p.add_argument('--spacing', type=int, default=49)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--calendar', action='store_true', help='Supply identical future-known calendar cycles')
    p.add_argument('--seasonal-year', type=int, help='Use four complete seasons ending with summer of this year')
    p.add_argument('--cutoffs-per-season', type=int, default=24)
    args = p.parse_args()
    if args.context < 1 or args.cutoffs < 1 or args.cutoffs_per_season < 1 or args.spacing < 24:
        p.error('Context and counts must be positive; spacing must be at least 24 hours')
    torch.set_num_threads(args.threads)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(args.input)
    raw['ds'] = pd.to_datetime(raw.ds, format='mixed', errors='raise')
    if raw.ds.dt.tz is not None:
        raw['ds'] = raw.ds.dt.tz_convert('America/Montreal').dt.tz_localize(None)
    raw['ds'] = raw.ds.dt.floor('h')
    raw = raw.sort_values('ds').drop_duplicates('ds', keep='last').set_index('ds')
    data = raw[SOURCE].apply(pd.to_numeric, errors='coerce')
    data.columns = TARGETS
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='h'))
    cutoffs = []
    end = len(data) - 25
    while end >= args.context - 1 and len(cutoffs) < args.cutoffs:
        hist = data.iloc[end-args.context+1:end+1].ffill(limit=6)
        truth = data.iloc[end+1:end+25]
        if np.isfinite(hist.to_numpy()).all() and np.isfinite(truth.to_numpy()).all():
            cutoffs.append(end)
        end -= args.spacing
    if not cutoffs:
        raise ValueError('No fully observed scoring windows with usable context')
    cutoffs.sort()
    if args.seasonal_year:
        cutoffs, audit = seasonal_cutoffs(data, year=args.seasonal_year, context=args.context,
                                          spacing=args.spacing, count=args.cutoffs_per_season)
        audit.to_csv(out/'cutoff_audit.csv', index=False)
    meta = dict(arguments=vars(args), input_sha256=hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                data_start=str(data.index.min()), data_end=str(data.index.max()),
                cutoffs=[str(data.index[i]) for i in cutoffs], torch=torch.__version__,
                timezone='America/Montreal', horizon=24,
                packages={name: version(name) for name in ['timestral', 'chronos-forecasting', 'transformers', 'numpy', 'pandas']},
                caveat='Exploratory baseline/calendar comparison; not a production routed-model comparison.')
    meta['local_checkpoint_sha256'] = {
        name: {f: hashlib.file_digest((Path(path)/f).open('rb'), 'sha256').hexdigest()
               for f in ['config.json', 'model.safetensors']}
        for name, path in [('timestral', args.timestral), ('chronos2', args.chronos)]
        if Path(path).is_dir()
    }
    (out/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    models = [('timestral', TimestralForecaster.from_pretrained(args.timestral, device='cpu')),
              ('chronos2', BaseChronosPipeline.from_pretrained(args.chronos, device_map='cpu'))]
    rows, timing = [], []
    for name, model in models:
        for end in cutoffs:
            history = data.iloc[end-args.context+1:end+1].ffill(limit=6)
            truth = data.iloc[end+1:end+25]
            month = truth.index[0].month
            season = ('winter' if month in [12, 1, 2] else 'spring' if month in [3, 4, 5]
                      else 'summer' if month in [6, 7, 8] else 'autumn')
            dates = history.index.append(truth.index)
            cov = pd.DataFrame(index=dates)
            if args.calendar:
                for label, phase in [('hour', dates.hour/24), ('week', dates.dayofweek/7),
                                     ('year', (dates.dayofyear-1)/365.25)]:
                    cov[label+'_sin'] = np.sin(2*np.pi*phase)
                    cov[label+'_cos'] = np.cos(2*np.pi*phase)
            start = time.perf_counter()
            if name == 'timestral':
                options = {'known_covariates': cov.to_numpy().T[None].astype('float32')} if args.calendar else {}
                result = model.forecast(history.to_numpy().T[None].astype('float32'), prediction_length=24, **options)
                quantiles = np.stack([result.quantile(q)[0] for q in LEVELS], axis=-1)
            else:
                frame = history.join(cov).rename_axis('ds').reset_index().assign(id='jgh')
                options = {'future_df': cov.iloc[-24:].rename_axis('ds').reset_index().assign(id='jgh')} if args.calendar else {}
                result = model.predict_df(frame, prediction_length=24, id_column='id',
                                          timestamp_column='ds', target=TARGETS, quantile_levels=LEVELS,
                                          context_length=args.context, **options)
                quantiles = np.stack([result[result.target_name == t].sort_values('ds')
                                      [[str(q) for q in LEVELS]].to_numpy() for t in TARGETS])
            elapsed = time.perf_counter()-start
            if quantiles.shape != (len(TARGETS), 24, 3) or not np.isfinite(quantiles).all():
                raise ValueError(f'Invalid {name} output: {quantiles.shape}')
            timing.append(dict(model=name, cutoff=str(history.index[-1]), seconds=elapsed))
            for j, target in enumerate(TARGETS):
                for h in range(24):
                    y = float(truth.iloc[h, j])
                    qs = quantiles[j, h]
                    error = y - qs
                    rows.append(dict(model=name, cutoff=str(history.index[-1]), target=target, season=season,
                                     horizon=h+1, actual=y, q10=qs[0], median=qs[1], q90=qs[2],
                                     absolute_error=abs(y-qs[1]),
                                     pinball=float(np.maximum(np.array(LEVELS)*error,
                                                              (np.array(LEVELS)-1)*error).mean()),
                                     covered80=bool(qs[0] <= y <= qs[2])))
            pd.DataFrame(rows).to_csv(out/'detail.csv', index=False)
            pd.DataFrame(timing).to_csv(out/'runtime.csv', index=False)
            print(f'{name} {history.index[-1]} {elapsed:.2f}s', flush=True)
    scores = pd.DataFrame(rows)
    for group, filename in [(['model', 'target'], 'summary.csv'),
                            (['season', 'model', 'target'], 'by_season.csv'),
                            (['model', 'target', 'horizon'], 'by_horizon.csv')]:
        scores.groupby(group).agg(mae=('absolute_error', 'mean'), pinball=('pinball', 'mean'),
                                  coverage80=('covered80', 'mean'), n=('actual', 'size')).to_csv(out/filename)
    print(pd.read_csv(out/'summary.csv').to_string(index=False), flush=True)
    if args.seasonal_year:
        # Retain the detailed predictions without large plain-text CSVs in git.
        temporary = out/'detail.csv.gz.tmp'
        with (out/'detail.csv').open('rb') as source, gzip.open(temporary, 'wb') as destination:
            shutil.copyfileobj(source, destination)
        temporary.replace(out/'detail.csv.gz')
        (out/'detail.csv').unlink()


if __name__ == '__main__':
    main()
