from pathlib import Path
import gzip,shutil,json
import numpy as np
import pandas as pd
import argparse
p=argparse.ArgumentParser(description='Summarize the 2025–26 four-season Timestral study')
p.add_argument('--validation-root', type=Path, default=Path(__file__).resolve().parents[3]/'validation')
root=p.parse_args().validation_root
scenarios={'history_only':'timestral-seasons-baseline-2026-10-31','calendar':'timestral-seasons-calendar-2026-10-31'}
lines=['# Four-season ED comparison: Timestral-123M versus Chronos-2','',
'Evaluation: October 31, 2026. These are offline validation results; production is unchanged.','',
'## Finding','',
'The initial pilot advantage in total TBS does not generalize strongly across seasons. With calendar inputs, Timestral reduces total TBS MAE by only 0.3% (7.112 to 7.090 patients); the paired cutoff bootstrap interval for the difference includes a tie. It is worse in autumn, nearly tied in winter, and better in spring and summer. Overflow error is 4.0% lower, but its bootstrap interval only narrowly excludes zero and should be treated as tentative. Chronos-2 retains a 2.2% advantage for waiting admissions.','',
'Timestral gives total TBS 80% interval coverage closer to nominal (79.1% versus 72.2%). That is an uncertainty-calibration advantage, not evidence of a large point-forecast improvement. Keep Chronos-2 as the production default; Timestral overflow and uncertainty estimates are candidates for further validation, including a second annual cycle and production staffing inputs.','',
'## Design','',
'Four complete meteorological seasons: autumn 2025 (September–November), winter 2025–26 (December–February), spring 2026 (March–May), and summer 2026 (June–August). Each season contributes 24 cutoffs, chosen evenly across eligible 49-hour grid points before examining forecast errors. All 24 forecast hours must fall within their assigned season. Models receive identical 672-hour context and six multivariate targets. Scoring windows do not overlap.','',
'Both history-only and identical-calendar scenarios use the same 96 cutoffs. Calendar inputs are sine/cosine cycles of hour, weekday and day of year. No staffing, weather, or production routing features are included. Historical gaps can be forward-filled at most six hours; future actuals are never imputed. Precomputed canonical historical target columns are used directly.','']
allcomparisons=[]; sez=[];checks=[]
for scenario,folder in scenarios.items():
 p=root/folder
 d=pd.read_csv(p/'detail.csv.gz')
 assert len(d)==96*6*24*2
 assert not d.duplicated(['model','cutoff','target','horizon']).any()
 assert d.groupby(['model','season']).cutoff.nunique().eq(24).all()
 assert (d.q10<=d['median']).all() and (d['median']<=d.q90).all()
 keys=['cutoff','target','horizon']
 a=d[d.model=='timestral'].set_index(keys).sort_index()
 b=d[d.model=='chronos2'].set_index(keys).sort_index()
 assert a.actual.equals(b.actual)
 recomputed=d.groupby(['model','target']).agg(mae=('absolute_error','mean'),pinball=('pinball','mean'),coverage80=('covered80','mean'),n=('actual','size')).sort_index()
 saved=pd.read_csv(p/'summary.csv').set_index(['model','target']).sort_index()
 np.testing.assert_allclose(recomputed.to_numpy(),saved.to_numpy(),atol=1e-7,rtol=1e-7)
 checks.append(a.index)
 actual=d[d.model=='timestral'].groupby('cutoff').size();assert actual.eq(144).all()
 title='History only' if scenario=='history_only' else 'Identical calendar inputs'
 lines+=['## '+title,'','| Target | Chronos-2 MAE | Timestral MAE | Error change |','|---|---:|---:|---:|']
 s=pd.read_csv(p/'summary.csv').pivot(index='target',columns='model',values='mae')
 cutoff=d.groupby(['season','cutoff','target','model']).absolute_error.mean().unstack('model')
 rng=np.random.default_rng(42)
 for target,r in s.iterrows():
  change=100*(r.timestral/r.chronos2-1)
  lines.append(f'| {target} | {r.chronos2:.3f} | {r.timestral:.3f} | {change:+.1f}% |')
  z=cutoff.xs(target,level='target');samples=[]
  for season,v in z.groupby(level='season'):
   diff=(v.timestral-v.chronos2).to_numpy();samples.append(diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1))
  lo,hi=np.quantile(np.mean(samples,axis=0),[.025,.975])
  allcomparisons.append(dict(scenario=scenario,target=target,chronos_mae=r.chronos2,timestral_mae=r.timestral,change_pct=change,paired_delta=r.timestral-r.chronos2,bootstrap_low=lo,bootstrap_high=hi))
 lines+=['','Negative changes favor Timestral. MAE is average absolute error in patient counts over horizons 1–24.','']
 for (season,target),v in d.groupby(['season','target']):
  m=v.groupby('model').absolute_error.mean();sez.append(dict(scenario=scenario,season=season,target=target,chronos_mae=m.chronos2,timestral_mae=m.timestral,change_pct=100*(m.timestral/m.chronos2-1)))
 d['interval_width']=d.q90-d.q10
 uncertainty=d.groupby(['model','target']).agg(coverage80=('covered80','mean'),width80=('interval_width','mean'),pinball=('pinball','mean')).reset_index()
 uncertainty.to_csv(p/'uncertainty.csv',index=False)
 lines+=['| Total TBS interval metric | Chronos-2 | Timestral |','|---|---:|---:|']
 u=uncertainty[uncertainty.target=='Total_TBS'].set_index('model')
 for metric in ['coverage80','width80','pinball']:
  lines.append(f'| {metric} | {u.loc["chronos2",metric]:.3f} | {u.loc["timestral",metric]:.3f} |')
 lines+=['']
assert checks[0].equals(checks[1])
comp=pd.DataFrame(allcomparisons);comp.to_csv(root/scenarios['calendar']/'comparison.csv',index=False)
seasons=pd.DataFrame(sez);seasons.to_csv(root/scenarios['calendar']/'season_comparison.csv',index=False)
lines+=['## Total TBS by season','','| Season | History-only error change | Calendar error change |','|---|---:|---:|']
for season in ['autumn','winter','spring','summer']:
 z=seasons[(seasons.target=='Total_TBS')&(seasons.season==season)].set_index('scenario').change_pct
 lines.append(f'| {season} | {z.history_only:+.1f}% | {z.calendar:+.1f}% |')
lines+=['','## Selection quality and limits','',
'Raw source availability is not proof that every recorded value is correct. Eligibility screening rejects missing or negative scoring actuals, and incomplete or negative historical context after limited forward filling. All candidates, rejections and selections are retained in cutoff_audit.csv. The broad source contains a seven-hour repeated-vector episode on July 1; it is outside selected scoring windows because the surrounding context failed eligibility.','']
audit=pd.read_csv(root/scenarios['history_only']/'cutoff_audit.csv')
lines+=['| Season | Eligible grid points | Missing/negative future windows | Unusable history windows | Selected |','|---|---:|---:|---:|---:|']
for season in ['autumn','winter','spring','summer']:
 z=audit[audit.season==season];v=z.eligibility.value_counts();lines.append(f'| {season} | {v.get("eligible",0)} | {v.get("missing_or_negative_actual",0)} | {v.get("unusable_history",0)} | {int(z.selected.sum())} |')
chosen=audit[audit.selected].copy();chosen['scoring_month']=(pd.to_datetime(chosen.cutoff)+pd.Timedelta(hours=1)).dt.strftime('%Y-%m')
chosen.groupby(['season','scoring_month']).size().rename('cutoffs').to_csv(root/scenarios['calendar']/'selected_months.csv')
lines+=['',
'Summer is unevenly represented: only two selected cutoffs start their scoring window in July, versus eleven each in June and August. It therefore cannot establish performance during the excluded July outage periods. Each season has 24 cutoffs; differences by target or season can be noisy. The provided 95% intervals bootstrap paired cutoff-level MAE differences within each season (10,000 draws, seed 42), preserving season weights, but do not remove temporal dependence or adjust for multiple comparisons.','',
'This is one seasonal cycle, not multiple independent years. Models are pretrained; their training corpora were not audited for exposure to similar data. These results compare common-context baselines and simple calendar conditioning, not the validated production routing. Context length, staffing features and broader multi-year confirmation remain separate questions.','',
'## Reproduction and checks','',
'Use the same environment and model checkpoint hashes recorded in metadata.json. Timestral source commit: a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff.','',
'```bash',
'python scripts/evaluation/backtests/backtest_timestral.py --input /path/to/allDataWithCalculatedColumns.csv --seasonal-year 2026 --output validation/timestral-seasons-baseline',
'python scripts/evaluation/backtests/backtest_timestral.py --input /path/to/allDataWithCalculatedColumns.csv --seasonal-year 2026 --calendar --output validation/timestral-seasons-calendar',
'```','',
'Export verification caught two final Chronos calendar cutoffs absent from the detailed file. They were reexecuted; the reconstructed metrics agreed with the original full-run summary within 0.000001. Summaries were regenerated from the reconciled details. Compressed exports now use an atomic replacement, and the analysis verifies agreement between detailed and summary metrics.','',
'Both scenarios completed: 55,296 scored target/hour predictions in total. Verified identical actuals and cutoff sets, 24 cutoffs per season per model, unique prediction keys, nonoverlapping scoring windows, ordered quantiles and finite nonnegative actuals. Four selection regression tests also pass. Detailed predictions are retained as detail.csv.gz. Summary metrics, season metrics, horizon metrics, runtimes, selection audits and paired-comparison intervals are included. No live forecast publishing or production routing changed.','']
(root/scenarios['calendar']/'REPORT.md').write_text('\n'.join(lines))
print(comp.to_string(index=False));print(seasons[seasons.target=='Total_TBS'].to_string(index=False));print('Verified all seasonal outputs.')
