# Four-season ED comparison: Timestral-123M versus Chronos-2

Evaluation: October 31, 2026. These are offline validation results; production is unchanged.

## Finding

The initial pilot advantage in total TBS does not generalize strongly across seasons. With calendar inputs, Timestral reduces total TBS MAE by only 0.3% (7.112 to 7.090 patients); the paired cutoff bootstrap interval for the difference includes a tie. It is worse in autumn, nearly tied in winter, and better in spring and summer. Overflow error is 4.0% lower, but its bootstrap interval only narrowly excludes zero and should be treated as tentative. Chronos-2 retains a 2.2% advantage for waiting admissions.

Timestral gives total TBS 80% interval coverage closer to nominal (79.1% versus 72.2%). That is an uncertainty-calibration advantage, not evidence of a large point-forecast improvement. Keep Chronos-2 as the production default; Timestral overflow and uncertainty estimates are candidates for further validation, including a second annual cycle and production staffing inputs.

## Design

Four complete meteorological seasons: autumn 2025 (September–November), winter 2025–26 (December–February), spring 2026 (March–May), and summer 2026 (June–August). Each season contributes 24 cutoffs, chosen evenly across eligible 49-hour grid points before examining forecast errors. All 24 forecast hours must fall within their assigned season. Models receive identical 672-hour context and six multivariate targets. Scoring windows do not overlap.

Both history-only and identical-calendar scenarios use the same 96 cutoffs. Calendar inputs are sine/cosine cycles of hour, weekday and day of year. No staffing, weather, or production routing features are included. Historical gaps can be forward-filled at most six hours; future actuals are never imputed. Precomputed canonical historical target columns are used directly.

## History only

| Target | Chronos-2 MAE | Timestral MAE | Error change |
|---|---:|---:|---:|
| Overflow | 3.630 | 3.429 | -5.5% |
| POD_TBS | 2.258 | 2.200 | -2.6% |
| TTStr | 8.560 | 8.505 | -0.7% |
| Total_TBS | 7.194 | 7.108 | -1.2% |
| Vertical_TBS | 6.280 | 6.210 | -1.1% |
| WAITINGADM | 4.238 | 4.200 | -0.9% |

Negative changes favor Timestral. MAE is average absolute error in patient counts over horizons 1–24.

| Total TBS interval metric | Chronos-2 | Timestral |
|---|---:|---:|
| coverage80 | 0.743 | 0.791 |
| width80 | 20.507 | 22.429 |
| pinball | 2.236 | 2.200 |

## Identical calendar inputs

| Target | Chronos-2 MAE | Timestral MAE | Error change |
|---|---:|---:|---:|
| Overflow | 3.536 | 3.395 | -4.0% |
| POD_TBS | 2.217 | 2.196 | -1.0% |
| TTStr | 8.335 | 8.412 | +0.9% |
| Total_TBS | 7.112 | 7.090 | -0.3% |
| Vertical_TBS | 6.260 | 6.195 | -1.0% |
| WAITINGADM | 4.045 | 4.134 | +2.2% |

Negative changes favor Timestral. MAE is average absolute error in patient counts over horizons 1–24.

| Total TBS interval metric | Chronos-2 | Timestral |
|---|---:|---:|
| coverage80 | 0.722 | 0.791 |
| width80 | 19.464 | 22.329 |
| pinball | 2.248 | 2.201 |

## Total TBS by season

| Season | History-only error change | Calendar error change |
|---|---:|---:|
| autumn | +1.1% | +3.1% |
| winter | -0.3% | +0.4% |
| spring | -2.8% | -3.4% |
| summer | -2.8% | -1.3% |

## Selection quality and limits

Raw source availability is not proof that every recorded value is correct. Eligibility screening rejects missing or negative scoring actuals, and incomplete or negative historical context after limited forward filling. All candidates, rejections and selections are retained in cutoff_audit.csv. The broad source contains a seven-hour repeated-vector episode on July 1; it is outside selected scoring windows because the surrounding context failed eligibility.

| Season | Eligible grid points | Missing/negative future windows | Unusable history windows | Selected |
|---|---:|---:|---:|---:|
| autumn | 43 | 2 | 0 | 24 |
| winter | 39 | 5 | 0 | 24 |
| spring | 35 | 10 | 0 | 24 |
| summer | 29 | 2 | 14 | 24 |

Summer is unevenly represented: only two selected cutoffs start their scoring window in July, versus eleven each in June and August. It therefore cannot establish performance during the excluded July outage periods. Each season has 24 cutoffs; differences by target or season can be noisy. The provided 95% intervals bootstrap paired cutoff-level MAE differences within each season (10,000 draws, seed 42), preserving season weights, but do not remove temporal dependence or adjust for multiple comparisons.

This is one seasonal cycle, not multiple independent years. Models are pretrained; their training corpora were not audited for exposure to similar data. These results compare common-context baselines and simple calendar conditioning, not the validated production routing. Context length, staffing features and broader multi-year confirmation remain separate questions.

## Reproduction and checks

Use the same environment and model checkpoint hashes recorded in metadata.json. Timestral source commit: a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff.

```bash
python scripts/evaluation/backtests/backtest_timestral.py --input /path/to/allDataWithCalculatedColumns.csv --seasonal-year 2026 --output validation/timestral-seasons-baseline
python scripts/evaluation/backtests/backtest_timestral.py --input /path/to/allDataWithCalculatedColumns.csv --seasonal-year 2026 --calendar --output validation/timestral-seasons-calendar
```

Export verification caught two final Chronos calendar cutoffs absent from the detailed file. They were reexecuted; the reconstructed metrics agreed with the original full-run summary within 0.000001. Summaries were regenerated from the reconciled details. Compressed exports now use an atomic replacement, and the analysis verifies agreement between detailed and summary metrics.

Both scenarios completed: 55,296 scored target/hour predictions in total. Verified identical actuals and cutoff sets, 24 cutoffs per season per model, unique prediction keys, nonoverlapping scoring windows, ordered quantiles and finite nonnegative actuals. Four selection regression tests also pass. Detailed predictions are retained as detail.csv.gz. Summary metrics, season metrics, horizon metrics, runtimes, selection audits and paired-comparison intervals are included. No live forecast publishing or production routing changed.
