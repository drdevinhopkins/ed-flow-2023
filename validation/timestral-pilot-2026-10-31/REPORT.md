# Timestral-123M versus Chronos-2: offline ED pilot

Evaluation date: October 31, 2026. Production is unchanged.

## Finding

Timestral-123M is a promising candidate for further validation. It reduced mean absolute error on five of six targets in a history-only comparison. With identical calendar inputs, its advantage narrowed: total TBS remained 5.2% better and overflow 7.4% better, while stretcher occupancy was effectively tied. Chronos-2 remained better for waiting admissions. This does not establish superiority over the current production forecast, which routes validated feature families by target and horizon.

## Method

- Same six targets, multivariate inputs, 672-hour (28-day) context, and 24-hour horizon for both models.
- 28 paired forecast cutoffs, September 5 through October 30, 2026; scored observations extend through October 31. Cutoffs are spaced 49 hours apart to rotate the time of day and avoid overlapping scoring windows.
- 672 observed forecast hours per target and model in each scenario. No future actual values were imputed. Short gaps in context were forward-filled for at most six hours; windows with remaining missing context were ineligible.
- Precomputed historical CSV columns `total_tbs`, `pod_tbs`, `vert_tbs`, `TTStr`, `overflow`, and `WAITINGADM` were used directly. Source timestamps are treated as Montreal local time.
- History-only and calendar scenarios. The calendar scenario supplies the same sine/cosine encodings of hour, weekday, and day of year to both models, including known future values. No weather or staffing inputs were used.
- CPU float32, four PyTorch threads. Models loaded locally after downloading public checkpoints. Raw quantiles were scored without clipping or rounding. Three-quantile pinball loss is reported; it is not a full CRPS estimate.

## History-only results

MAE is the average absolute error in patient counts across horizons 1–24. Negative percentage changes favor Timestral.

| Measure | Chronos-2 MAE | Timestral-123M MAE | Timestral error change |
|---|---:|---:|---:|
| Total TBS | 7.23 | 6.77 | -6.4% |
| POD TBS | 2.48 | 2.33 | -6.2% |
| Vertical TBS | 6.41 | 6.13 | -4.3% |
| Stretcher occupancy | 10.13 | 9.46 | -6.6% |
| Overflow | 4.29 | 3.88 | -9.7% |
| Waiting for admission | 4.77 | 4.97 | +4.2% |

## Identical calendar inputs

| Measure | Chronos-2 MAE | Timestral-123M MAE | Timestral error change |
|---|---:|---:|---:|
| Total TBS | 7.03 | 6.67 | -5.2% |
| POD TBS | 2.41 | 2.36 | -2.1% |
| Vertical TBS | 6.23 | 6.04 | -2.9% |
| Stretcher occupancy | 9.59 | 9.57 | -0.2% |
| Overflow | 4.23 | 3.91 | -7.4% |
| Waiting for admission | 4.72 | 4.87 | +3.3% |

## Uncertainty and runtime

### History only

| Model | Total TBS 80% interval coverage | Total TBS pinball loss | Median seconds per six-target forecast |
|---|---:|---:|---:|
| chronos2 | 78.3% | 2.302 | 0.389 |
| timestral | 84.1% | 2.167 | 0.459 |

### Calendar

| Model | Total TBS 80% interval coverage | Total TBS pinball loss | Median seconds per six-target forecast |
|---|---:|---:|---:|
| chronos2 | 77.1% | 2.253 | 0.720 |
| timestral | 84.1% | 2.145 | 1.510 |

Target interval coverage is 80%; coverage should be interpreted alongside interval sharpness and pinball loss. Runtime excludes downloads and model loading, and is specific to this shared CPU environment.

## Limits and next step

This is an exploratory 28-cutoff sample over roughly eight weeks, using only four weeks of history. It does not test all seasons, the production context length, staffing routing, or operational activation recommendations. Availability checks do not certify that every recorded source value is free of reporting artifacts. Cutoff-level comparisons are not a definitive significance test.

Next: expand to a year or more of seasonal cutoffs, test longer contexts and the production staffing/calendar feature routes, and retain Chronos-2 as the comparator and likely waiting-admission candidate. Clarify hospital eligibility under Mistral's Modified MIT employer-revenue condition before organizational deployment. No production switch is justified by this pilot alone.

## Reproduce

Use a separate Python environment with a suitable CPU or GPU PyTorch wheel. The run used Python 3.12, Timestral source commit `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`, chronos-forecasting 2.3.2, and transformers 5.22.1. Model file and source CSV SHA-256 hashes, exact cutoffs, and package versions are in each metadata.json.

```bash
pip install git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff chronos-forecasting==2.3.2 transformers==5.22.1 pandas
python scripts/evaluation/backtests/backtest_timestral.py --input /path/to/allDataWithCalculatedColumns.csv --output validation/timestral-baseline
python scripts/evaluation/backtests/backtest_timestral.py --input /path/to/allDataWithCalculatedColumns.csv --calendar --output validation/timestral-calendar
```

The model arguments accept local directories for offline operation. Download model files only under applicable licence terms. Each result folder includes summary.csv, by_horizon.csv, detail.csv, runtime.csv, and metadata.json.
