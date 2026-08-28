# Intraday day-completion forecast

## Operational question

At each hourly report cutoff, estimate the final number of ED arrivals in the current
`America/Montreal` calendar day. The model predicts remaining arrivals:

```text
predicted final total = observed arrivals so far + max(0, predicted remaining arrivals)
```

This is a separate validation workflow. It does not modify the daily Chronos forecast,
the hourly flow forecast, timers, publishing, Dropbox destinations, or production feature
routing.

## Training snapshots

The backtest creates one row per complete calendar day and cutoff hour. A day is complete
only when its observed local-hour labels match the expected Montreal day, including
23-hour spring DST days and 25-hour autumn DST days. Source cumulative-arrival counters
are not trusted because their reset conventions have varied; cumulative arrivals are
reconstructed from hourly `Inflow_Total`.

Snapshot feature families are:

- **Progress:** observed cumulative arrivals, recent 1/2/3/6/12-hour pace, acceleration,
  arrival-mode mix, hours remaining, and a trailing pre-day volume prior.
- **Calendar:** weekday and seasonality, with optional reuse of the repository's rich JGH
  holiday, closure, long-weekend, construction-vacation, and school-calendar features.
- **ED state:** the contemporaneous operational metrics plus short changes. `Total_TBS`
  uses the canonical sum of triage hallway TBS, the three POD TBS fields, RAZ TBS,
  ambulatory Vertical TBS, QTrack TBS, and Garage TBS.
- **Weather:** optional current and trailing observed weather. The merge is backward-only,
  so no observation after the forecast cutoff can enter a row.

Archived forecast-time weather snapshots will be required before testing remaining-day
forecast weather. Realized future weather must not be substituted in retrospective tests.

## Comparators

1. Historical completion curve by cutoff hour.
2. Pre-day trailing-volume prior updated by the day's pace residual.
3. Pooled quantile gradient boosting using progress features.
4. Progress plus calendar.
5. Progress plus ED state.
6. Full available feature set, with a calendar-plus-weather ablation when weather exists.

The boosted models predict P10 and P90 remaining-arrival bounds plus an expected
remaining-arrival point forecast. The point model uses squared-error loss to target the
expected final count and mean-bias gate; the bounds use quantile loss. Each outer backtest
fold also reserves the most recent 56 training days as an inner calibration window.
Models fit on the earlier inner block generate genuine forward residuals for that
window. Hour-specific tail residual quantiles and the mean point residual are then
shrunk toward their pooled corrections. The mean point correction directly targets the
release bias gate; the tail corrections target interval coverage.
The corrected variants have a `_calibrated` suffix. The outer test block remains unseen
by model fitting, feature-curve fitting, and calibration.

All predictions are clipped at zero remaining before being added to the observed count.
Quantiles are sorted after correction, so the interval cannot cross and the final point
forecast cannot be lower than arrivals already observed.

## Validation

Validation uses expanding, time-ordered folds. Outputs include:

- `predictions.csv`: every out-of-sample forecast;
- `summary.csv`: MAE, RMSE, bias, nominal P80 coverage, and interval width overall and by
  cutoff hour;
- `best_by_hour.csv`: lowest-MAE candidate at each cutoff hour;
- `feature_sets.csv`: exact feature membership for each boosted scenario; and
- `run_config.json`: input and backtest configuration; and
- `readiness.json`: deterministic gate results for the fixed
  `boosted_progress_calibrated` candidate versus `prior_update`.

Model selection should be made by cutoff hour. A simple completion curve may remain best
very late in the day even if the pooled boosted model is superior earlier.

## Production-readiness gates

The experiment remains shadow-only until all gates pass. The initial retrospective gates
are deliberately operational rather than based on a single headline metric:

- at least 5% lower MAE than `prior_update`, both overall and across the 11:00–18:00
  operational window;
- absolute bias no greater than 2 patients overall and 3 patients at every operational
  cutoff hour;
- empirical coverage for the nominal P80 interval between 75% and 85%, overall and in
  the operational window;
- every forecast is at least the observed count, intervals never cross, and stale or
  incomplete input suppresses output rather than fabricating a forecast; and
- results hold in a prospective shadow run for at least 28 complete days (56 preferred),
  with versioned model/calibration artifacts and no unresolved data-quality failures.

Passing a retrospective gate does not authorize production publishing. Production also
requires a deterministic fallback, freshness monitoring, a scoring/runbook path, and an
explicit go/no-go review after prospective evidence is available.

## Example server run

```bash
python scripts/evaluation/backtests/backtest_intraday_day_completion.py \
  --calendar-context rich \
  --cutoff-hours 6-22 \
  --n-folds 6 \
  --test-days 28 \
  --min-train-days 365 \
  --calibration-days 56 \
  --calibration-shrinkage-days 28 \
  --output-dir validation/intraday-day-completion
```

Add `--weather-csv PATH` only when the file contains hourly weather observations whose
timestamps represent information available by the corresponding cutoff.
