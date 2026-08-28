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

The boosted models predict P10, P50, and P90 remaining arrivals. All predictions are
clipped at zero remaining before being added to the observed count.

## Validation

Validation uses expanding, time-ordered folds. Outputs include:

- `predictions.csv`: every out-of-sample forecast;
- `summary.csv`: MAE, RMSE, bias, nominal P80 coverage, and interval width overall and by
  cutoff hour;
- `best_by_hour.csv`: lowest-MAE candidate at each cutoff hour;
- `feature_sets.csv`: exact feature membership for each boosted scenario; and
- `run_config.json`: input and backtest configuration.

Model selection should be made by cutoff hour. A simple completion curve may remain best
very late in the day even if the pooled boosted model is superior earlier.

## Example server run

```bash
python scripts/evaluation/backtests/backtest_intraday_day_completion.py \
  --calendar-context rich \
  --cutoff-hours 6-22 \
  --n-folds 6 \
  --test-days 28 \
  --min-train-days 365 \
  --output-dir validation/intraday-day-completion
```

Add `--weather-csv PATH` only when the file contains hourly weather observations whose
timestamps represent information available by the corresponding cutoff.
