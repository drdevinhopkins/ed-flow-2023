# Prophet inputs to Chronos-2: daily arrivals experiment

This additive experiment tests whether Prophet contributes signal beyond native
Chronos-2 and its calendar/closure covariates. It does not change production.

## Comparisons

All scenarios receive identical Chronos target history and score identical dates:

| Scenario | Inputs or combination |
|---|---|
| baseline | Chronos with target history only |
| calendar_closure | Chronos with the existing calendar/closure features |
| prophet_components | Calendar control plus Prophet trend, weekly, yearly and calendar effects |
| prophet_yhat | Calendar control plus Prophet point forecast |
| prophet_only | Standalone Prophet |
| blend_25pct_prophet | 75% calendar-control Chronos plus 25% Prophet |

The blend weight is fixed before scoring. No weights or feature choices are fitted to
the evaluation outcomes. This is a comparison against calendar controls, **not the full
weather-enabled production model**. Observed future weather must not be substituted
for archived forecasts in a subsequent production comparison.

## Availability and leakage

- Targets use the shared exact-hour coverage policy. Missing or unverified days are
  never imputed. Explicit missing dates are restored before window selection.
- Chronos history stops at the latest missing day. The context cap is 365 days.
- Prophet uses up to three calendar years of verified observations, omitting missing
  targets. It may use observations on both sides of older gaps.
- Each historical Prophet input is a genuine D+1 through D+7 forecast from a model
  fitted strictly through a preceding weekly origin. It is never an in-sample fitted
  value from the final Prophet model. Future features use the evaluation cutoff.
- Prophet requires 365 verified training days; yearly seasonality is enabled only
  when the training observations span at least 730 days. Weekly seasonality is always
  enabled. The four calendar regressors and regularization are fixed in the script.
- Every scenario shares the resulting trimmed target context. This avoids a history
  length advantage for the control or the augmented scenarios.
- Selection uses dates, coverage and season, never observed target magnitude or
  forecast errors. Scoring horizons do not overlap within a pass. Cutoffs are spread
  within each available season; a shortfall from the requested count is reported.

The strict pass requires 120 uninterrupted context days. The supplementary pass uses
the production minimum of 28 days to cover seasons unavailable under the strict rule.
The supplementary results include a context-length breakdown and should be interpreted
separately. The passes can share dates and are not independent replications.

## Run

```bash
pip install -r chronos-requirements.txt 'prophet>=1.2,<2'
python tests/test_prophet_chronos_daily.py
PYTHONPATH=scripts python scripts/evaluation/backtests/backtest_prophet_chronos_daily.py \
  --num-cutoffs 24 --output-dir validation-output-prophet/strict
PYTHONPATH=scripts python scripts/evaluation/backtests/backtest_prophet_chronos_daily.py \
  --daily-csv validation-output-prophet/strict/verified_daily.csv \
  --min-history-days 28 --num-cutoffs 24 \
  --output-dir validation-output-prophet/all-seasons
```

Use `--prepare-only` to inspect coverage before loading a model. `--daily-csv` accepts
only a previously verified `ds,daily_visits` snapshot; it cannot verify arbitrary daily
totals against hourly coverage. For another Chronos context cap, use `--context-days`;
the existing production cap is 1095, although gaps often shorten actual history.

The GitHub workflow `prophet-chronos-backtest.yml` is manual-only. Starting it explicitly runs both passes and stores ED-derived
artifacts in GitHub Actions; review the repository access policy before starting it.
There is no automatic run on push or pull request.

Outputs include paired predictions, MAE/RMSE/WAPE/bias overall and by season, horizon,
holiday status, context group and cutoff; exact past/future Prophet feature snapshots;
training-vintage audits; the verified daily input; and a target fingerprint, package
versions and model revision. Positive improvement percentages mean lower MAE.

Promotion requires consistent gains across seasons and cutoffs, followed by validation
alongside archived forecast-time weather. Small aggregate changes on these selected
windows are exploratory evidence, not a demonstrated production improvement.
