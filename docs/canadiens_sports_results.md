# Canadiens forecast experiment: historical results

**Recommendation: do not add Canadiens features to production.** The tested
features provide no material or consistent incremental prediction benefit.
This is a negative screening result, not evidence that hockey never affects ED
attendance or that exceptional broadcasts cannot matter.

## Data and comparison

Run date: November 7, 2026. Five closed Canadiens seasons (2021–22 through
2025–26) supplied 434 regular-season/playoff games. Historical JGH data within
schedule coverage provided 1,747 complete days for training/evaluation.
Four expanding chronological folds held out 720 complete days spanning
May 19, 2024–June 27, 2026. No production workflow or model was changed.

Each baseline/sports pair has the same rows, model settings and existing
calendar controls. Comparisons cover 17,280 hourly predictions per target and
12,240 day-completion predictions (17 cutoffs per held-out day). Models remain
fixed within each fold; recent-arrival and prior-day inputs update as those
observations become available. These are rolling-origin comparisons, not
one forecast issued at the beginning of each six-month test block.

Primary weather-adjusted screening used public Open-Meteo historical weather
for Montreal, with complete weather-variable coverage over the retrieved grid.
Weather inputs are restricted to current/trailing timestamps for completion,
and to the preceding report timestamp for one-hour-ahead predictions.
Historical weather estimates are retrospective, not exact as-issued records.

## Primary result: weather included, three-hour scheduled game window

Mean absolute error in patients; lower is better. Completion errors describe
final-day totals inferred from an hourly cutoff, not a seven-day daily outlook.

| Target | Baseline MAE | With sports | Change |
|---|---:|---:|---:|
| Hourly total arrivals | 2.397797 | 2.397410 | -0.000387 |
| Hourly non-ambulance arrivals (total minus ambulances) | 2.260170 | 2.259243 | -0.000927 |
| Hourly ambulance arrivals | 1.003416 | 1.003619 | +0.000203 |
| Day-completion total | 9.714352 | 9.716050 | +0.001698 |

All four overall paired day-bootstrap 95% intervals include zero. For total
hourly arrivals, the MAE-difference interval is approximately -0.00185 to
+0.00090 patients/hour. For day completion, it is -0.05864 to +0.05919 patients.
These intervals describe this screening design; they do not account for every
source of serial dependence, schedule revision or repeated comparison.

During scheduled game windows, hourly total MAE changes from 2.4917 to 2.4907;
afterward, from 1.9425 to 1.9461. Neither paired interval excludes zero.
Day-completion performance improves in the first fold but worsens in the next
three: mean MAE changes are -0.2016, +0.0975, +0.0483 and +0.0626 patients.

Playoff scheduled-window hourly total MAE is 2.6079 without sports versus
2.6174 with sports, over only 66 hourly samples on 22 report dates. Regular-season
scheduled-window MAE is 2.4749 versus 2.4724. These small samples/differences do
not establish a distinct playoff benefit.

## Sensitivity checks

| Run | Hourly total MAE change | Completion MAE change |
|---|---:|---:|
| No weather, interval-ending, 3h | +0.001084 | -0.064340 |
| No weather, interval-starting, 3h | -0.000162 | -0.064340 |
| Weather, interval-ending, 2.5h | -0.000868 | +0.002043 |
| Weather, interval-ending, 3h | -0.000387 | +0.001698 |
| Weather, interval-ending, 3.5h | +0.000151 | -0.000989 |

The unadjusted completion gain is about 0.66% (9.8048 to 9.7405 patients),
is worse in the latest fold, and does not persist with weather controls.
This does not prove weather causally explains the original difference. Changing
timing and duration produces near-zero differences with varying signs. The
interval-start sensitivity was run without weather; all three duration checks
included weather. No interpretation was selected based on the best score.

## Decision and limits

Keep the experiment isolated and leave production forecasts unchanged. These
results do not justify adding more routine sports calendars now. A separate
focused experiment could address widely watched Canada/Olympic games, which
were not tested here.

This boosted screening model is not the locked production ensemble and does
not validate 24-hour TBS predictions. Historical NHL schedules can contain
reschedules unknown at an old forecast time; fixed game windows omit actual
end times and overtime intentionally. Daily targets retain the existing
stored-date convention. The held-out period ends in June 2026 and does not test
the October–November 2026 inflow surge. No causal or staffing claim follows.

Seven new tests and twelve existing intraday tests pass. Source fingerprints,
parameters and aggregate fold/subgroup MAE, bias and paired intervals are in
`validation/canadiens-sports/manifest.json` and `metrics.csv`. ED source rows and
individual prediction rows are excluded from those checked-in artifacts.
