# Canadiens schedule experiment

Status: historical screening complete; **no production promotion recommended**.
See [results](canadiens_sports_results.md) and the aggregate metrics under
`validation/canadiens-sports/`.
Production workflows, routing, publishing and feature flags are unchanged.

## Question and design

Does scheduled Canadiens game timing improve JGH arrival predictions beyond
recent arrivals, calendar context, and (when supplied) cutoff-safe weather?
Start with the Canadiens; do not add other sports until this ablation is useful.

Two paired comparisons use expanding chronological folds, with identical
training/test rows and model parameters in each baseline/sports pair:

- Rolling one-hour arrivals: total, ambulance, and **total minus ambulances**.
  This is a one-hour-ahead screening task, not validation of a 24-hour forecast.
- Day completion: final stored-calendar arrivals from hourly cutoffs 06–22.
  The baseline reuses the existing progress/calendar/state/weather feature sets
  and a completion curve fitted on training days only. It is a boosted ablation,
  **not a reproduction of the locked production ensemble**.

The default four 180-complete-day test blocks require at least 365 preceding
training days. Training expands across folds. Complete days follow the existing
loader's rules; negative/nonfinite arrival days are additionally excluded.
Targets are not imputed, and strict hourly lag joins preserve gaps.
Calendar controls include rich JGH holiday and school context by default.

## Public schedule and timing features

`data/sports/canadiens_2021_2025.json` contains five closed seasons (2021–22
through 2025–26), downloaded from the NHL club season schedule endpoint. It
retains IDs, season, scheduled UTC start, home/away and playoff indicators;
it discards scores, results, actual game end and overtime. Preseason is excluded.
The JSON records retrieval time, sources, season coverage and schedule vintage.
No-game dates outside retrieved coverage must not be inferred.

Features describe local game day, home/playoffs, two hours before the start,
a fixed three-hour scheduled game window, three hours afterward, and scheduled
game hours remaining before midnight. Home and playoff interactions are included.
Late games and after-game windows cross midnight in Montreal local time.
A scheduled end is a proxy; **no post-game surge is assumed**.

Historical NHL responses are the latest retrospective schedule, not archived
forecast-time vintages. Reschedules can therefore create optimistic results.
This supports screening only. Before promotion, audit changed schedules and
archive schedules prospectively, including publication/observation times.

The hourly feed's interval convention is unresolved (see
`intraday_prospective_audit.md`). Default interval-ending exposure uses ds minus
30 minutes as the hourly midpoint. Repeat with interval-starting exposure as a
sensitivity check. Repeat game duration at 2.5 and 3.5 hours as well. Day-completion
features are evaluated at the actual cutoff; its stored-date target is unchanged.

## Run

Use a locally available, authorized historical ED CSV and optional weather CSV.
The backtest **never fetches ED data**. Normal repository dependencies are needed,
including pandas, numpy, scikit-learn, threadpoolctl and holidays for rich calendar.

```bash
python scripts/canadiens_features.py --first-season 2021 --last-season 2025 \
  --output data/sports/canadiens_2021_2025.json

python scripts/backtest_canadiens_features.py \
  --flow-csv /path/to/authorized/historical-flow.csv \
  --schedule-json data/sports/canadiens_2021_2025.json \
  --weather-csv /path/to/historical-weather.csv \
  --output-dir validation/canadiens/end-3h
```

Repeat in separate output directories with `--alignment interval-start`,
`--duration-hours 2.5` and `--duration-hours 3.5`. Do not select a timing
interpretation solely because it scores better: verify the upstream convention.
Without weather, report residual weather confounding. `--calendar-mode basic`
is available for smoke tests but is not the preferred scientific comparison.

Outputs: paired predictions with training cutoff, fold/subgroup MAE and bias,
paired MAE difference with a day-block bootstrap interval, and run metadata.
Negative MAE difference favors sports. Subgroups include all hours, game days,
scheduled game window, after-game window and non-game days. These groups overlap.
Bootstrap days together to avoid treating multiple hourly cutoffs as independent;
serial dependence across days and multiple comparisons still limit inference.

## Acceptance and next validation

Require consistent improvement on held-out game windows across seasons, without
material overall/non-game degradation. Inspect the size in patients/hour, MAE,
bias, subgroup sample sizes and bootstrap intervals; do not promote from an
isolated favorable fold. Check regular season versus playoffs separately before
claiming a playoff effect: playoff sample sizes can be small. Inspect post-game
and late-night errors, recent inflow drift, and missing-data/outage sensitivity.

If screening is positive, freeze features/parameters and test an untouched later
period, with forecast-time schedule vintages. Then test the actual production
ensemble, 24-hour horizons, prediction intervals and operational TBS outcomes.
Closed seasons here do not evaluate the October–November 2026 inflow surge.
A result from this screening runner alone cannot justify production activation
or staffing changes.

## Validation so far

Seven automated tests verify timing boundaries, midnight clipping, timezone/DST
handling, schedule coverage, invariance to game results, strict missing-hour lag
joins, and a synthetic end-to-end run with chronological folds and all targets.
Synthetic data exercise the implementation; they provide **no evidence of JGH
forecast improvement**.

With explicit ED-feed authorization, five historical comparisons were completed
on November 7, 2026. Weather-adjusted errors were effectively unchanged, and
timing/duration sensitivity checks showed no consistent advantage. See the
[historical results](canadiens_sports_results.md).
