# Hourly history and daily-target audit

This read-only diagnostic compares the hourly history with its published daily
totals and, optionally, the prospective scoring detail. It does not download,
publish, overwrite or impute operational data. Track corrective work in [#65](https://github.com/drdevinhopkins/ed-flow-2023/issues/65).

## Repeat the audit

Download a consistent set of current aggregate files to a separate working
directory, then run from the repository root:

```bash
python scripts/evaluation/audit_flow_history.py \
  --hourly /path/to/allData.csv \
  --daily /path/to/daily_inflow.csv \
  --scored-detail /path/to/daily_visits_prospective_detail.csv \
  --output-dir /path/to/audit-results
```

Dependencies: pandas and NumPy; no GPU, model, Dropbox credentials or network
access is needed. `--scored-detail` is optional. `--now` accepts an aware ISO
timestamp for reproducing the original audit time. Outputs are `summary.json`,
`daily_quality.csv`, `missing_hours.csv` and, when supplied, `scoring_quality.csv`.
The summary includes SHA-256 input hashes.

## Semantics

- Keep the existing calendar grouping by the stored local `ds` date, including
  00:00. This audit does not decide whether an upstream interval-end timestamp
  belongs to the prior calendar day; that requires source verification.
- A day needs every clock-hour slot, unique timestamps and finite nonnegative
  arrivals. Twenty-three rows on an ordinary day are incomplete. Duplicate rows
  cannot cover a missing slot, and a completely absent internal day is visible.
- Mark first-day partial coverage, the current calendar day, and DST transition
  dates separately. Naive wall-clock timestamps cannot verify the elapsed-hour
  convention or repeated autumn hour. DST flags are review flags, not imputed
  patient counts or proof of a missing clinical observation.
- `audit_eligible` requires complete ordinary-day hourly coverage and a matching
  published daily sum. This is conservative diagnostic eligibility, not a new
  production forecast route. A matching partial sum remains unverified.
- For scoring diagnostics, trace each actual to its daily source and mirror the
  current scorer's eight most recent same-weekday baseline dates **at or before
  the original cutoff**, with the existing 28-day fallback. Flag unverified
  dates and scored actuals differing from the downloaded daily totals. Do not
  revise issued predictions, calculate corrected accuracy, or change readiness.

## October 7, 2026 findings

The checked-in diagnostics in `validation/history-quality-2026-10-07/` use these
published files, inspected independently on October 7:

| Source | Operational coverage | Dropbox upload, Montreal |
| --- | --- | --- |
| `allData.csv` | Jan 1, 2021 01:00–Oct 7, 2026 16:00; 50,342 rows | Oct 7 16:08:03 |
| `current.csv` | Latest row Oct 7 16:00 | Oct 7 16:08:05 |
| `forecast-v2.1.csv` | All 384 rows use origin Oct 7 16:00 | Oct 7 16:22:20 |
| `hourly_forecast_blurbs.csv` | Latest data time Oct 7 16:00; generated 16:15:16; ready | Oct 7 16:15:22 |
| `daily_inflow.csv` | Daily targets through Oct 6 | Oct 7 00:08:10 |
| `daily_visits_forecast.csv` | Cutoff Oct 6; targets Oct 7–13; generated Oct 7 06:16:53 | Oct 7 06:16:54 |
| `daily_visits_forecast_explained.csv` | Same seven dates, cutoff and issued values | Oct 7 06:17:25 |
| `daily_arrival_outlook.csv` | Today intraday cutoff Oct 7 16:00 plus six future daily rows | Oct 7 16:22:58 |
| `daily_visits_prospective_detail.csv` | 277 scored rows, 40 original issue cutoffs | Oct 7 13:34:37 |

The live source, hourly forecast and blurb are fresh. The daily forecast is also
fresh, and its manual-only GitHub workflow intentionally delegates scheduled
publishing to the hospital wrapper during the 06:00 Montreal hour. No stale
workflow repair or duplicate publisher was introduced.

The saved recovery record at
`/Apps/ed-flow-2023/backfill_backups/20261006T213339Z/publication.json`
reports recovery of **42 hours from October 2 15:00 through October 4 08:00**,
adding 471 hourly arrivals while retaining existing rows and backing up six
outputs. Current `allData.csv` independently has complete coverage across this
interval and all October 1–6 dates. The audit record reports preservation of
existing rows; this investigation did not re-parse its source PDF or compare all
backup cells. Actual Power BI rendering/refresh and the original PDF failure
logs remain unverified in #62.

The separate older-history problem is material:

| Finding | Count |
| --- | ---: |
| Missing internal wall-clock slots | 194 |
| Missing slots on DST dates, requiring separate verification | 5 |
| Ordinary incomplete previous days published as numeric daily totals | 69 |
| Such days within the last 1,095 calendar days | 53 |
| Duplicate timestamps / invalid hourly inflow rows | 0 / 0 |
| Published daily sums differing from available hourly sums | 0 |
| Contiguous verified ordinary days ending October 6 | 37 |
| Scored rows using an unverified actual (August 30) | 7 |
| Scored rows whose baseline includes an unverified daily total | 74 |

June 28 has only eight recorded hours and a partial daily sum of **52**. June 29
has 14 hours and **208**; August 30 has 18 hours and **219**. These sums are not
verified full-day arrivals. The producer currently sums all available rows and
drops only the final date; the operational daily wrapper accepts numeric daily
totals without hour-coverage metadata. Consequently, its reported 1,095-day
history includes 53 ordinary incomplete totals. Sum consistency does not prove
target completeness.

The current scoring artifact reports `evidence_ready=True` on sample size/span,
MAE 17.10 versus baseline 18.26, and 71.1% coverage for nominal 80% intervals.
These are the existing published results, **not corrected validation**: seven
actuals use the August 30 partial total, and 74 baseline rows use August 15,
August 30 or June 29 partial totals. Recompute with quality-approved actuals and
baseline days before a promotion decision in #51. Keep every original immutable
forecast. Do not interpret backfilled historical observations as prospective
forecasts for the missed issue dates.

Only quality diagnostics and this report are committed. Raw source files and
complete daily histories remain outside the repository. Twelve regression tests
cover single-hour gaps, absent days, duplicates, invalid counts, DST, boundary
days, internal gaps/invalid counts on the first date, daily mismatches, timestamp
validation and cutoff-safe scoring tracing.
