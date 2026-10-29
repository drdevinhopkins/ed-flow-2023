# Experimental arrival-day correction

The source report labels hourly arrivals 1–24. `get_current.py` stores Time 24
as the next midnight endpoint. Grouping endpoints by calendar date consequently
moves the closing arrival interval into the next day's total. The opt-in v2
target groups arrival intervals by `ds - 1 hour`; raw `ds`, occupancy, backlog,
weather as-of timestamps, and production defaults remain unchanged.

| Report day | Existing stored-calendar total | Interval-day total | Source report total |
|---|---:|---:|---:|
| 2026-10-06 | 323 | 319 | 319 |
| 2026-10-07 | 283 | 286 | 286 |

This establishes agreement with displayed report labels. Independent SQL interval
predicate and historical DST verification remain open. DST days stay quarantined.

## Version separation

- Target: `arrival-day-interval-end-v2`.
- Quality: `daily-arrivals-interval-quality-v2`.
- Intraday model: `intraday-ensemble-interval-v2-2026-10-08`; reduced iteration
  research fits receive an additional `research-iterN` suffix.
- All outputs have separate `interval_v2` names. Existing output files and inputs
  cannot be overwritten. No workflow, timer, publishing destination or route changes.
- Missing, duplicated, negative or nonnumeric intervals never become numeric daily
  targets. The current partial day is excluded from daily training. Intraday live
  prefixes require every endpoint 1 through the current cutoff; midnight is the
  preceding day's closing interval, not a new day's live issue.
- Daily inference and explanation use the same corrected target history and verify
  its fingerprint. Daily and intraday scoring select v2 forecasts before choosing
  original issues and score only against v2 actuals. Legacy forecasts and frozen
  audits must never be relabelled, pooled or rescored as v2 evidence.

## Isolated runner

Provide an immutable local hourly snapshot and a timezone-aware evaluation time.
Use a fresh directory for every invocation:

```bash
python scripts/experiments/arrival_day/run_interval_day.py targets \
  --hourly /path/to/allData.csv --now 2026-10-08T17:52:58-04:00 \
  --output-dir /path/to/new-target-review
```

The verified research environment uses the versions in
`scripts/experiments/arrival_day/constraints.txt`. Install with
`python -m pip install -r chronos-requirements.txt -c scripts/experiments/arrival_day/constraints.txt scikit-learn`.
The isolated CI applies these constraints; production dependency files are unchanged.
Unconstrained CI installed scikit-learn 1.9.1 and failed on the synthetic fit's
all-missing state features. Compatibility with that newer version is unvalidated.

The same command supports `daily`, `intraday`, `backtest`, `score-daily` and
`score-intraday`. Daily and intraday forecasting require a local `--weather-csv`.
Daily inference may download Chronos weights, but does not fetch weather or publish
forecasts. Provide horizon weather of the appropriate forecast vintage; future
observed weather cannot establish prospective performance.

For the full retrospective intraday run, use `backtest --weather-csv ...` with
default 6 expanding folds, 28 test days, 365 minimum training days, 200 iterations,
56 calibration days and 28-day shrinkage. The daily route retains its 28-day hard
minimum and 120-day validated-context warning; the correction does not lower gates.

`score-daily --archive-dir ...` expects local forecast CSV snapshots.
`score-intraday --artifacts ... --runs ... --zip-dir ...` requires original ZIPs
and the audit manifests. ZIP digest and JSON/CSV consistency checks still apply.
An intraday collection without valid original v2 forecasts cannot be scored.

## Validation and promotion

The regression suite exercises missing/duplicate/invalid arrivals, DST quarantine,
immutable endpoint/state values, correct live prefixes, fresh synthetic CPU model
fitting, daily frame/explanation consistency and original-artifact version separation.
Daily inference tests use a fake pipeline to verify data contracts; they do not
measure Chronos accuracy. Reduced-iteration historical smoke runs verify execution,
not the locked ensemble's performance.

Before deployment, complete the full corrected-target backtest and daily model
validation, independently resolve source semantics, and review an explicit producer
and consumer rollout. Collect new original v2 prospective issues. Retain the audit's
28 complete operational days, post-incident streak and per-cutoff error, coverage
and bias requirements. Every runner summary explicitly keeps promotion false.
This PR supplies the isolated correction and validation tools; it does not deploy it.
