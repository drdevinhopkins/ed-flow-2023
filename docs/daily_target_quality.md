# Verified daily arrival targets

Follow-on to the merged [history audit](history_quality_audit.md) and
[#65](https://github.com/drdevinhopkins/ed-flow-2023/issues/65).

## Producer and consumer contract

`scripts/daily_arrival_quality.py` shares the audited completeness rule across
the daily publisher, operational forecast, explanation and prospective scorer.
The hourly source is preserved. Daily aggregation still uses the stored naive
Montreal `ds` calendar date, including 00:00; no missing arrivals are imputed.

`get_current.py` preserves the two-column `daily_inflow.csv` contract:
`ds,Daily_Inflow_Total`. Prior dates with incomplete or invalid coverage have a
blank target. Completely absent internal dates are retained with a blank target.
The current Montreal date is excluded using the calendar, rather than always
dropping the final available date. Complete ordinary-day totals are unchanged.

The companion `daily_inflow_quality.csv` retains the available observed sum,
observed/valid row counts, distinct hours, duplicate count, missing clock hours,
DST and boundary status, matching-sum flag, `audit_eligible`, quality version,
source hourly cutoff and verification time. An observed partial sum is not a
verified daily target. Eligibility requires all 24 unique ordinary clock slots,
finite nonnegative arrivals, no duplicates and a matching daily sum.

DST dates remain unverified even if all 24 naive clock slots exist. Upstream
interval-end and repeated-hour semantics require source confirmation; this
change does not decide them or recover older missing hours.

During mixed-version rollout, consumers download both `/daily_inflow.csv` and
`/allData.csv` and independently verify coverage and sums. They do not require
the companion file to exist. A mismatch masks that date. The operational wrapper
requires yesterday's Montreal target to be verified; it refuses to silently
forecast from an older cutoff. Scoring can still evaluate older verified dates.

Internal blank targets break the contiguous Chronos context. The existing
28-day model minimum, 1,095-day maximum context, calendar/weather route and
120-day validated-context warning threshold are preserved. Forecasts add
`target_quality_version=daily-arrivals-quality-v1` and `target_history_sha256`.
Explanations require the same historical target fingerprint and context length,
so an old forecast cannot be attributed to a corrected, different context.
The explanation comparator uses the most recent eight verified same-weekday
observations at or before the issued cutoff, including older observations
outside the model's contiguous context, matching prospective scoring.

The manual forecast workflow's output check now honors the existing 28-day
minimum and verifies the below-120-day warning and quality metadata. The scorer
adds a quality-version summary to its artifacts. Timers, destinations of existing
outputs and model routing are unchanged. A separate CI workflow runs contract
tests without credentials, publishing or downloading model weights.

## Corrected replay on October 7, 2026

The replay uses all 63 original immutable forecast snapshots, retaining the
earliest issue per cutoff. It never reruns or alters issued predictions. Raw
hourly/daily source files and the archive remain outside the repository; SHA-256
provenance and corrected evaluation outputs are in
`validation/daily-target-quality-2026-10-07/`.

| Measure | Corrected result |
| --- | ---: |
| Matured rows / issue cutoffs | 270 / 40 |
| Excluded incomplete outcomes (August 30) | 7 |
| Retained original forecast rows unchanged | 270 |
| Retained comparator values changed | 68 |
| Forecast MAE / verified weekday baseline MAE | 16.91 / 17.90 |
| MAE improvement | 5.5% |
| Forecast bias | −4.65 arrivals |
| Nominal 80% interval coverage | 73.0% |
| Collection sufficient by original count/span rule | Yes |
| Corrected-route `evidence_ready` | False |

Of 2,105 prior calendar dates, 81 are masked: 69 ordinary incomplete days,
11 DST dates and one leading partial date. All 2,024 remaining totals match the
original daily values exactly. The current verified model context is **37 days**,
August 31–October 6, with `short_context_warning=True`.

Corrected accuracy describes forecasts issued with legacy target inputs. It
does not validate the new 37-day context or justify promotion in #51.
`evidence_ready` now requires both verified actuals and a sufficient collection
of forecasts issued with the corrected input version. The existing overall
eight-row summary remains, and `daily_visits_prospective_quality_summary.csv`
separates legacy and corrected collections so legacy sample size cannot make a
new collection ready. Readiness is an evidence-availability gate, not an
accuracy or calibration go decision. Restart corrected-route collection after
deployment; preserve the legacy archive.

Reproduce locally without publishing:

```bash
python scripts/evaluation/prospective/score_daily_visits_forecast.py \
  --daily-csv /path/to/daily_inflow.csv \
  --hourly-csv /path/to/allData.csv \
  --archive-dir /path/to/daily_visits_forecast_snapshots \
  --now 2026-10-07T21:00:48.437708+00:00 \
  --detail-output /path/to/output/detail.csv \
  --summary-output /path/to/output/summary.csv \
  --quality-summary-output /path/to/output/quality-summary.csv \
  --no-dropbox-output
```

Local replay requires all three source arguments and `--no-dropbox-output`.
The original reporting time and each snapshot hash are in `provenance.json`.

## Deployment verification still required

After merge, verify hospital checkout adoption and its next scheduled producer
and forecast runs. Check the two-column target file, quality companion, fresh
Montreal cutoff, actual context length/warning, seven-day forecast and matching
explanation fingerprint, then verify scoring version separation. Full Chronos
inference and live publication are not exercised by these offline contract tests.
Keep #65 open until those checks and upstream timestamp/DST verification are
complete. No existing historical backtest result is relabeled as validated under
the corrected policy; rerun research separately when needed.
