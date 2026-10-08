# Intraday prospective audit — October 7, 2026

Decision: **no-go for decision-support promotion**. Keep the frozen companion
experimental. This audit changes no production model, inputs, workflow, output or
publication destination. It adds a local read-only scorer and isolated offline CI.

## Evidence and original-issue safeguards

The evidence cutoff is October 7 at 22:37:32 America/Montreal
(`2026-10-08T02:37:32Z`); matured outcomes end October 6. Results are deliberately
frozen at this cutoff, not represented as the latest live collection.

Recovered 728 retained `intraday-daily-inflow-forecast` ZIP artifacts from main
workflow runs in the supplied August 28–October 7 local-time inventory. Every ZIP's
SHA-256 matches GitHub's recorded digest. The earliest retained artifact is September
3 evening; the archive is left-censored. The two preintegration August days from
#49 are not pooled into this separately verifiable postintegration collection.

Status JSON, not a possibly leftover latest CSV, determines whether a run issued a
forecast. There are 517 successful forecast artifacts, 210 data-quality/window
suppressions and one model-error suppression (missing
`.cache/intraday/weather_backfilled.csv`). Most quality suppressions are expected
outside 06:00–22:00; one reports input age 140 minutes, exceeding the 90-minute gate.
Missing artifacts do not imply successful or explicitly suppressed forecasts.

Successful issues must agree between JSON and CSV, have finite nonnegative values,
consistent cutoff/day/issue timestamps, valid interval/remaining-arrival identities,
and artifact retention before the forecast day ends. Choose the earliest issue for
each day/hour/model version, not the most accurate revision. Two later repeats are
excluded, leaving 515 distinct issues; 16 October 7 issues are not yet matured.
No forecast was reconstructed, refitted, recalibrated or retrospectively replaced.

499 original forecasts on 31 dates have verified outcomes; all belong to
`intraday-ensemble-v1-2026-08-28`. Actuals are independently constructed from hourly
history using `daily-arrivals-quality-v1`: incomplete, duplicate, invalid, leading
partial and unverified DST dates cannot become full-day outcomes. Authoritative
October backfills may contribute to actuals; they never alter original predictions
or their recorded prior-update baselines. No matured outcome in this retained
window fails the completeness checks.

## Results

Bias is forecast minus actual. Coverage is inclusive P10–P90 coverage. Units are
arrivals, except improvement and coverage. Rows on the same day share an outcome;
these are descriptive issue-weighted metrics, not independent patient observations
or a statistical significance claim.

| Scope | Issues / days | MAE | Baseline MAE | Improvement | Bias | P80 coverage | Mean interval width |
|---|---:|---:|---:|---:|---:|---:|---:|
| All model hours 06–22 | 499 / 31 | 11.07 | 12.99 | 14.75% | +0.82 | 76.35% | 35.64 |
| Available operational hours 11–18 | 234 / 30 | 11.13 | 12.68 | 12.21% | −0.51 | 74.36% | 35.28 |
| Complete operational days only | 216 / 27 | 11.27 | 12.60 | 10.48% | −0.48 | 74.54% | 35.47 |

| Montreal cutoff | Issues | MAE | Baseline MAE | Bias | P80 coverage | Mean width |
|---|---:|---:|---:|---:|---:|---:|
| 11:00 | 30 | 15.78 | 19.15 | +3.32 | 73.33% | 44.83 |
| 12:00 | 30 | 15.35 | 18.10 | +1.03 | 63.33% | 42.01 |
| 13:00 | 29 | 13.78 | 15.30 | +0.96 | 68.97% | 37.85 |
| 14:00 | 30 | 10.34 | 11.73 | −1.17 | 73.33% | 35.09 |
| 15:00 | 28 | 8.88 | 9.48 | −2.07 | 78.57% | 34.19 |
| 16:00 | 29 | 8.88 | 9.76 | −3.16 | 86.21% | 34.33 |
| 17:00 | 29 | 8.37 | 9.28 | −2.26 | 75.86% | 28.56 |
| 18:00 | 29 | 7.32 | 8.17 | −0.92 | 75.86% | 24.80 |

Other hours are reported separately in `metrics.csv`; their collection does not
promote them into the operational window. At 22:00, model MAE 3.52 is worse than
baseline 2.30 (30 issues), reinforcing the separate 19:00–22:00 acceptance gate.

## Collection and readiness

27 dates have all eight original 11:00–18:00 cutoffs. September 3 is left-censored;
September 4 lacks 13:00 and September 16 lacks 15:00. October 2 retains only 11:00–14:00;
October 3–5 have no operational issues. These absences are consistent with the
documented October input incident (#62), but the audit does not assign every missing
artifact a cause. Recovering actuals cannot recover unissued forecasts.

The longest complete collection streak is 15 days (September 17–October 1).
The latest matured streak is only one day (October 6). The original requirement
for seven consecutive days has been observed; requiring a fresh postincident
seven-day streak is an additional restart check, not a rewritten original gate.

| Prespecified check | Observed | Result |
|---|---|---|
| At least 28 complete prospective days | 27 | Fail |
| Seven consecutive complete days observed | Longest 15 | Pass; latest streak 1 |
| At least 5% MAE improvement overall and operationally | 14.75% / 12.21% | Pass |
| Absolute overall bias ≤2 | 0.82 | Pass |
| Every operational-hour absolute bias ≤3 | 11:00 +3.32; 16:00 −3.16 | Fail |
| P80 coverage 75–85%, overall and operationally | 76.35% / 74.36% | Operational fail |
| Upstream day-boundary semantics, monitoring and Power BI acceptance | Not independently resolved | Pending |
| Explicit operational go | Not granted | No-go |

Do not tune the frozen version on this sample. Resolve the target definition first,
continue prospective collection, and only then design an isolated recalibration
study. An accepted model/target change must receive a new version and new validation.

## Midnight boundary: evidence, not a silent correction

The trained target and primary scoring target sum `Inflow_Total` by the stored
naive Montreal `ds` calendar date, **00:00–23:00**. Observed arrivals are reconstructed
the same way, not copied from `Inflow_Cum_Total`.

Across all 34 comparable complete source days September 3–October 6, every hourly
cumulative counter exactly equals the sum of **01:00 through the next 00:00**.
31/34 days have different totals under the two conventions. This strongly supports
an interval-ending report convention but is not independent upstream confirmation.
For October 6, the stored-calendar total is 323 versus 319 under the shifted sum
and next-midnight counter (remove the opening 00:00 inflow 9, add closing inflow 5).

The model's label "total by midnight" may therefore differ from the operational
report's day. Do not switch only scoring actuals, shift history, change cumulative
features or retroactively relabel issued forecasts. Verify the upstream report's
interval convention and DST behavior under #65, then review an explicit coordinated
target/version migration if necessary. Current scores evaluate the target actually
trained; they do not validate a newly shifted operational target.

## Reproduction and provenance

Checked-in evidence: `validation/intraday-prospective-2026-10-07/` contains scored
aggregate issues, hourly metrics, complete-day calendar, exclusions, artifact/run
IDs and head SHAs, original ZIP digests, boundary diagnostics and input SHA-256s.
Raw hospital hourly inputs and original ZIPs are not committed. Re-download retained
ZIPs by artifact ID through GitHub before retention expires; an expired or mismatched
archive is quarantined. Reproduction requires the exact private hourly snapshot
with the recorded hash, the original artifact/run inventories and these ZIPs.

```bash
python scripts/evaluation/prospective/audit_intraday_forecasts.py \
  --artifacts /path/to/artifacts.json --runs /path/to/runs.json \
  --zip-dir /path/to/zips --hourly /path/to/allData-audit.csv \
  --now 2026-10-08T02:37:32Z --output-dir /tmp/intraday-audit
python -m unittest discover -s tests -p test_intraday_prospective_audit.py -v
```

The eight offline tests cover suppression precedence, ZIP hash quarantine, earliest
duplicate selection, issue invariants, end-of-day retention, outcome completeness
and maturity, missing-date streaks, and diagnostic-only boundary comparison.
This audit does not introduce a scheduled collector or alter artifact retention.

Related: #49 (prospective readiness), #65 (source/target semantics), #24 (roadmap).
