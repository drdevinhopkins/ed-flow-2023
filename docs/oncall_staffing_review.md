# On-call staffing review: September 29 regression

Implemented on `fix/oncall-backlog-review`; tracking issue [#60](https://github.com/drdevinhopkins/ed-flow-2023/issues/60). These are proposed changes, not a validated activation policy.

## Decision semantics

Workload escalation is evaluated separately from historical activation probability and associational scenario estimates. Either model can inform review; neither can veto it. The system does not automatically prescribe activation.

Provisional review triggers use canonical Total_TBS:

| Trigger | Definition |
|---|---|
| High current backlog | Current TBS >=50 |
| Sustained backlog | TBS >=45 at three consecutive hourly observations, including now |
| Rapid worsening | Current TBS >=40, rising >=10 over two hours, with all three observations available |
| Forecast persistence | TBS >=45 at >=3 of the next six hourly endpoints, requiring a complete six-hour forecast |

Forecast duration counts hourly endpoints; it is not an interpolated measure of exact elapsed time. Missing observations cannot satisfy a sustained/rising trigger. Boarding/overflow alone does not trigger this physician-capacity rule. High backlog triggers human review even when the forecast predicts rapid improvement.

`STAFFING REVIEW REQUIRED` means assess existing coverage, zone distribution, usable treatment capacity, on-call availability, sick-call use and next-morning duties. Known unavailability redirects review toward alternative coverage. Already active on-call does not remove a persistent-workload review. At 21:00–06:59 the narrative explicitly notes the limited activation window; this clock boundary is provisional, not an institutional rule.

`CONSIDER` can flag a high historical activation probability or a moderate probability with a favourable associational contrast. `NO ESCALATION DETECTED` reports that no workload rule fired and no model review flag fired; it does not establish that on-call is unnecessary. Missing model evidence alone yields `NO CLEAR RECOMMENDATION`. Missing/stale model files do not prevent a workload review when current canonical forecast inputs pass readiness.

The optional schedule export supplies unique scheduled working physician counts now and in four hours, scheduled on-call slots, concurrent regular-role assignments, and next-morning shifts. Counts exclude teaching/on-call roles and deduplicate simultaneous roles for the same physician. Shift overlaps at handoff can temporarily increase the count. Schedule membership is not proof of attendance, availability, activation or sick-call substitution. Historical exports may contain later edits.

An optional `oncall_operational_context.json` can carry **confirmed** availability for one exact data hour:

```json
{"data_hour":"2026-09-29T16:00:00-04:00","availability":"unavailable"}
```

Accepted availability values are `available`, `unavailable`, `already_active`, `unknown`. Stale context is ignored. Without confirmed context, availability remains unknown. No automated confirmed-availability feed currently exists.

## Label and impact safeguards

Missing activation labels stay unknown in both models. Probability training excludes unknown current labels and incomplete future outcome windows; targets use timestamp lookup, not shifted rows across gaps. Hourly reindexing precedes lag features. Old cached models are invalidated by the changed training-spec version. Live predictions retain the live input origin and disclose unknown activation status plus the latest observed label.

The committed label file contains 38,287 contiguous timestamps from January 1, 2022 through May 15, 2026 at 06:00. The code fix does not replenish that file or verify the truth of its historical zeroes. Recent labels must still be acquired from a trustworthy activation record.

The impact script publishes `status=unavailable` and no numeric impact when the required activation history has unknown hours. Its output keeps the existing numeric columns, adding origin/status fields. Consumers must handle blank numeric fields and unavailable status. The deterministic blurb ignores legacy impact files with no verifiable forecast origin and stale origin rows. Occupancy differences are percentage points; converting to an approximate stretcher-patient difference uses 53/100.

## Observed-state replay

Inputs were fetched from Dropbox on October 1, 2026:

- `allDataWithCalculatedColumns.csv`, server modified 23:08:12 UTC, existing canonical `total_tbs` column.
- `shiftadmin/all_shifts.csv`, server modified 14:11:32 UTC.
- `hourly_forecast_blurbs.csv`, server modified 23:14:04 UTC, used to confirm the original recommendations and midnight estimates.

All decision times below are Montreal local. The replay evaluates observations available through each hour. Subsequent outcomes are recorded only after making the decision. It does **not** reconstruct historical forecast trajectories, rerun the models, establish actual on-call use, or estimate benefit.

| Case | First observed-state flag that day | TBS at flag | Schedule evidence |
|---|---|---:|---|
| September 21 | 11:00 | 47 | One scheduled on-call slot; availability unknown |
| September 23 | 15:00 | 45 | One scheduled on-call slot; availability unknown |
| September 29 | 15:00 | 42 | One scheduled on-call slot; availability unknown |

On September 29 TBS increased from 31 at 13:00 to 42 at 15:00, satisfying the worsening rule. At 16:00 there were 56 TBS, satisfying the high-current rule independently of the predicted midnight value of 30. Observed midnight TBS was 50. The 17:00 archived blurb explicitly said "Today's peak appears to have passed"; the new deterministic and LLM paths cannot infer that from a next-day maximum.

The September 29 schedule lists one OC1 slot from 08:00 to 01:00 and no next-morning regular shift for that scheduled physician in the export. This does not prove they were available or had not already been activated. There are 12 unique scheduled regular physicians at the 16:00 handoff overlap, falling to seven by 18:00; temporary shift overlap should not be interpreted as sustained extra coverage.

The observed-only rules flag **48 of 332 recorded hours (14.5%)**, on 12 of the 14 dates from September 18 through October 1. The October 1 extract ends at 19:00. These are flagged hours, not activation episodes or proven missed opportunities. Forecast persistence may add flags in live use; it cannot be evaluated from the archived prose alone. See `validation/oncall-review/replay_2026-09-18_to_2026-10-01.csv`.

Reproduce without uploading or changing production:

```bash
python scripts/evaluation/replay_oncall_review.py \
  --hourly /path/to/allDataWithCalculatedColumns.csv \
  --shifts /path/to/all_shifts.csv \
  --output /tmp/oncall_review.csv
```

## Validation and remaining work

Targeted tests cover workload boundaries, incomplete/gapped labels, clock-hour outcomes, missing model files, an adverse-impact/low-probability September 29 case, availability/late-hour qualifications, next-day peaks, forecast duration, and LLM preservation of the required action. A CPU smoke run trained all three 700-iteration CatBoost classifiers on synthetic data, confirming prediction with unknown live activation features and exclusion of unlabelled outcomes. It is a runtime check, not a model-performance assessment. No GPU Chronos inference or hospital deployment was performed.

Before production adoption: verify refreshed activation records and operational availability; calibrate thresholds against more history and clinician review; distinguish repeated flags from activation episodes; assess prospective alert burden, calibration and outcomes; and validate causal/conditional impact before making benefit claims. The current probability calibrator is still fitted and scored on the same holdout, so its reported calibrated performance remains optimistic. Production schedules, routing, flags, timers and publishing destinations are unchanged by this branch.
