# Next-day arrivals: first intraday-prefix experiment

This is a retrospective screening experiment, isolated from production. The fixed
calendar/history ridge control is compared with the same model plus today's
observed prefix and a past-prefix-based estimate of today's total. Neither model
uses today's final observed total. All training labels end before the issue day.

| Issue slot | Eligible forecasts | Calendar/history control MAE | With intraday features MAE | Change in MAE |
| --- | ---: | ---: | ---: | ---: |
| 06:00 Montréal | 285 | 16.35 | 16.65 | +1.85% (worse) |
| 18:00 Montréal | 283 | 16.34 | 16.69 | +2.18% (worse) |

The evaluation window is November 4, 2024–November 3, 2026 (730 calendar days).
Eligibility requirements leave 285 distinct issue dates and 568 issue-slot rows;
this is not 730 consecutive complete observations. The source has gaps, and the
prefix, recent-history, verified-target and training-count gates exclude cases.
Quarterly breakdowns, including absent quarters, must be considered when interpreting
the aggregate. The evening candidate improves in 2026Q4 on 34 forecasts, but its
overall performance does not support promotion.

Decision: keep the candidate offline. These results do not establish whether
updated forecast-time weather makes the production Chronos evening refresh more
accurate. That requires the newly archived morning/evening prospective issues.

`detail.csv` records each issue, its training target cutoff and paired model
predictions. `summary.csv` includes weekday baseline, control and candidate metrics
by slot and quarter. `provenance.json` records the source hash, audit time and fixed
settings. No source data, issued production forecasts or routes were modified.

Limitations: historical source rows may contain later revisions; this is not a
replay of immutable hourly vintages. No archived Chronos or historical forecast-time
weather comparator was used. Uncertainty intervals and prospective calibration
have not been evaluated for this ridge candidate.
