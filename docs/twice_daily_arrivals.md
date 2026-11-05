# Morning and evening daily arrivals forecasts

The hospital hourly wrapper attempts the full daily forecast, explanation and
outlook publication on its first successful cycle at or after **06:00** and
**18:00 America/Montreal**. These are local-time windows, including across DST;
they are not promises of publication precisely on the hour. Morning retries stop
when the evening window starts. An evening catch-up is never labeled morning.
Successful publication is recorded as `YYYY-MM-DD morning` or `YYYY-MM-DD evening`
in the existing completion marker. A former date-only marker represents morning
completion and does not block evening. Failures leave the marker unchanged.

The production model still uses verified completed daily targets plus the existing
calendar/closure and weather features. An evening run obtains current forecast-time
weather. It does not use today's observed pace or insert a predicted daily total
into the observed history. Today's hourly intraday forecast remains unchanged.
The published `daily_arrival_outlook.csv` schema and source URL remain unchanged.

## Forecast and evaluation snapshots

Each forecast and its exact weather inputs retain their existing immutable,
timestamped Dropbox archives. New forecast and weather snapshots additionally carry:

- `forecast_issue_slot`: `morning`, `evening` or `manual`.
- `forecast_issue_day_local`: actual Montréal issue date.
- The existing `forecast_generated_at_utc`: actual generation time, also used to
  calculate lead time. This is retained when an issue is delayed or retried.

Scheduled labels require yesterday's verified target cutoff and a generation time
within that slot's window. Old archives and manual runs are not retrospectively
assigned to a slot. Explanation snapshots preserve the original forecast metadata.

The existing headline score remains the earliest issue per completed-day cutoff.
The scorer also retains the earliest issue within **each** scheduled slot and writes:

| Output | Purpose |
| --- | --- |
| `daily_visits_prospective_issue_detail.csv` | Both scheduled issues, original predictions, verified outcomes and actual lead hours |
| `daily_visits_prospective_issue_summary.csv` | Accuracy and interval coverage separately by slot, horizon and input-quality version |
| `daily_visits_prospective_refresh_comparison.csv` | Morning/evening comparison on matched target dates and input-quality versions |

Positive paired `mae_improvement` means evening has smaller absolute error. A failed
or missing slot never borrows an unmatched outcome from the other slot. Repeated
issues cannot be cherry-picked after seeing outcomes. Collection readiness is
computed separately for each slot and quality version. It is not an accuracy or
calibration promotion decision.

Existing `horizon_day` labels are relative to the last **completed** day. Thus
D+1 is the issue day, D+2 is tomorrow, and D+7 is six days after the issue day.
The new score outputs expose `calendar_days_ahead` to make that distinction clear.
`lead_hours_to_target_start` is measured to local midnight at the start of the target
day; it is negative for a forecast of the current day issued after midnight.

## Intraday-informed next-day experiment

`scripts/evaluation/backtests/backtest_next_day_intraday.py` is an isolated,
retrospective experiment. It compares a fixed calendar/history ridge model with
that same model plus observed arrival-prefix features at 06:00 or 18:00, predicting
the next local calendar day's total. The weekday baseline is a third comparator.

Features include observed arrivals, recent hourly arrivals, pace versus past
same-weekday prefixes and a simple projected total for today. That projection is
an uncertain predictor, never a replacement for an observed daily target.
Both model variants train separately by issue slot. At each issue, training
outcomes must end strictly before that issue's date; fitting/scaling use those
training rows only. Current-day final totals and tomorrow's arrivals cannot enter
predictors or training. Missing, duplicated or invalid prefix hours are excluded.
Targets and historical daily inputs use the shared verified-target policy.

The fixed ridge penalty is 20; the training window is at most 1,095 days with at
least 365 completed training outcomes. Evaluation uses a two-calendar-year window,
with quarterly slices and all eligibility exclusions preserved. The runner has
no publishing option and is not called by the hospital wrapper.

This initial experiment does not compare against archived Chronos morning/evening
issues and does not reconstruct historical forecast-time weather. Raw historical
hourly data can contain later revisions. It is screening evidence, not prospective
validation or a basis for replacing the operational forecast. Results and provenance
are in `validation/next-day-intraday-2026-11-05/`.

```bash
python scripts/evaluation/backtests/backtest_next_day_intraday.py \
  --hourly-csv /path/to/allData.csv \
  --test-days 730 \
  --now 2026-11-05T19:30:00Z \
  --output-dir /path/to/isolated/results
```

After merge, pull `main` on the hospital host. No timer restart or completion-marker
reset is needed. Verify that the next due slot writes a fresh seven-row daily
forecast, matching explanation metadata, and the updated outlook. Live inference,
Dropbox publication and the resulting Power BI refresh still require host-side
verification; offline tests do not establish that deployment succeeded.
