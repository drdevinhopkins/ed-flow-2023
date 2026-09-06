# Power BI: ED Arrival Outlook

## Purpose

Use `daily_arrival_outlook.csv` as the single presentation-layer source for ED daily arrival forecasting.

The pipeline already decides which forecast is the best current estimate:

- `horizon_day = 0`, `forecast_stage = intraday`: today's guarded intraday completion forecast when valid.
- `horizon_day = 0`, `forecast_stage = day_ahead`: today's daily forecast when no valid intraday estimate is available.
- `horizon_day = 1..7`, `forecast_stage = daily`: future daily Chronos-2 forecasts.

Do not merge the intraday and daily model outputs again in Power BI. That routing is deliberately handled upstream in `scripts/build_daily_arrival_outlook.py`.

## Dataflow Gen2 ingestion

1. Create a direct/raw Dropbox shared URL for `/daily_arrival_outlook.csv`.
2. In Dataflow Gen2, create a text parameter named `DailyArrivalOutlookUrl` containing that raw URL.
3. Create a blank query and paste `powerbi/daily_arrival_outlook_powerquery.m` into Advanced Editor.
4. Set the destination to the existing ED Flow Lakehouse and name the table `daily_arrival_outlook`.
5. Use **Replace** semantics. The file intentionally contains only the current best estimate for today through seven days ahead; it is not a historical archive.
6. Refresh the semantic model after the Dataflow refresh, or use Direct Lake if the existing model architecture supports it.

The query validates `lower_80 <= predicted_arrivals <= upper_80` before loading.

## Semantic-model measures

Create the measures in `powerbi/ed_arrival_outlook_measures.dax` on the `daily_arrival_outlook` table.

Important: use `horizon_day = 0` to identify today instead of `TODAY()`. The pipeline computes `horizon_day` using `America/Montreal`; this avoids Power BI Service UTC date-boundary errors.

## Recommended page layout

### Page title

**ED Arrival Outlook**

Subtitle: **Expected ED arrivals today and over the next 7 days**

### Row 1 — today's outlook

Use a wide card group across the top.

**Primary card**

- Value: `[Arrival Outlook Today]`
- Title: `Expected arrivals today`
- Subtitle / reference label: `[Arrival Outlook Today Summary]`

The primary card should be visually dominant.

**Observed card**

- Value: `[Arrivals Observed Today]`
- Title: `Arrived so far`
- Hide or allow blank when the day-ahead fallback is active.

**Remaining card**

- Value: `[Arrivals Expected Remaining Today]`
- Title: `Expected remaining`
- Hide or allow blank when the day-ahead fallback is active.

**Baseline comparison card**

- Value: `[Arrival Outlook Today Delta vs Baseline]`
- Title: `vs typical same weekday`
- Show explicit +/− formatting.

Small metadata text beneath the cards:

- `[Arrival Outlook Today Stage]`
- `[Arrival Outlook Today Generated]`
- `[Arrival Outlook Today Source]`

This makes it clear whether the current number is intraday or day-ahead without forcing users to understand model names.

### Row 2 — today trajectory / uncertainty

Left two-thirds: a simple comparison visual for today's final total.

- Actual/observed-so-far marker when intraday is available.
- Predicted final total.
- 80% lower and upper bounds.
- Seasonal same-weekday baseline as a reference marker.

Do not present the 80% interval as a probability that the exact rounded count will occur. Label it simply `80% forecast range`.

Right one-third: explanatory text card.

- Value: `[Arrival Outlook Explanation Today]`
- Heading: `Why this forecast?`

For the current intraday model this text describes observed arrivals, predicted total and expected remaining arrivals. Future daily rows carry driver attribution from the daily explainability layer.

### Row 3 — next 7 days

Use a line-and-range or column visual with:

- X axis: `target_date`
- Main value: `predicted_arrivals`
- Lower bound: `lower_80`
- Upper bound: `upper_80`
- Reference: `seasonal_weekday_baseline`

Filter to `horizon_day = 1..7` if today's card section already covers `horizon_day = 0`.

Recommended tooltip fields:

- `forecast_stage`
- `predicted_arrivals`
- `lower_80`
- `upper_80`
- `seasonal_weekday_baseline`
- `delta_vs_baseline`
- `top_driver_1`
- `top_driver_1_effect`
- `top_driver_2`
- `top_driver_2_effect`
- `top_driver_3`
- `top_driver_3_effect`
- `explanation_text`

### Row 4 — forecast drivers

Use a table or compact matrix for `horizon_day = 1..7`:

- Date
- Predicted arrivals
- Delta vs same-weekday baseline
- Top driver 1 / effect
- Top driver 2 / effect
- Top driver 3 / effect

Keep this below the operational forecast. Management should see the expected demand first and the model explanation second.

## Field contract

| Field | Meaning |
| --- | --- |
| `target_date` | Calendar date being forecast |
| `generated_at_local` | Forecast generation time in Montreal-local offset-aware form |
| `forecast_stage` | `intraday`, `day_ahead`, or `daily` |
| `horizon_day` | 0 for today, 1..7 for future days |
| `predicted_arrivals` | Best current estimate of total ED arrivals for target date |
| `lower_80` | Lower bound of nominal 80% forecast interval |
| `upper_80` | Upper bound of nominal 80% forecast interval |
| `observed_arrivals` | Arrivals observed so far; populated for valid intraday forecast |
| `expected_remaining` | Predicted additional arrivals before midnight; populated for intraday forecast |
| `seasonal_weekday_baseline` | Recent same-weekday baseline used for context |
| `delta_vs_baseline` | Prediction minus baseline |
| `top_driver_1..3` | Grouped explanatory driver names from daily model |
| `top_driver_1_effect..3_effect` | Estimated effect in arrivals relative to neutralized counterfactual |
| `explainability_method` | Explanation method identifier |
| `explanation_text` | Human-readable explanation |
| `source_model` | Presentation-layer source route |
| `data_cutoff` | Latest data available to that forecast |
| `model_version` | Model/version identifier |

## Prospective accuracy page / later addition

Do not mix forecast accuracy into the main operational page initially. The prospective scorer currently has limited evidence and explicitly gates interpretation with `evidence_ready`.

Once enough prospective issue dates have matured, add a separate **Forecast Performance** page with:

- MAE by D+1..D+7 horizon.
- Same-weekday baseline MAE beside model MAE.
- Relative improvement vs baseline.
- 80% interval coverage.
- Sample count and date span.
- An obvious `evidence_ready` indicator.

This prevents early validation noise from distracting from the operational demand view while preserving model-governance transparency.

## Acceptance checklist

The ED Arrival Outlook page is ready for routine use when:

- `daily_arrival_outlook` refreshes successfully from Dropbox into the Lakehouse.
- Exactly one row exists per `target_date`.
- `horizon_day = 0` has exactly one row.
- Today switches automatically from `day_ahead` to `intraday` when a valid intraday forecast becomes available.
- The main card, lower/upper range, observed count and remaining count agree with the source CSV.
- Future D+1..D+7 rows continue to come from `source_model = daily_chronos2`.
- Refresh failures leave the last valid report visible rather than loading malformed intervals.
