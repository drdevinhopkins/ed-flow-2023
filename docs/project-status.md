# ED Flow Forecasting Project Status

_Last updated: 2026-08-23_

This document is the repository-level status summary for the current forecasting work. Detailed task tracking remains in GitHub issues, with #24 as the master feature-engineering roadmap.

## Current architecture

The project now deliberately keeps three modeling problems separate:

1. **Daily ED arrivals / inflow**
2. **Hourly operational flow**
3. **On-call decision support**

## Daily arrivals

### Calendar / holiday engineering
Status: **completed and merged**.

Includes Quebec/federal/institutional holidays, major Jewish holidays/eves, closure structure, long-weekend shoulders, pre/post-holiday effects, and related native Chronos-2 ablations.

### Weather engineering
Status: **completed and merged** via PR #21.

Validated `raw_plus_snow` representation improved daily-arrival MAE by approximately 6.3% versus the calendar/closure control in the research-history configuration, with larger gains on weather-event and post-major-snow days.

Staffing is intentionally excluded from the default daily-arrivals model because scheduled staffing is operational supply rather than an arrival driver.

## Hourly operational flow

Targets:
- `Total_TBS`
- `POD_TBS`
- `Vertical_TBS`
- `TTStr`
- `Overflow`
- `WAITINGADM`

### Feature engineering
Status: **completed**.

Validated families include:
- calendar / demand structure
- raw and engineered weather
- current/simple staffing
- engineered staffing structure
- physician-aware staffing representations and shrunk historical effects

Final common-base native Chronos-2 ablation:
- workflow run `32675780990`
- result: no universal feature bundle; the best representation depends on target and forecast horizon

### Safe target × horizon routing

Weather routes remain disabled by default pending prospective forecast-time validation.

| Target | h1–4 | h5–8 | h9–12 | h13–24 |
|---|---|---|---|---|
| Overflow | baseline | baseline | baseline | baseline |
| POD_TBS | staffing_structure_effects | baseline | staffing_current | staffing_current |
| TTStr | staffing_current | staffing_current | staffing_current | calendar_demand |
| Total_TBS | calendar_demand | staffing_current | staffing_current | staffing_current |
| Vertical_TBS | calendar_demand | staffing_current | staffing_current | staffing_current |
| WAITINGADM | staffing_structure_effects | staffing_structure_effects | staffing_structure_effects | baseline |

### Independent forecast v2
Status: **merged and validated** via PR #25.

The legacy hourly pipeline is intentionally unchanged.

A separate workflow (`hourly-forecast-v2.yml`) runs after a successful legacy `Hourly update` and writes only:
- `forecast-v2.csv`

The v2 output is long format with 288 rows:
- 6 targets × 24 historical observed hours
- 6 targets × 24 future forecast hours

Power BI-oriented schema:
- `actual` only on historical rows
- `forecast`, `forecast_lower`, `forecast_upper` only on future rows
- `row_type` distinguishes observed vs forecast
- anomaly reference intervals on all rows
- separate actual and forecast anomaly / colour fields
- route metadata on future rows

Runtime validation of the 288-row contract:
- workflow run `32682087737`

### Directional historical anomaly colours
Status: **merged and validated** via PR #26.

Historical actual colours now encode operational desirability:
- above upper anomaly bound → red `#D13438`
- above expected but within upper bound → amber `#FFB900`
- at/below expected, including below the lower anomaly bound → green `#107C10`

A below-lower-bound value still retains `actual_anomaly = yes` for statistical tracking; only its display colour is green.

Forecast colour logic remains two-sided.

Runtime validation including synthetic directional-colour regression cases:
- workflow run `32682987665`

## Physician staffing

Status: **feature-engineering issue completed** (#23).

Important interpretation:
- physician identity and historical physician effects were tested
- raw identity was not a universal winner
- simpler/current staffing and engineered structure were promoted only in the target/horizon cells where they improved validation
- physician-associated effects are predictive associations, not causal productivity scores

## On-call decision support

Status: **active / not complete**.

### Probability model — #22
Current CatBoost 4h/6h/8h activation models achieved approximate AUROC:
- 4h: 0.91
- 6h: 0.88
- 8h: 0.85

Remaining priorities:
- verify historical activation-label completeness
- independent calibration / Brier validation
- baseline comparison and feature ablations
- prospective probability logging and calibration monitoring

### Counterfactual impact model — #7
Current model forecasts no-on-call versus 4h/6h/8h activation scenarios.

These outputs are model-based counterfactuals, **not causal treatment-effect estimates**.

Remaining priorities:
- test validated staffing representations within the impact model
- matched-state / propensity / doubly robust / heterogeneous-treatment-effect validation
- prospective validation
- combine probability + expected impact in an operational Power BI view

## Near-term priorities

1. Build and validate the new Power BI report against `forecast-v2.csv`.
2. Accumulate prospective weather snapshots and re-test the guarded hourly weather routes.
3. Complete on-call label-quality and independent calibration validation.
4. Perform causal/matched-state validation of on-call impact scenarios.
5. Combine on-call probability and expected impact into decision-support reporting.

## Tracking

- #24 — master feature-engineering roadmap
- #23 — physician staffing feature engineering (**completed**)
- #18 — forecast input gates / initial covariate ablation (**completed**)
- #22 — on-call activation probability
- #7 — on-call counterfactual impact
- PR #25 — independent target/horizon-routed hourly forecast v2
- PR #26 — directional historical anomaly colours
