# Candidate metric ablation decision note

Status: **pre-production only**. This note interprets the 8-common-cutoff, 24-hour Chronos-2 ablation recorded in this directory. It does not change production targets or routing.

## Decision rule

For a non-weather route to be considered robust enough for prospective evaluation, require all of the following within a horizon band:

1. aggregate MAE improves over the history-only baseline;
2. mean paired cutoff improvement is positive;
3. median paired cutoff improvement is positive; and
4. the scenario beats baseline on at least 5 of 8 common cutoffs.

Weather scenarios remain retrospective signal-potential only and are not promotion candidates until forecast-time weather validation is available.

## Candidate interpretation

### Workup_Delay_Burden — strongest candidate, but not uniformly robust

- **1–4 h:** `staffing_current` improves aggregate MAE by 2.08%; 5/8 cutoff wins; positive mean and median paired improvement. **Advance to prospective evaluation.**
- **5–8 h:** `calendar_demand` improves aggregate MAE by 6.99%; 6/8 cutoff wins; positive mean and median paired improvement. **Advance.**
- **9–12 h:** `staffing_current` improves aggregate MAE by 10.58%; 5/8 cutoff wins; positive mean and median paired improvement. **Advance.**
- **13–24 h:** `staffing_current` improves aggregate MAE by 6.57%, but only 3/8 cutoffs improve and the median paired improvement is negative. The aggregate result is therefore likely driven by a minority of large wins. **Keep baseline / do not route yet.**

This composite is a burden index, not a unique-patient count; a patient can contribute to multiple delayed-process components.

### INFLOW_STRETCHER — useful short/intermediate-horizon target

- **1–4 h:** `staffing_current` improves aggregate MAE by 3.21%; 5/8 wins; positive mean and median. **Advance.**
- **5–8 h:** aggregate-best `staffing_current` improves MAE by 1.92% but wins only 4/8 cutoffs. `calendar_demand` is slightly less favorable in aggregate but wins 6/8 with positive mean/median paired improvement. **Prefer robustness-aware `calendar_demand` for prospective testing rather than the raw aggregate winner.**
- **9–12 h:** `staffing_current` improves aggregate MAE by 2.90%; 6/8 wins; positive mean and median. **Advance.**
- **13–24 h:** no convincing non-weather improvement. **Keep baseline.**

### AdmissionRequests_New — selective signal

- **1–4 h:** no robust non-weather improvement. **Keep baseline.**
- **5–8 h:** `calendar_demand` wins 7/8 cutoffs with positive mean/median paired improvement; `staffing_structure_effects` has the best aggregate MAE improvement (4.30%) but 6/8 wins. **Advance, with calendar route favored for robustness unless prospective data supports the aggregate-best route.**
- **9–12 h:** no robust non-weather improvement. **Keep baseline.**
- **13–24 h:** `staffing_structure_effects` improves aggregate MAE by 2.13% and wins 8/8 cutoffs with positive mean/median paired improvement. **Advance.**

Because this is a low-count hourly increment, absolute-error metrics should be supplemented by calibration and threshold/event metrics during prospective scoring.

### INFLOW_AMBULANCES — short-horizon only

- **1–4 h:** `staffing_structure_effects` improves aggregate MAE by 4.70%; 6/8 wins; positive mean and median. **Advance.**
- **5–8 h:** aggregate improvement is only 0.34% and the aggregate-best calendar route wins 2/8 cutoffs with negative median paired improvement. **Keep baseline.**
- **9–12 h:** no non-weather improvement. **Keep baseline.**
- **13–24 h:** aggregate improvement is effectively zero (0.04%). **Keep baseline.**

### Inflow_Total — useful mainly at 1–4 h

- **1–4 h:** `calendar_demand` improves aggregate MAE by 4.56%; 5/8 wins; positive mean/median paired improvement. **Advance.**
- **5–8 h and 9–12 h:** all safe feature routes regress relative to baseline. **Keep baseline.**
- **13–24 h:** nominal aggregate gain is only 0.06% and is not operationally meaningful. **Keep baseline.**

## Provisional prospective target set

Advance all five candidate series into the prospective scoreboard so their raw predictability can be measured in real forecast-time conditions, but only treat the horizon bands above as candidate feature-routed forecasts. For all other bands, retain the history-only baseline as the challenger.

Do **not** add these targets to production outputs yet. The next step is to wire a branch-only prospective scoring schema that records forecast issue time, target, horizon, scenario/route, point forecast, observed value when available, absolute error, signed error, squared error, WAPE denominator contribution, and route-vs-baseline comparison.
