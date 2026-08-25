# Triage hallway hourly feature ablation

This validation supports adding `TRG_HALLWAY1` (triage hallway occupancy) and `TRG_HALLWAY_TBS` to the routed hourly Chronos-2 forecasts.

## Design

- 8 common weekly cutoffs
- 24-hour forecast horizon
- the same complete 8-target Chronos-2 bundle used by the proposed forecast pipeline
- scenarios: baseline, calendar demand, raw weather, raw weather plus snow/recovery, current staffing, and staffing structure plus physician-effect features
- results summarized by the production horizon bands: 1-4h, 5-8h, 9-12h, and 13-24h

Historical weather is a stitched/revised series rather than archived forecast-time snapshots. Weather results therefore remain signal-potential estimates and are not eligible for unguarded production routing.

## Selected non-weather routes

| Target | 1-4h | 5-8h | 9-12h | 13-24h |
| --- | --- | --- | --- | --- |
| TRG_HALLWAY1 | staffing_current (+16.37% MAE) | calendar_demand (+3.55%) | calendar_demand (+3.04%) | staffing_current (+4.68%) |
| TRG_HALLWAY_TBS | staffing_structure_effects (+12.05%) | calendar_demand (+4.82%) | calendar_demand (+3.74%) | staffing_current (+1.52%) |

Percentages are aggregate MAE improvements relative to the history-only baseline across the eight cutoffs. The 1-4h improvements are the strongest and most consistent; later-horizon gains are smaller and should be interpreted accordingly.

No weather scenario won a triage-hallway horizon band.

The detailed per-cutoff rows and all aggregate tables are retained in this directory for audit and future re-evaluation.
