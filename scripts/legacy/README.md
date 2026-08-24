# Legacy scripts

This directory contains historical prototypes retained for reproducibility and context. Files here are **not production entry points** and may depend on older paths, assumptions, or data contracts.

## `chronos_forecast_v2.py`

First-generation on-call counterfactual prototype from May 2026. It compares a 24-hour no-on-call scenario with an 8-hour on-call scenario using `oncall_busy` as a future covariate.

Current replacements:
- hourly operational forecast v2: `scripts/hourly_forecast_v2.py`
- on-call activation probability: `scripts/forecast_oncall_probability.py`
- on-call impact scenarios: `scripts/forecast_oncall_impact.py`

See `docs/2026-05-10_oncall_impact_analysis.md` for the historical analysis.
