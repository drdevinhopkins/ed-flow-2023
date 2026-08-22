# On-call probability model

Implemented on branch `codex` in `scripts/forecast_oncall_probability.py`.

The model estimates the probability that the ED will activate the on-call physician within the next 4, 6, and 8 hours.

## Modeling approach

- CatBoost binary classifiers, one model per horizon (4h/6h/8h)
- physician identity retained as categorical role features (`physician__<id>`)
- explicit `oncall_physician_id`
- aggregate staffing counts by role
- ED flow-state variables plus short lags, deltas, and rolling means
- weather, calendar, and holiday covariates where available
- only rows where on-call is not already active are used as decision points
- final probabilities are isotonic-calibrated on the most recent 20% chronological holdout

## Outputs

- `oncall_need_probability.csv`: current raw and calibrated 4h/6h/8h probabilities
- `oncall_need_probability_validation.csv`: AUROC, average precision, event rates, and Brier scores
- `models/oncall_probability/`: CatBoost models, calibration objects, and feature metadata

## Interpretation

These probabilities estimate historical **activation behavior** under similar operational states. They should not be interpreted as a causal or normative statement that extra staffing is objectively required. That question is handled separately by the counterfactual impact model in `scripts/forecast_oncall_impact.py`.

The historical label merge currently assumes that a missing row means no activation. If label capture was incomplete during any period, the training interval should be restricted to dates with verified-complete labels before deployment.
