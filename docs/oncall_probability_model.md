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
- `models/oncall_probability/`: CatBoost models plus calibration thresholds and feature metadata

## Runtime retraining policy

The first hourly workflow run at or after 04:00 `America/Montreal` retrains the models once each morning. Runs before that local boundary reuse the prior morning's compatible artifacts. The calculation uses the IANA timezone database, so the schedule remains at 04:00 local time across daylight-saving transitions. Because the Dropbox PDF normally arrives a few minutes after the hour, retraining normally begins on that first post-04:00 watcher run rather than at exactly 04:00:00.

The workflow also retrains immediately when the cache is missing or corrupt, model hashes do not match the atomic metadata manifest, the training-spec version changes, the feature/categorical schema changes, the configured horizons change, `hourly_oncall_used_for_busy_since_2022.csv` changes, or `ED_FLOW_ONCALL_FORCE_RETRAIN=1` is set for a run. Validation metrics from the training run are retained in metadata and republished during cached inference. Cached model loading and prediction share one fallback boundary, so a loadable but incompatible artifact also triggers retraining.

The first run after deployment retrains once to create versioned cache metadata. Logs report either `On-call model cache: reused` or the explicit reason for retraining.

## Interpretation

These probabilities estimate historical **activation behavior** under similar operational states. They should not be interpreted as a causal or normative statement that extra staffing is objectively required. The impact model is also associational and cannot establish that question reliably. Operational staffing review is handled independently in `scripts/automation/compute_blurb_facts.py`; neither model can veto a workload trigger.

Missing activation labels remain unknown. Training uses explicit inactive decision points with every clock hour in all 4/6/8-hour outcome windows observed. Gaps and the trailing unlabelled interval are excluded; zero is never imputed. The live origin remains the latest joined input hour, even when its activation status is unknown. Activation-history lag features retain missing values. Changing the label handling increments the training-spec version, invalidating older cached models.

The output adds `activation_label_latest`, `current_activation_status`, and `probability_semantics`. Unknown live status must be checked operationally before interpreting a pre-activation probability. These fields do not infer activation from schedule membership.

On 2026-10-01 the committed file contained 38,287 contiguous hourly labels from 2022-01-01 through **2026-05-15 06:00**. Complete timestamps do not verify capture accuracy. The recent activation history still needs to be obtained and independently verified. The impact script publishes an explicit unavailable status when its activation history has unknown hours, rather than inventing inactive states or moving its forecast origin back to the label cutoff.

See [on-call staffing review](oncall_staffing_review.md) for provisional workload triggers, the observed-state replay, limitations and deployment checks. Independent probability calibration and causal validation of impact estimates remain open.
