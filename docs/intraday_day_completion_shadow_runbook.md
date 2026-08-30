# Intraday day-completion shadow runbook

This runner is intentionally isolated from the production forecast and publishing paths. It writes an append-only forecast ledger, data-quality status, a scored ledger, readiness metrics, and a versioned model artifact. It must remain shadow-only until the prospective readiness report passes after at least 28 complete scored days.

## One-cycle validation

From a checkout of `codex/intraday-day-completion` with its virtual environment active:

```bash
INTRADAY_SHADOW_OUTPUT_DIR=/var/lib/ed-flow/intraday-shadow \
  scripts/evaluation/prospective/run_intraday_shadow_cycle.sh
```

Confirm that `latest-status.json` is either `shadow_only` or an explicit `suppressed_data_quality`. Never bypass a suppression. Verify the artifact SHA-256 in `artifact-manifest.json` against the runtime `.joblib` file before deployment or recovery.

## Isolated scheduler installation

The templates in `automation/systemd/` run hourly from 11:10 through 18:10 Montreal time. Review the user, checkout, virtual-environment, and output paths before copying them to `/etc/systemd/system`. Create the output directory with write access for the service user. Then use `systemd-analyze verify` and one manual `systemctl start ed-flow-intraday-shadow.service` before enabling the timer.

Do not install or enable this timer in the production checkout. Do not add operational publishing, Dropbox writes, or production workflow dependencies. The service uses a non-blocking lock so overlapping weather/model runs are skipped safely.

Until that isolated timer is reviewed and installed, a timestamp-only update to `validation/intraday-day-completion-shadow/shadow-trigger.txt` may trigger the branch workflow. Update it only for the intended 11:00–18:00 Montreal cutoff collection. This mechanism changes no model code or production workflow and should be retired when the isolated timer is enabled.

## Monitoring and recovery

- Alert if `latest-status.json` is missing or older than 90 minutes during the shadow window.
- Review `monitor-health.json`; `critical` stops the cycle, while `healthy_idle` is the expected state after the 18:00 cutoff.
- Treat `suppressed_data_quality` as a data incident to investigate, not a forecast.
- Treat `shadow_fallback` as a model incident. It records the deterministic prior-update baseline without an interval and is excluded from candidate readiness evidence.
- Confirm `prospective-readiness.json` advances only after a complete day is available.
- Preserve `forecasts.csv` and `scores.csv`; their model/day/hour keys are immutable and idempotent.
- A serialization or SHA mismatch is a hard failure. Rebuild from the frozen branch commit; do not load an unverified artifact.
- Compare the versioned state and calendar/weather training-matrix fingerprints when a
  functional model fingerprint changes. A state-only change points to revised ED history;
  a weather-route change points to revised cutoff-safe weather history. Quarantine the
  forecast until the source revision is explained; never relabel it as the frozen model.
  If both training matrices are unchanged, treat the event as fitting or dependency
  nondeterminism and freeze the verified daily artifact before resuming collection.

## Frozen candidate and calibration review

Prospective observations evaluate the frozen candidate; they are not a rolling tuning
set. Do not alter the ensemble weights, point correction, interval correction, feature
routes, or model version in response to an individual day. `score-summary.json` reports
both pooled and per-day MAE, bias, baseline MAE, and P80 coverage. Its early diagnostic
may flag a bias-sign reversal, but it must recommend `collect_without_recalibration`
until at least seven complete scored days are available.

At seven days, review calibration only as a prespecified diagnostic. A proposed correction
must be evaluated on locked historical folds without using those prospective outcomes to
select its magnitude. Any accepted model or calibration change requires a new version,
a new retrospective readiness assessment, and a fresh prospective collection period;
previous forecasts remain evidence for the old version. Fallback and quarantined rows
never count toward model accuracy, interval coverage, or clean-collection gates.

Production remains a no-go until all retrospective gates stay green, at least 28 complete prospective days are scored, every 11:00–18:00 cutoff has adequate samples, the seven most recent completed days each contain all eight unquarantined cutoffs, safeguards have operated cleanly, and the final review explicitly approves activation.

`production-readiness-assessment.json` consolidates those objective gates and always leaves
`production_ready` false. Once `objective_evidence_ready` becomes true, its recommendation
changes only to `pending_manual_go_no_go`; production publishing still requires an explicit
review and a separate authorized change.
