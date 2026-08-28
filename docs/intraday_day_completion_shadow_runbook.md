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

## Monitoring and recovery

- Alert if `latest-status.json` is missing or older than 90 minutes during the shadow window.
- Treat `suppressed_data_quality` as a data incident to investigate, not a forecast.
- Treat `shadow_fallback` as a model incident. It records the deterministic prior-update baseline without an interval and is excluded from candidate readiness evidence.
- Confirm `prospective-readiness.json` advances only after a complete day is available.
- Preserve `forecasts.csv` and `scores.csv`; their model/day/hour keys are immutable and idempotent.
- A serialization or SHA mismatch is a hard failure. Rebuild from the frozen branch commit; do not load an unverified artifact.

Production remains a no-go until all retrospective gates stay green, at least 28 complete prospective days are scored, every 11:00–18:00 cutoff has adequate samples, safeguards have operated cleanly, and the final review explicitly approves activation.
