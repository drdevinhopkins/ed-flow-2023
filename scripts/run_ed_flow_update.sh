#!/usr/bin/env bash
set -euo pipefail

cd /home/dhopkins/apps/ed-flow-2023

source .venv/bin/activate

# Make direct/manual invocation behave like the Dropbox watcher. A dotenv file
# is not necessarily valid Bash syntax (values may contain spaces, parentheses,
# etc.), so parse it with python-dotenv and emit shell-quoted exports instead of
# sourcing .env directly. Preserve variables already exported by systemd/the
# Dropbox watcher.
if [[ -f .env ]]; then
    dotenv_exports="$(python - <<'PY'
import os
import re
import shlex

from dotenv import dotenv_values

for key, value in dotenv_values('.env').items():
    if value is None or key in os.environ:
        continue
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key):
        raise SystemExit(f'Invalid environment variable name in .env: {key!r}')
    print(f'export {key}={shlex.quote(value)}')
PY
)"
    eval "$dotenv_exports"
    unset dotenv_exports
fi

# Reserve physical GPU 0 for the ED flow pipeline. CUDA processes launched by
# this wrapper can see only GPU 0, so Chronos and CatBoost cannot spill onto the
# GPUs reserved for Scribbler. Each model chooses GPU automatically when CUDA
# is available and otherwise falls back to CPU.
export CUDA_VISIBLE_DEVICES=0

# Hourly weather routes remain disabled in production until prospective
# forecast-time weather validation is sufficient.
export CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=0

run_step() {
    printf "\n=== START: %s ===\n" "$*"
    "$@"
    printf "=== DONE: %s ===\n" "$*"
}

run_optional_step() {
    printf "\n=== START (additive): %s ===\n" "$*"
    if "$@"; then
        printf "=== DONE: %s ===\n" "$*"
    else
        rc=$?
        printf "=== WARNING: additive step failed (%d): %s ===\n" "$rc" "$*" >&2
    fi
}

# Refresh source data and staffing. METAR refresh is intentionally not run on
# jgh000533svaps: the hospital server cannot currently reach the IEM/Mesonet
# endpoint, and neither the legacy hourly forecast nor forecast v2 consumes the
# METAR history directly. Weather covariates come from weather.csv instead.
run_step python scripts/get_current.py

# Hard gate: do not forecast from stale/incomplete ED, staffing, or weather data.
run_step python scripts/validate_forecast_inputs.py

# Existing production forecast and decision-support outputs.
run_step python scripts/chronos_forecast.py
run_step python scripts/forecast_oncall_impact.py
run_step python scripts/forecast_oncall_probability.py
# run_step python scripts/calculated_kpis.py
# run_step python scripts/alerts.py

# Independent additive v2 output for the new Power BI report. Run this last and
# non-blocking so a v2 failure never prevents the established pipeline outputs.
run_step python scripts/hourly_forecast_v2.py

run_step python scripts/shiftadmin.py
