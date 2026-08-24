#!/usr/bin/env bash
set -euo pipefail

cd /home/dhopkins/apps/ed-flow-2023

source .venv/bin/activate

# Make direct/manual invocation behave like the Dropbox watcher by loading the
# repository environment into the shell before running scripts that expect
# credentials to already be exported.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# Hourly weather routes remain disabled in production until prospective
# forecast-time weather validation is sufficient.
export CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=0

run_step() {
    printf "\n=== START: %s ===\n" "$*"
    "$@"
    printf "=== DONE: %s ===\n" "$*"
}

run_best_effort_step() {
    printf "\n=== START (best-effort): %s ===\n" "$*"
    if "$@"; then
        printf "=== DONE: %s ===\n" "$*"
    else
        rc=$?
        printf "=== WARNING: best-effort step failed (%d): %s ===\n" "$rc" "$*" >&2
    fi
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

# Refresh source data and known-future covariates.
run_step python scripts/get_current.py
# Match the GitHub hourly workflow: refresh METAR when possible, but retain the
# last good METAR history if the upstream service is temporarily unavailable.
run_best_effort_step python scripts/update_metar.py
run_step python scripts/shiftadmin.py

# Hard gate: do not forecast from stale/incomplete ED, staffing, or weather data.
run_step python scripts/validate_forecast_inputs.py

# Existing production forecast and decision-support outputs.
run_step python scripts/chronos_forecast.py
run_step python scripts/forecast_oncall_impact.py
run_step python scripts/forecast_oncall_probability.py
run_step python scripts/calculated_kpis.py
run_step python scripts/alerts.py

# Independent additive v2 output for the new Power BI report. Run this last and
# non-blocking so a v2 failure never prevents the established pipeline outputs.
run_optional_step python scripts/hourly_forecast_v2.py
