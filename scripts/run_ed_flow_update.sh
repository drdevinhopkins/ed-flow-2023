#!/usr/bin/env bash
set -euo pipefail

cd /home/dhopkins/apps/ed-flow-2023

# Use your actual venv path once fixed
source .venv/bin/activate

run_step() {
    printf "\n=== START: %s ===\n" "$*"
    "$@"
    printf "=== DONE: %s ===\n" "$*"
}

# Replace these with the exact commands your GitHub Action currently runs
run_step python scripts/get_current.py
run_step python scripts/shiftadmin.py
run_step python scripts/chronos_forecast.py
run_step python scripts/forecast_oncall_impact.py
run_step python scripts/forecast_oncall_probability.py
run_step python scripts/calculated_kpis.py
run_step python scripts/alerts.py
