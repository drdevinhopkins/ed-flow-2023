#!/usr/bin/env bash
set -euo pipefail

cd /home/dhopkins/apps/ed-flow-2023

# Use your actual venv path once fixed
source .venv/bin/activate

# Replace these with the exact commands your GitHub Action currently runs
python scripts/get_current.py
python scripts/shiftadmin.py
python scripts/chronos_forecast.py
python scripts/calculated_kpis.py
python scripts/alerts.py