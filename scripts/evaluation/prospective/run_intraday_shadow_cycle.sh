#!/usr/bin/env bash
set -euo pipefail

# Isolated shadow cycle: writes only to the configured validation directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_DIR="${INTRADAY_SHADOW_OUTPUT_DIR:-$ROOT/validation/intraday-day-completion-shadow}"
RUNTIME_DIR="${INTRADAY_SHADOW_RUNTIME_DIR:-$OUTPUT_DIR/runtime}"
mkdir -p "$OUTPUT_DIR" "$RUNTIME_DIR"

exec 9>"$RUNTIME_DIR/cycle.lock"
flock -n 9 || exit 0

python "$ROOT/scripts/build_weather_history.py" \
  --start-date 2021-01-01 \
  --output "$RUNTIME_DIR/weather_backfilled.csv"
python "$ROOT/scripts/evaluation/prospective/run_shadow_intraday_day_completion.py" \
  --weather-csv "$RUNTIME_DIR/weather_backfilled.csv" \
  --output-csv "$OUTPUT_DIR/forecasts.csv" \
  --status-json "$OUTPUT_DIR/latest-status.json" \
  --status-history-csv "$OUTPUT_DIR/status-history.csv" \
  --metadata-json "$OUTPUT_DIR/model-metadata.json" \
  --artifact-joblib "$RUNTIME_DIR/intraday-ensemble-v1.joblib" \
  --artifact-manifest-json "$OUTPUT_DIR/artifact-manifest.json"
python "$ROOT/scripts/evaluation/prospective/score_shadow_intraday_day_completion.py" \
  --forecasts-csv "$OUTPUT_DIR/forecasts.csv" \
  --scores-csv "$OUTPUT_DIR/scores.csv" \
  --summary-json "$OUTPUT_DIR/score-summary.json" \
  --readiness-json "$OUTPUT_DIR/prospective-readiness.json"
health_exit=0
python "$ROOT/scripts/evaluation/prospective/check_shadow_intraday_day_completion.py" \
  --status-json "$OUTPUT_DIR/latest-status.json" \
  --status-history-csv "$OUTPUT_DIR/status-history.csv" \
  --forecasts-csv "$OUTPUT_DIR/forecasts.csv" \
  --artifact-manifest-json "$OUTPUT_DIR/artifact-manifest.json" \
  --readiness-json "$OUTPUT_DIR/prospective-readiness.json" \
  --output-json "$OUTPUT_DIR/monitor-health.json" || health_exit=$?
python "$ROOT/scripts/evaluation/prospective/build_intraday_production_readiness.py" \
  --retrospective-json "$ROOT/validation/intraday-day-completion/readiness.json" \
  --prospective-json "$OUTPUT_DIR/prospective-readiness.json" \
  --health-json "$OUTPUT_DIR/monitor-health.json" \
  --artifact-manifest-json "$OUTPUT_DIR/artifact-manifest.json" \
  --output-json "$OUTPUT_DIR/production-readiness-assessment.json"
exit "$health_exit"
