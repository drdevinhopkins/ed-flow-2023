#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

if ! python -c 'import autogluon.timeseries' >/dev/null 2>&1; then
  cat >&2 <<'EOF'
AutoGluon TimeSeries is not installed in this Python environment.
Install the pinned benchmark dependency with:

  python -m pip install -r scripts/experiments/autogluon/requirements-native-vs-autogluon-gpu.txt

Then rerun this command.
EOF
  exit 2
fi

exec python scripts/experiments/autogluon/benchmark_native_vs_autogluon_gpu.py "$@"
