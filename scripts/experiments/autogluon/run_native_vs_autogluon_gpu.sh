#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ENV_DIR="${AUTOGLUON_BENCH_VENV:-$ROOT/.venv-autogluon}"
if [[ -x "$ENV_DIR/bin/python" ]]; then
  PY="$ENV_DIR/bin/python"
else
  PY="$(command -v python)"
fi

if ! "$PY" -c 'import autogluon.timeseries' >/dev/null 2>&1; then
  cat >&2 <<EOF
AutoGluon TimeSeries is not installed in the benchmark Python environment.
Create the isolated V100-compatible benchmark environment with:

  bash scripts/experiments/autogluon/setup_native_vs_autogluon_gpu_env.sh

This intentionally avoids changing the production .venv.
EOF
  exit 2
fi

if ! "$PY" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and "sm_70" in torch.cuda.get_arch_list() else 1)
PY
then
  cat >&2 <<EOF
The selected Python environment does not have a V100/SM70-compatible CUDA PyTorch build.
Recreate the isolated benchmark environment with:

  rm -rf "$ENV_DIR"
  bash scripts/experiments/autogluon/setup_native_vs_autogluon_gpu_env.sh
EOF
  exit 3
fi

exec "$PY" scripts/experiments/autogluon/benchmark_native_vs_autogluon_gpu.py "$@"
