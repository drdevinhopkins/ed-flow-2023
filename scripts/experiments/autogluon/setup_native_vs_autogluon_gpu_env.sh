#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

ENV_DIR="${AUTOGLUON_BENCH_VENV:-$ROOT/.venv-autogluon}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

PY="$ENV_DIR/bin/python"

"$PY" -m pip install --upgrade pip setuptools wheel

# PyTorch >=2.11 CUDA 13.x wheels drop Volta/SM70 support. The CUDA 12.6
# wheels remain the supported binary path for V100-class GPUs.
"$PY" -m pip install --upgrade --no-cache-dir \
  torch==2.13.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Install AutoGluon only after the V100-compatible torch wheel is present.
# torch==2.13.0+cu126 satisfies AutoGluon 1.6.x's torch>=2.10,<2.14 range,
# preventing pip from selecting the default CUDA 13.0 wheel.
"$PY" -m pip install --upgrade \
  -r scripts/experiments/autogluon/requirements-native-vs-autogluon-gpu.txt

"$PY" - <<'PY'
import torch
import autogluon.timeseries

print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA arch list:", torch.cuda.get_arch_list())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the benchmark environment")
if "sm_70" not in torch.cuda.get_arch_list():
    raise SystemExit("This PyTorch build does not contain V100/SM70 kernels")
print("GPU 0:", torch.cuda.get_device_name(0))
print("AutoGluon TimeSeries import: OK")
PY

"$PY" -m pip check

echo
echo "Benchmark environment ready: $ENV_DIR"
echo "Run:"
echo "  bash scripts/experiments/autogluon/run_native_vs_autogluon_gpu.sh --gpu 0 --num-cutoffs 1"
