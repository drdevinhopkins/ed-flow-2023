# Chronos-2 native vs AutoGluon: single-GPU benchmark

This is a manual, non-production experiment for `jgh000533svaps`. It answers two separate questions:

1. Would moving the current six-target hourly Chronos-2 call to AutoGluon improve single-GPU throughput/utilization?
2. If AutoGluon differs, is the difference caused by the AutoGluon wrapper or by AutoGluon's default Chronos-2 `cross_learning=True` behavior?

The benchmark does not upload anything to Dropbox and does not modify any production forecast files.

## Benchmark arms

The script runs each arm in a fresh Python process and pins that process to exactly one physical GPU with `CUDA_VISIBLE_DEVICES`.

- `native_production` — native `chronos-forecasting`, using one wide input with the six target columns. This mirrors the target shape used by `scripts/hourly_forecast_v2.py`; `cross_learning=False`.
- `native_cross` — native `chronos-forecasting`, with each flow target represented as its own time series and `cross_learning=True`.
- `autogluon` — AutoGluon `TimeSeriesPredictor` + `Chronos2`, using the same six-series representation, `cross_learning=True`, and the same batch/context settings as `native_cross`.

`native_production` vs `autogluon` is the practical migration comparison. `native_cross` vs `autogluon` isolates AutoGluon wrapper overhead because the underlying Chronos-2 formulation is matched.

## Targets

- `Total_TBS`
- `POD_TBS`
- `Vertical_TBS`
- `TTStr`
- `Overflow`
- `WAITINGADM`

The script freezes a local copy of the same hourly source used by the existing backtests, derives the canonical target columns when needed, and gives every arm identical cutoffs and ground truth.

## Requirements

Use an environment that already has working CUDA PyTorch for the V100 plus:

```bash
python -m pip install "chronos-forecasting>=2.0" "autogluon.timeseries>=1.6.1,<1.7"
```

If the production environment has carefully pinned PyTorch/CUDA packages, prefer a separate virtual environment for this benchmark rather than changing production dependencies.

The script also requires `nvidia-smi`.

## Run

From the repository root:

```bash
python scripts/experiments/autogluon/benchmark_native_vs_autogluon_gpu.py --gpu 0
```

Defaults:

- one physical GPU (`--gpu 0`)
- `amazon/chronos-2`
- 24-hour forecast horizon
- up to 8192 hours of context
- 8 retrospective cutoffs
- cutoffs spaced 24 hours apart
- inference batch size 256
- all three benchmark arms

For a longer utilization test:

```bash
python scripts/experiments/autogluon/benchmark_native_vs_autogluon_gpu.py \
  --gpu 0 \
  --num-cutoffs 24 \
  --spacing-hours 24
```

To use another physical V100, for example GPU 5:

```bash
python scripts/experiments/autogluon/benchmark_native_vs_autogluon_gpu.py --gpu 5
```

To repeat only the matched native-vs-AutoGluon comparison:

```bash
python scripts/experiments/autogluon/benchmark_native_vs_autogluon_gpu.py \
  --gpu 0 \
  --arms native_cross,autogluon
```

## Outputs

Each run creates a timestamped directory under:

```text
benchmark-results/chronos2-native-vs-autogluon/
```

Important files:

- `summary.csv` — inference time, throughput, GPU utilization, peak memory, MAE/RMSE/WAPE
- `accuracy_by_target.csv` — accuracy broken down by flow target
- `gpu_telemetry_<arm>.csv` — sampled `nvidia-smi` utilization and memory
- `predictions_<arm>.csv` — every forecast and matching actual value
- `<arm>.log` — stdout/stderr for each fresh worker process
- `flow_snapshot.csv` — frozen input data used for all arms
- `cutoffs.csv` — exact retrospective forecast origins

## Interpreting "uses the GPU better"

Do not use average GPU utilization alone. Compare primarily:

1. `inference_seconds` / `seconds_per_cutoff`
2. `forecast_values_per_second`
3. `gpu_util_mean_pct` and `gpu_util_p95_pct` during the actual model calls
4. `memory_used_peak_mib`
5. forecast accuracy by target

A higher utilization percentage with the same or worse throughput is not an improvement by itself.

If `native_cross` and `autogluon` are nearly identical in inference speed and GPU utilization, AutoGluon is not making the underlying Chronos-2 inference engine faster; its benefit is orchestration/defaults. If `autogluon` beats `native_production` but matches `native_cross`, the gain is mainly from the cross-learning/data-layout choice rather than the wrapper.
