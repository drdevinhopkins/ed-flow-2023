#!/usr/bin/env python3
"""Benchmark native Chronos-2 vs AutoGluon Chronos-2 on one GPU.

Non-production experiment: reads the hourly ED flow source and writes only under
benchmark-results/. Each arm runs in a fresh subprocess pinned to one physical GPU.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd

FLOW_URL = "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
TARGETS = ["Total_TBS", "POD_TBS", "Vertical_TBS", "TTStr", "Overflow", "WAITINGADM"]
ARMS = ["native_production", "native_cross", "autogluon"]
MODEL = "amazon/chronos-2"
COMPONENTS = {
    "Total_TBS": ["TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS", "RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"],
    "POD_TBS": ["TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS"],
    "Vertical_TBS": ["RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"],
    "Overflow": ["POST_POD1", "TRG_HALLWAY1"],
}
ALIASES = {
    "Total_TBS": ["Total_TBS", "total_tbs", "TOTAL_TBS"],
    "POD_TBS": ["POD_TBS", "pod_tbs", "Pod_TBS"],
    "Vertical_TBS": ["Vertical_TBS", "vertical_tbs", "VERTICAL_TBS", "vert_tbs", "VERT_TBS"],
    "TTStr": ["TTStr"],
    "Overflow": ["Overflow", "overflow", "OVERFLOW"],
    "WAITINGADM": ["WAITINGADM"],
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, default=0, help="physical GPU index")
    p.add_argument("--num-cutoffs", type=int, default=8)
    p.add_argument("--spacing-hours", type=int, default=24)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--context-hours", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sample-interval", type=float, default=0.5)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--flow-url", default=FLOW_URL)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    p.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    return p.parse_args()


def resolve(raw: pd.DataFrame, target: str) -> pd.Series:
    for c in ALIASES[target]:
        if c in raw:
            return pd.to_numeric(raw[c], errors="coerce")
    parts = COMPONENTS.get(target)
    if parts and all(c in raw for c in parts):
        x = raw[parts].apply(pd.to_numeric, errors="coerce")
        return x.sum(axis=1, min_count=len(parts))
    raise ValueError(f"Could not resolve {target}")


def freeze_flow(url: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(url)
    flow = pd.DataFrame({"ds": pd.to_datetime(raw["ds"], format="mixed", errors="coerce").dt.floor("h")})
    for t in TARGETS:
        flow[t] = resolve(raw, t)
    flow = flow.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")
    idx = pd.date_range(flow.ds.min(), flow.ds.max(), freq="h", name="ds")
    flow = flow.set_index("ds").reindex(idx).reset_index()
    flow[TARGETS] = flow[TARGETS].apply(pd.to_numeric, errors="coerce").ffill()
    flow = flow.dropna(subset=TARGETS)
    flow.to_csv(path, index=False)
    return flow


def cutoffs(flow: pd.DataFrame, n: int, spacing: int, horizon: int) -> list[pd.Timestamp]:
    end = pd.Timestamp(flow.ds.max()) - pd.Timedelta(hours=horizon)
    out = [end - pd.Timedelta(hours=i * spacing) for i in range(n)]
    if out[-1] <= pd.Timestamp(flow.ds.min()):
        raise ValueError("Not enough history for requested cutoffs")
    return sorted(out)


def long_context(flow: pd.DataFrame, cutoff: pd.Timestamp, hours: int) -> pd.DataFrame:
    x = flow.loc[flow.ds.le(cutoff), ["ds", *TARGETS]].tail(hours)
    return x.melt("ds", var_name="item_id", value_name="target").sort_values(["item_id", "ds"])


def truth(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    x = flow.loc[flow.ds.gt(cutoff), ["ds", *TARGETS]].head(horizon)
    y = x.melt("ds", var_name="target_name", value_name="actual")
    y["horizon_hour"] = y.groupby("target_name").cumcount() + 1
    return y


def run_worker(a: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from chronos import BaseChronosPipeline

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker must see exactly one CUDA GPU")
    flow = pd.read_csv(a.snapshot)
    flow["ds"] = pd.to_datetime(flow.ds)
    cs = cutoffs(flow, a.num_cutoffs, a.spacing_hours, a.horizon)
    load0 = time.time()

    if a.worker == "autogluon":
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        def tsdf(x: pd.DataFrame):
            return TimeSeriesDataFrame.from_data_frame(x, id_column="item_id", timestamp_column="ds")

        ag_path = Path("/tmp") / f"ed-flow-autogluon-bench-{os.getpid()}"
        predictor = TimeSeriesPredictor(
            target="target", prediction_length=a.horizon, freq="h",
            quantile_levels=[0.1, 0.5, 0.9], path=ag_path,
            verbosity=1, log_to_file=True,
        ).fit(
            tsdf(long_context(flow, cs[0], a.context_hours)),
            hyperparameters={"Chronos2": {
                "model_path": a.model, "device": "cuda", "batch_size": a.batch_size,
                "context_length": a.context_hours, "cross_learning": True,
            }},
            skip_model_selection=True, enable_ensemble=False, verbosity=1,
        )
        predictor.persist(models="all")
        pipeline = None
        cross = True
    else:
        pipeline = BaseChronosPipeline.from_pretrained(a.model, device_map="cuda")
        predictor = None
        cross = a.worker == "native_cross"
    torch.cuda.synchronize()
    load1 = time.time()

    frames, windows, timings = [], [], []
    loop0 = time.time()
    for cutoff in cs:
        start_epoch = time.time()
        start = time.perf_counter()
        if a.worker == "native_production":
            ctx = flow.loc[flow.ds.le(cutoff), ["ds", *TARGETS]].tail(a.context_hours).copy()
            ctx["id"] = "jgh"
            pred = pipeline.predict_df(
                ctx, id_column="id", timestamp_column="ds", target=TARGETS,
                prediction_length=a.horizon, quantile_levels=[0.1, 0.5, 0.9],
                batch_size=a.batch_size, context_length=a.context_hours,
                cross_learning=False, freq="h",
            ).rename(columns={"predictions": "forecast"})[["target_name", "ds", "forecast"]]
        elif a.worker == "native_cross":
            pred = pipeline.predict_df(
                long_context(flow, cutoff, a.context_hours),
                id_column="item_id", timestamp_column="ds", target="target",
                prediction_length=a.horizon, quantile_levels=[0.1, 0.5, 0.9],
                batch_size=a.batch_size, context_length=a.context_hours,
                cross_learning=True, freq="h",
            ).rename(columns={"item_id": "target_name", "predictions": "forecast"})[["target_name", "ds", "forecast"]]
        else:
            p = predictor.predict(tsdf(long_context(flow, cutoff, a.context_hours))).reset_index()
            value = "0.5" if "0.5" in p else "mean"
            pred = p.rename(columns={"item_id": "target_name", "timestamp": "ds", value: "forecast"})[["target_name", "ds", "forecast"]]
        torch.cuda.synchronize()
        sec = time.perf_counter() - start
        end_epoch = time.time()
        pred["ds"] = pd.to_datetime(pred.ds)
        merged = pred.merge(truth(flow, cutoff, a.horizon), on=["target_name", "ds"], validate="one_to_one")
        if len(merged) != len(TARGETS) * a.horizon:
            raise RuntimeError(f"Incomplete predictions at {cutoff}: {len(merged)} rows")
        merged["cutoff"] = cutoff
        merged["arm"] = a.worker
        merged["error"] = merged.forecast - merged.actual
        merged["abs_error"] = merged.error.abs()
        merged["sq_error"] = merged.error ** 2
        frames.append(merged)
        timings.append(sec)
        windows.append([start_epoch, end_epoch])
        print(f"{a.worker}: {cutoff} {sec:.3f}s", flush=True)
    loop1 = time.time()

    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(a.worker_json.parent / f"predictions_{a.worker}.csv", index=False)
    by_target = pred.groupby("target_name").agg(mae=("abs_error", "mean"), mse=("sq_error", "mean"), abs_error=("abs_error", "sum"), abs_actual=("actual", lambda s: s.abs().sum())).reset_index()
    by_target["rmse"] = np.sqrt(by_target.mse)
    by_target["wape"] = by_target.abs_error / by_target.abs_actual
    by_target.insert(0, "arm", a.worker)
    by_target[["arm", "target_name", "mae", "rmse", "wape"]].to_csv(a.worker_json.parent / f"accuracy_{a.worker}.csv", index=False)

    inference = float(sum(timings))
    result = {
        "arm": a.worker, "cross_learning": cross, "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load1 - load0, "loop_seconds": loop1 - loop0,
        "inference_seconds": inference, "seconds_per_cutoff": inference / len(cs),
        "forecast_values_per_second": len(pred) / inference,
        "mae": float(pred.abs_error.mean()),
        "rmse": float(math.sqrt(pred.sq_error.mean())),
        "wape": float(pred.abs_error.sum() / pred.actual.abs().sum()),
        "windows": windows,
    }
    a.worker_json.write_text(json.dumps(result, indent=2))
    if a.worker == "autogluon":
        predictor.unpersist()
        shutil.rmtree(ag_path, ignore_errors=True)
    return 0


def gpu_sample(gpu: int) -> dict[str, float]:
    cmd = ["nvidia-smi", f"--id={gpu}", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    line = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[0]
    row = next(csv.reader([line]))
    return {"epoch": time.time(), "gpu_util_pct": float(row[0]), "memory_used_mib": float(row[1]), "memory_total_mib": float(row[2])}


def run_arm(a: argparse.Namespace, arm: str, snapshot: Path, out: Path) -> tuple[dict, pd.DataFrame]:
    result_path = out / f"worker_{arm}.json"
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", arm, "--snapshot", str(snapshot), "--worker-json", str(result_path), "--model", a.model, "--horizon", str(a.horizon), "--context-hours", str(a.context_hours), "--num-cutoffs", str(a.num_cutoffs), "--spacing-hours", str(a.spacing_hours), "--batch-size", str(a.batch_size)]
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    samples, stop = [], threading.Event()

    def sample_loop():
        while not stop.is_set():
            try: samples.append(gpu_sample(a.gpu))
            except Exception: pass
            stop.wait(a.sample_interval)

    with (out / f"{arm}.log").open("w") as log:
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t = threading.Thread(target=sample_loop, daemon=True); t.start()
        try:
            for line in p.stdout:
                print(line, end=""); log.write(line)
            rc = p.wait()
        finally:
            stop.set(); t.join(timeout=2)
    if rc:
        raise RuntimeError(f"{arm} failed; see {out / f'{arm}.log'}")
    telemetry = pd.DataFrame(samples)
    telemetry.to_csv(out / f"gpu_telemetry_{arm}.csv", index=False)
    result = json.loads(result_path.read_text())
    result_path.unlink(missing_ok=True)
    return result, telemetry


def summarize_gpu(result: dict, telemetry: pd.DataFrame) -> dict:
    if telemetry.empty:
        return {}
    mask = pd.Series(False, index=telemetry.index)
    for start, end in result["windows"]:
        mask |= telemetry.epoch.between(start, end)
    x = telemetry.loc[mask] if mask.any() else telemetry
    return {
        "gpu_util_mean_pct": x.gpu_util_pct.mean(),
        "gpu_util_p95_pct": x.gpu_util_pct.quantile(0.95),
        "gpu_util_max_pct": x.gpu_util_pct.max(),
        "memory_used_peak_mib": x.memory_used_mib.max(),
    }


def run_parent(a: argparse.Namespace) -> int:
    selected = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not selected or any(x not in ARMS for x in selected):
        raise ValueError(f"--arms must be a subset of {ARMS}")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found")
    out = a.output_dir or Path("benchmark-results") / "chronos2-native-vs-autogluon" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "flow_snapshot.csv"
    flow = freeze_flow(a.flow_url, snapshot)
    pd.DataFrame({"cutoff": cutoffs(flow, a.num_cutoffs, a.spacing_hours, a.horizon)}).to_csv(out / "cutoffs.csv", index=False)

    rows, accuracy = [], []
    for arm in selected:
        result, telemetry = run_arm(a, arm, snapshot, out)
        row = {k: result[k] for k in ["arm", "cross_learning", "load_seconds", "loop_seconds", "inference_seconds", "seconds_per_cutoff", "forecast_values_per_second", "mae", "rmse", "wape"]}
        row.update(summarize_gpu(result, telemetry)); rows.append(row)
        accuracy.append(pd.read_csv(out / f"accuracy_{arm}.csv"))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(accuracy, ignore_index=True).to_csv(out / "accuracy_by_target.csv", index=False)
    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nResults: {out}")
    return 0


def main() -> int:
    a = args()
    return run_worker(a) if a.worker else run_parent(a)


if __name__ == "__main__":
    raise SystemExit(main())
