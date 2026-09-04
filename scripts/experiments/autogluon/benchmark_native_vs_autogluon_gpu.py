#!/usr/bin/env python3
"""Benchmark native Chronos-2 vs AutoGluon Chronos-2 on one GPU.

Non-production experiment. Each arm runs in a fresh subprocess pinned to one
physical GPU, uses the same frozen ED-flow snapshot/cutoffs, and writes only
under benchmark-results/.
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

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
TARGETS = ["Total_TBS", "POD_TBS", "Vertical_TBS", "TTStr", "Overflow", "WAITINGADM"]
ARMS = ["native_production", "native_cross", "autogluon"]
MODEL = "amazon/chronos-2"

COMPONENTS = {
    "Total_TBS": [
        "TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS",
        "POD_ORANGE_TBS", "RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS",
    ],
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


def parse_args() -> argparse.Namespace:
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


def resolve_target(raw: pd.DataFrame, target: str) -> pd.Series:
    for column in ALIASES[target]:
        if column in raw.columns:
            return pd.to_numeric(raw[column], errors="coerce")
    components = COMPONENTS.get(target)
    if components and all(column in raw.columns for column in components):
        numeric = raw[components].apply(pd.to_numeric, errors="coerce")
        return numeric.sum(axis=1, min_count=len(components))
    raise ValueError(f"Could not resolve target {target}")


def freeze_flow(url: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(url)
    flow = pd.DataFrame(
        {"ds": pd.to_datetime(raw["ds"], format="mixed", errors="coerce").dt.floor("h")}
    )
    for target in TARGETS:
        flow[target] = resolve_target(raw, target)
    flow = flow.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")
    index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    flow = flow.set_index("ds").reindex(index).reset_index()
    flow[TARGETS] = flow[TARGETS].apply(pd.to_numeric, errors="coerce").ffill()
    flow = flow.dropna(subset=TARGETS)
    flow.to_csv(path, index=False)
    return flow


def select_cutoffs(flow: pd.DataFrame, n: int, spacing_hours: int, horizon: int) -> list[pd.Timestamp]:
    latest = pd.Timestamp(flow["ds"].max()) - pd.Timedelta(hours=horizon)
    result = [latest - pd.Timedelta(hours=i * spacing_hours) for i in range(n)]
    if result[-1] <= pd.Timestamp(flow["ds"].min()):
        raise ValueError("Not enough history for requested cutoffs")
    return sorted(result)


def long_context(flow: pd.DataFrame, cutoff: pd.Timestamp, hours: int) -> pd.DataFrame:
    context = flow.loc[flow["ds"].le(cutoff), ["ds", *TARGETS]].tail(hours)
    return (
        context.melt("ds", var_name="item_id", value_name="target")
        .sort_values(["item_id", "ds"])
        .reset_index(drop=True)
    )


def truth(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    future = flow.loc[flow["ds"].gt(cutoff), ["ds", *TARGETS]].head(horizon)
    result = future.melt("ds", var_name="target_name", value_name="actual")
    result["horizon_hour"] = result.groupby("target_name").cumcount() + 1
    return result


def normalize_native_cross_output(pred: pd.DataFrame) -> pd.DataFrame:
    required = {"item_id", "ds", "predictions"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Unexpected native cross-learning output; missing {sorted(missing)}")
    return pred.rename(
        columns={"item_id": "target_name_ed", "predictions": "forecast"}
    )[["target_name_ed", "ds", "forecast"]].rename(
        columns={"target_name_ed": "target_name"}
    )


def run_worker(a: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from chronos import BaseChronosPipeline

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Worker must see exactly one CUDA GPU; sees {torch.cuda.device_count()}")

    flow = pd.read_csv(a.snapshot)
    flow["ds"] = pd.to_datetime(flow["ds"])
    cutoffs = select_cutoffs(flow, a.num_cutoffs, a.spacing_hours, a.horizon)

    load_start = time.perf_counter()
    ag_path: Path | None = None

    if a.worker == "autogluon":
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

        def to_tsdf(frame: pd.DataFrame) -> TimeSeriesDataFrame:
            return TimeSeriesDataFrame.from_data_frame(
                frame, id_column="item_id", timestamp_column="ds"
            )

        ag_path = Path("/tmp") / f"ed-flow-autogluon-bench-{os.getpid()}"
        predictor = TimeSeriesPredictor(
            target="target",
            prediction_length=a.horizon,
            freq="h",
            quantile_levels=[0.1, 0.5, 0.9],
            path=ag_path,
            verbosity=1,
            log_to_file=True,
        ).fit(
            to_tsdf(long_context(flow, cutoffs[0], a.context_hours)),
            hyperparameters={
                "Chronos2": {
                    "model_path": a.model,
                    "device": "cuda",
                    "batch_size": a.batch_size,
                    "context_length": a.context_hours,
                    "cross_learning": True,
                }
            },
            skip_model_selection=True,
            enable_ensemble=False,
            verbosity=1,
        )
        if hasattr(predictor, "persist"):
            predictor.persist(models="all")
        pipeline = None
        cross_learning = True
    else:
        pipeline = BaseChronosPipeline.from_pretrained(a.model, device_map="cuda")
        predictor = None
        cross_learning = a.worker == "native_cross"

    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start

    frames: list[pd.DataFrame] = []
    windows: list[list[float]] = []
    timings: list[float] = []

    for cutoff in cutoffs:
        epoch_start = time.time()
        infer_start = time.perf_counter()

        if a.worker == "native_production":
            context = flow.loc[flow["ds"].le(cutoff), ["ds", *TARGETS]].tail(a.context_hours).copy()
            context["id"] = "jgh"
            pred = pipeline.predict_df(
                context,
                id_column="id",
                timestamp_column="ds",
                target=TARGETS,
                prediction_length=a.horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                batch_size=a.batch_size,
                context_length=a.context_hours,
                cross_learning=False,
            )
            pred = pred.rename(columns={"predictions": "forecast"})[
                ["target_name", "ds", "forecast"]
            ]
        elif a.worker == "native_cross":
            raw_pred = pipeline.predict_df(
                long_context(flow, cutoff, a.context_hours),
                id_column="item_id",
                timestamp_column="ds",
                target="target",
                prediction_length=a.horizon,
                quantile_levels=[0.1, 0.5, 0.9],
                batch_size=a.batch_size,
                context_length=a.context_hours,
                cross_learning=True,
            )
            pred = normalize_native_cross_output(raw_pred)
        else:
            ag_pred = predictor.predict(to_tsdf(long_context(flow, cutoff, a.context_hours))).reset_index()
            value_column = "0.5" if "0.5" in ag_pred.columns else "mean"
            pred = ag_pred.rename(
                columns={
                    "item_id": "target_name",
                    "timestamp": "ds",
                    value_column: "forecast",
                }
            )[["target_name", "ds", "forecast"]]

        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - infer_start
        epoch_end = time.time()

        pred["ds"] = pd.to_datetime(pred["ds"])
        merged = pred.merge(
            truth(flow, cutoff, a.horizon),
            on=["target_name", "ds"],
            validate="one_to_one",
        )
        expected = len(TARGETS) * a.horizon
        if len(merged) != expected:
            raise RuntimeError(
                f"Incomplete predictions at {cutoff}: expected {expected}, got {len(merged)}"
            )

        merged["cutoff"] = cutoff
        merged["arm"] = a.worker
        merged["error"] = merged["forecast"] - merged["actual"]
        merged["abs_error"] = merged["error"].abs()
        merged["sq_error"] = merged["error"] ** 2
        frames.append(merged)
        timings.append(inference_seconds)
        windows.append([epoch_start, epoch_end])
        print(f"{a.worker}: {cutoff} {inference_seconds:.3f}s", flush=True)

    predictions = pd.concat(frames, ignore_index=True)
    out = a.worker_json.parent
    predictions.to_csv(out / f"predictions_{a.worker}.csv", index=False)

    by_target = (
        predictions.groupby("target_name")
        .agg(
            mae=("abs_error", "mean"),
            mse=("sq_error", "mean"),
            abs_error=("abs_error", "sum"),
            abs_actual=("actual", lambda s: s.abs().sum()),
        )
        .reset_index()
    )
    by_target["rmse"] = np.sqrt(by_target["mse"])
    by_target["wape"] = by_target["abs_error"] / by_target["abs_actual"]
    by_target.insert(0, "arm", a.worker)
    by_target[["arm", "target_name", "mae", "rmse", "wape"]].to_csv(
        out / f"accuracy_{a.worker}.csv", index=False
    )

    total_inference = float(sum(timings))
    result = {
        "arm": a.worker,
        "cross_learning": cross_learning,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_seconds,
        "inference_seconds": total_inference,
        "seconds_per_cutoff": total_inference / len(cutoffs),
        "forecast_values_per_second": len(predictions) / total_inference,
        "mae": float(predictions["abs_error"].mean()),
        "rmse": float(math.sqrt(predictions["sq_error"].mean())),
        "wape": float(predictions["abs_error"].sum() / predictions["actual"].abs().sum()),
        "windows": windows,
    }
    a.worker_json.write_text(json.dumps(result, indent=2))

    if a.worker == "autogluon":
        if hasattr(predictor, "unpersist"):
            predictor.unpersist()
        if ag_path is not None:
            shutil.rmtree(ag_path, ignore_errors=True)

    return 0


def gpu_sample(gpu: int) -> dict[str, float]:
    cmd = [
        "nvidia-smi",
        f"--id={gpu}",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    line = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip().splitlines()[0]
    row = next(csv.reader([line]))
    return {
        "epoch": time.time(),
        "gpu_util_pct": float(row[0]),
        "memory_used_mib": float(row[1]),
        "memory_total_mib": float(row[2]),
    }


def run_arm(a: argparse.Namespace, arm: str, snapshot: Path, out: Path) -> tuple[dict, pd.DataFrame]:
    result_path = out / f"worker_{arm}.json"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker", arm,
        "--snapshot", str(snapshot),
        "--worker-json", str(result_path),
        "--model", a.model,
        "--horizon", str(a.horizon),
        "--context-hours", str(a.context_hours),
        "--num-cutoffs", str(a.num_cutoffs),
        "--spacing-hours", str(a.spacing_hours),
        "--batch-size", str(a.batch_size),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    samples: list[dict[str, float]] = []
    stop = threading.Event()

    def sample_loop() -> None:
        while not stop.is_set():
            try:
                samples.append(gpu_sample(a.gpu))
            except Exception:
                pass
            stop.wait(a.sample_interval)

    with (out / f"{arm}.log").open("w") as log:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        thread = threading.Thread(target=sample_loop, daemon=True)
        thread.start()
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
            return_code = process.wait()
        finally:
            stop.set()
            thread.join(timeout=2)

    if return_code:
        raise RuntimeError(f"{arm} failed; see {out / f'{arm}.log'}")

    telemetry = pd.DataFrame(samples)
    telemetry.to_csv(out / f"gpu_telemetry_{arm}.csv", index=False)
    result = json.loads(result_path.read_text())
    result_path.unlink(missing_ok=True)
    return result, telemetry


def summarize_gpu(result: dict, telemetry: pd.DataFrame) -> dict[str, float]:
    if telemetry.empty:
        return {}
    mask = pd.Series(False, index=telemetry.index)
    for start, end in result["windows"]:
        mask |= telemetry["epoch"].between(start, end)
    measured = telemetry.loc[mask] if mask.any() else telemetry
    return {
        "gpu_util_mean_pct": float(measured["gpu_util_pct"].mean()),
        "gpu_util_p95_pct": float(measured["gpu_util_pct"].quantile(0.95)),
        "gpu_util_max_pct": float(measured["gpu_util_pct"].max()),
        "memory_used_peak_mib": float(measured["memory_used_mib"].max()),
    }


def run_parent(a: argparse.Namespace) -> int:
    selected = [arm.strip() for arm in a.arms.split(",") if arm.strip()]
    if not selected or any(arm not in ARMS for arm in selected):
        raise ValueError(f"--arms must be a subset of {ARMS}")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found")

    out = a.output_dir or (
        Path("benchmark-results")
        / "chronos2-native-vs-autogluon"
        / time.strftime("%Y%m%d-%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=False)

    snapshot = out / "flow_snapshot.csv"
    flow = freeze_flow(a.flow_url, snapshot)
    pd.DataFrame(
        {"cutoff": select_cutoffs(flow, a.num_cutoffs, a.spacing_hours, a.horizon)}
    ).to_csv(out / "cutoffs.csv", index=False)

    rows = []
    accuracy = []
    for arm in selected:
        result, telemetry = run_arm(a, arm, snapshot, out)
        row = {
            key: result[key]
            for key in [
                "arm",
                "cross_learning",
                "load_seconds",
                "inference_seconds",
                "seconds_per_cutoff",
                "forecast_values_per_second",
                "mae",
                "rmse",
                "wape",
            ]
        }
        row.update(summarize_gpu(result, telemetry))
        rows.append(row)
        accuracy.append(pd.read_csv(out / f"accuracy_{arm}.csv"))

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(accuracy, ignore_index=True).to_csv(out / "accuracy_by_target.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print(f"\nResults: {out}")
    return 0


def main() -> int:
    a = parse_args()
    return run_worker(a) if a.worker else run_parent(a)


if __name__ == "__main__":
    raise SystemExit(main())
