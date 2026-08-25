#!/usr/bin/env python3
"""Rolling Chronos-2 backtest of holidays, staffing, and weather covariates.

Unlike ``forecast_variable_effects.csv``, which measures how much a covariate changes
one forecast, this script measures whether the covariate improves forecast accuracy at
repeated historical cutoffs.

Important weather limitation: ``weather.csv`` is a rolling Open-Meteo table, not an
archive of the exact forecast snapshot available at each historical cutoff. Weather
scenarios therefore measure weather signal/potential and may be optimistic. They should
not be described as leakage-free real-time weather validation until forecast snapshots
are archived prospectively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from forecast_oncall_impact import (
    add_holiday_flags,
    build_staffing_features,
    derive_flow_metrics,
)

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
SHIFT_URL = (
    "https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/"
    "all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1"
)
WEATHER_URL = (
    "https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/"
    "weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1"
)
MODEL_ID = "amazon/chronos-2"
FLOW_TARGETS = [
    "Total_TBS",
    "POD_TBS",
    "Vertical_TBS",
    "TTStr",
    "Overflow",
    "WAITINGADM",
    "TRG_HALLWAY1",
    "TRG_HALLWAY_TBS",
]
SCENARIOS = ["baseline", "holidays", "staffing", "weather", "all_covariates"]


def parse_ds(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("America/Montreal").dt.tz_localize(None)
    return parsed.dt.floor("h")


def load_flow() -> pd.DataFrame:
    """Load hourly data and construct the eight operational forecast targets."""
    raw = pd.read_csv(FLOW_URL)
    raw["ds"] = parse_ds(raw["ds"])
    raw = raw.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    derived, _ = derive_flow_metrics(raw)
    source_map = {
        "Total_TBS": "total_tbs",
        "POD_TBS": "pod_tbs",
        "Vertical_TBS": "vertical_tbs",
        "Overflow": "overflow",
    }
    for target, source in source_map.items():
        if target not in derived.columns:
            if source not in derived.columns:
                raise ValueError(f"Could not derive required target {target} from {source}")
            derived[target] = pd.to_numeric(derived[source], errors="coerce")

    missing = [target for target in FLOW_TARGETS if target not in derived.columns]
    if missing:
        raise ValueError(f"Missing required flow target(s): {', '.join(missing)}")

    for target in FLOW_TARGETS:
        derived[target] = pd.to_numeric(derived[target], errors="coerce")
        if derived[target].notna().sum() == 0:
            raise ValueError(f"Target contains no numeric observations: {target}")

    index = pd.date_range(derived["ds"].min(), derived["ds"].max(), freq="h", name="ds")
    flow = derived.set_index("ds").reindex(index).reset_index()
    for target in FLOW_TARGETS:
        # Match the production hourly regularization without back-filling from future rows.
        flow[target] = flow[target].ffill()
    return flow[["ds", *FLOW_TARGETS]]


def load_staffing() -> pd.DataFrame:
    staffing = build_staffing_features(pd.read_csv(SHIFT_URL)).copy()
    staffing["ds"] = parse_ds(staffing["ds"])
    return staffing.dropna(subset=["ds"]).drop_duplicates("ds", keep="last").sort_values("ds")


def load_weather() -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_URL)
    weather["ds"] = parse_ds(weather["ds"])
    weather = weather.dropna(subset=["ds"]).drop_duplicates("ds", keep="last").sort_values("ds")
    columns = [column for column in weather.columns if column != "ds"]
    for column in columns:
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    # Keep the backtest robust to occasional sparse weather fields. This is part of the
    # explicitly documented weather signal/potential analysis, not leakage-free replay.
    weather[columns] = weather[columns].ffill().bfill()
    return weather


def select_cutoffs(
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    horizon: int,
    num_cutoffs: int,
    spacing_hours: int,
    min_history_hours: int,
) -> list[pd.Timestamp]:
    common_end = min(flow["ds"].max(), staffing["ds"].max(), weather["ds"].max())
    common_end -= pd.Timedelta(hours=horizon)
    common_start = max(flow["ds"].min(), staffing["ds"].min(), weather["ds"].min())
    common_start += pd.Timedelta(hours=min_history_hours)
    if common_end < common_start:
        raise ValueError(f"No common backtest period: {common_start} to {common_end}")

    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible historical cutoffs found")
    return sorted(cutoffs)


def normalize_numeric_covariates(
    history: pd.DataFrame, future: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = history.copy()
    future = future.copy()
    for column in future.columns:
        if column in {"id", "ds"} or column not in history.columns:
            continue
        if pd.api.types.is_numeric_dtype(history[column]) or pd.api.types.is_numeric_dtype(
            future[column]
        ):
            history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
            future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")
    return history, future


def scenario_frames(
    scenario: str,
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    history_start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    history = flow.loc[(flow["ds"] >= history_start) & (flow["ds"] <= cutoff)].copy()
    if history.empty:
        raise ValueError(f"No history available at cutoff {cutoff}")
    if history[FLOW_TARGETS].isna().any().any():
        bad = history[FLOW_TARGETS].columns[history[FLOW_TARGETS].isna().any()].tolist()
        raise ValueError(f"Missing target history at cutoff {cutoff}: {bad}")

    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})

    if scenario in {"staffing", "all_covariates"}:
        staff_columns = [column for column in staffing.columns if column != "ds"]
        history = history.merge(staffing, on="ds", how="left")
        future = future.merge(staffing, on="ds", how="left")
        for column in staff_columns:
            if column.startswith("physician__"):
                history[column] = history[column].fillna("NotWorking")
                future[column] = future[column].fillna("NotWorking")
            elif column == "oncall_physician_id":
                history[column] = history[column].fillna("None")
                future[column] = future[column].fillna("None")
            else:
                history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0)
                future[column] = pd.to_numeric(future[column], errors="coerce").fillna(0)

    if scenario in {"weather", "all_covariates"}:
        weather_columns = [column for column in weather.columns if column != "ds"]
        history = history.merge(weather, on="ds", how="left")
        future = future.merge(weather, on="ds", how="left")
        history[weather_columns] = history[weather_columns].ffill().bfill()
        if future[weather_columns].isna().any().any():
            bad = future.loc[future[weather_columns].isna().any(axis=1), "ds"].head().tolist()
            raise ValueError(f"Missing future weather values at {bad}")

    if scenario in {"holidays", "all_covariates"}:
        history = add_holiday_flags(history)
        future = add_holiday_flags(future)

    history["id"] = "jgh"
    future["id"] = "jgh"

    if scenario == "baseline":
        return history[["id", "ds", *FLOW_TARGETS]], None

    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    history, future = normalize_numeric_covariates(history, future)
    return (
        history[["id", "ds", *FLOW_TARGETS, *covariates]],
        future[["id", "ds", *covariates]],
    )


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    horizon: int,
) -> pd.DataFrame:
    kwargs = {
        "prediction_length": horizon,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": FLOW_TARGETS,
        "quantile_levels": [0.5],
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].copy()


def actuals_long(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.loc[flow["ds"].isin(hours), ["ds", *FLOW_TARGETS]].copy()
    return actual.melt(id_vars="ds", var_name="target_name", value_name="actual")


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    summary = detail.groupby(["target_name", "scenario"], as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    summary["rmse"] = np.sqrt(summary.pop("mse"))
    summary["wape"] = summary["abs_error_sum"] / summary["abs_actual_sum"].replace(0, np.nan)
    summary = summary.drop(columns=["abs_error_sum", "abs_actual_sum"])

    baseline = summary.loc[summary["scenario"] == "baseline", ["target_name", "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    summary = summary.merge(baseline, on="target_name", how="left")
    summary["mae_improvement"] = summary["baseline_mae"] - summary["mae"]
    summary["mae_improvement_pct"] = (
        summary["mae_improvement"] / summary["baseline_mae"].replace(0, np.nan) * 100
    )
    summary["weather_validation_mode"] = np.where(
        summary["scenario"].isin(["weather", "all_covariates"]),
        "historical weather values; not archived forecast snapshots",
        "not_applicable",
    )
    return summary.sort_values(["target_name", "mae"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=180)
    parser.add_argument("--min-history-hours", type=int, default=24 * 14)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.horizon, args.num_cutoffs, args.spacing_hours, args.max_history_days) < 1:
        raise ValueError("horizon, cutoffs, spacing, and history must be positive")

    flow = load_flow()
    staffing = load_staffing()
    weather = load_weather()
    cutoffs = select_cutoffs(
        flow,
        staffing,
        weather,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print(f"Scenarios: {args.scenarios}")
    if any(s in {"weather", "all_covariates"} for s in args.scenarios):
        print(
            "WARNING: weather.csv is not an archived forecast-snapshot dataset; "
            "weather results may be optimistic."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff in cutoffs:
        actual = actuals_long(flow, cutoff, args.horizon)
        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future = scenario_frames(
                scenario,
                flow,
                staffing,
                weather,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
            )
            forecast = run_forecast(pipeline, history, future, horizon=args.horizon).rename(
                columns={"predictions": "prediction"}
            )
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    if not frames:
        raise RuntimeError("Backtest produced no rows")
    detail = pd.concat(frames, ignore_index=True).dropna(subset=["prediction", "actual"])
    summary = summarize(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "covariate_ablation_backtest.csv"
    summary_path = args.output_dir / "covariate_ablation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print("MAE summary (lower is better; positive improvement beats baseline):")
    print(
        summary[
            ["target_name", "scenario", "mae", "rmse", "wape", "mae_improvement_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
