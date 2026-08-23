#!/usr/bin/env python3
"""Rolling Chronos-2 backtest of staffing, holiday, and weather covariates.

This answers a different question from forecast_variable_effects.csv. Rather than asking
"how much did a covariate change today's forecast?", this script asks "did the covariate
reduce retrospective forecast error?" across repeated historical cutoffs.

Weather caveat
--------------
The repository's weather.csv is a rolling Open-Meteo table, not a versioned archive of
what the weather forecast looked like at each historical cutoff. Therefore scenarios
using weather are *signal/potential* backtests and can be optimistic. They must not be
reported as leakage-free real-time weather-forecast validation until forecast snapshots
are archived.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

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
]
SCENARIOS = ["baseline", "holidays", "staffing", "weather", "all_covariates"]

SHIFT_TYPES = {
    "W1":"flow", "X1":"pod", "X3":"pod", "X4":"vertical", "X2":"vertical",
    "WOC1":"oncall", "WOC2":"oncall", "WOC3":"oncall", "X5":"pod", "W3":"overlap",
    "Y1":"pod", "Y3":"pod", "Y4":"vertical", "Y2":"vertical", "Y5":"pod",
    "Z1":"night", "Z2":"night", "D1":"pod", "R1":"pod", "P1":"vertical",
    "D2":"vertical", "OC1":"oncall", "OC2":"oncall", "V1":"flow", "A1":"pod",
    "G1":"vertical", "E1":"pod", "R2":"pod", "A2":"pod", "P2":"vertical",
    "E2":"vertical", "N1":"night", "N2":"night", "L2":"overlap", "L4":"overlap",
    "H1":"teaching", "B1":"vertical", "L1":"overlap", "W5":"overlap", "L6":"overlap",
    "B2":"vertical",
}


def parse_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("America/Montreal").dt.tz_localize(None)
    return parsed.dt.floor("h")


def validate_targets(flow: pd.DataFrame) -> None:
    missing = [target for target in FLOW_TARGETS if target not in flow.columns]
    if missing:
        raise ValueError(f"Missing required flow target(s): {', '.join(missing)}")
    for target in FLOW_TARGETS:
        flow[target] = pd.to_numeric(flow[target], errors="coerce")
        if flow[target].notna().sum() == 0:
            raise ValueError(f"Flow target contains no numeric values: {target}")


def add_holiday_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["ds"], errors="coerce").dt.date
    years = dates.dropna().map(lambda value: value.year)
    if years.empty:
        raise ValueError("No valid timestamps available for holiday features")
    year_range = range(int(years.min()), int(years.max()) + 1)
    qc = holidays.Canada(subdiv="QC", years=year_range, observed=True)
    il = holidays.Israel(years=year_range, observed=True)
    out["is_qc_holiday"] = dates.map(
        lambda value: "yes" if pd.notna(value) and value in qc else "no"
    )
    out["is_jewish_holiday"] = dates.map(
        lambda value: "yes" if pd.notna(value) and value in il else "no"
    )
    return out


def build_staffing_features(shifts: pd.DataFrame) -> pd.DataFrame:
    shifts = shifts.copy()
    for column in ("shift_start", "shift_end"):
        shifts[column] = parse_timestamp(shifts[column])
    shifts = shifts.dropna(subset=["shift_start", "shift_end"])
    shifts["shift_type"] = shifts["shift_short_name"].map(SHIFT_TYPES).fillna("unknown")
    shifts["physician"] = (
        shifts["first_name"].fillna("").astype(str)
        + shifts["last_name"].fillna("").astype(str)
    )

    rows: list[dict[str, object]] = []
    for row in shifts.itertuples(index=False):
        for timestamp in pd.date_range(
            row.shift_start, row.shift_end, freq="h", inclusive="left"
        ):
            rows.append(
                {
                    "ds": timestamp,
                    "physician": row.physician,
                    "shift_type": row.shift_type,
                }
            )
    if not rows:
        raise ValueError("No staffing hours could be generated from all_shifts.csv")

    expanded = pd.DataFrame(rows)
    physician = expanded.pivot_table(
        index="ds",
        columns="physician",
        values="shift_type",
        aggfunc="first",
    ).fillna("NotWorking")
    physician.columns = [f"physician__{column}" for column in physician.columns]

    counts = (
        expanded.groupby(["ds", "shift_type"]).size().unstack(fill_value=0)
    )
    counts.columns = [f"n_{column}" for column in counts.columns]

    return physician.join(counts, how="outer").reset_index().sort_values("ds")


def complete_flow_history(flow: pd.DataFrame) -> pd.DataFrame:
    flow = flow.copy()
    flow["ds"] = parse_timestamp(flow["ds"])
    flow = flow.dropna(subset=["ds"]).sort_values("ds")
    flow = flow.drop_duplicates("ds", keep="last")
    validate_targets(flow)

    index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    out = flow.set_index("ds").reindex(index).reset_index()
    for target in FLOW_TARGETS:
        # Forward fill reproduces the production regularization without allowing future
        # observations to leak backwards into a historical cutoff.
        out[target] = pd.to_numeric(out[target], errors="coerce").ffill()
    return out[["ds", *FLOW_TARGETS]]


def load_weather() -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_URL)
    weather["ds"] = parse_timestamp(weather["ds"])
    weather = weather.dropna(subset=["ds"]).drop_duplicates("ds", keep="last")
    weather = weather.sort_values("ds")
    for column in weather.columns:
        if column == "ds":
            continue
        weather[column] = pd.to_numeric(weather[column], errors="coerce")
    return weather


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
    common_end = min(
        flow["ds"].max(), staffing["ds"].max(), weather["ds"].max()
    ) - pd.Timedelta(hours=horizon)
    common_start = max(
        flow["ds"].min(), staffing["ds"].min(), weather["ds"].min()
    ) + pd.Timedelta(hours=min_history_hours)
    if common_end < common_start:
        raise ValueError(
            f"No common backtest period: start={common_start}, end={common_end}"
        )

    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible historical backtest cutoffs were found")
    return sorted(cutoffs)


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
    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})

    if scenario in {"staffing", "all_covariates"}:
        staff_columns = [column for column in staffing.columns if column != "ds"]
        history = history.merge(staffing, on="ds", how="left")
        future = future.merge(staffing, on="ds", how="left")
        physician_columns = [c for c in staff_columns if c.startswith("physician__")]
        count_columns = [c for c in staff_columns if c.startswith("n_")]
        history[physician_columns] = history[physician_columns].fillna("NotWorking")
        future[physician_columns] = future[physician_columns].fillna("NotWorking")
        history[count_columns] = history[count_columns].fillna(0)
        future[count_columns] = future[count_columns].fillna(0)

    if scenario in {"weather", "all_covariates"}:
        weather_columns = [column for column in weather.columns if column != "ds"]
        history = history.merge(weather, on="ds", how="left")
        future = future.merge(weather, on="ds", how="left")
        history[weather_columns] = history[weather_columns].ffill().bfill()
        future[weather_columns] = future[weather_columns].ffill().bfill()
        missing_future = future[weather_columns].isna().any(axis=1)
        if missing_future.any():
            bad = future.loc[missing_future, "ds"].head().tolist()
            raise ValueError(f"Weather covariates missing for future hours: {bad}")

    if scenario in {"holidays", "all_covariates"}:
        history = add_holiday_features(history)
        future = add_holiday_features(future)

    history["id"] = "jgh"
    future["id"] = "jgh"
    history, future = normalize_numeric_covariates(history, future)

    if scenario == "baseline":
        return history[["id", "ds", *FLOW_TARGETS]], None

    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
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
    kwargs = dict(
        prediction_length=horizon,
        id_column="id",
        timestamp_column="ds",
        target=FLOW_TARGETS,
        quantile_levels=[0.5],
    )
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    needed = {"ds", "target_name", "predictions"}
    missing = needed - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing columns: {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].copy()


def actuals_long(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.loc[flow["ds"].isin(future_hours), ["ds", *FLOW_TARGETS]].copy()
    return actual.melt(id_vars="ds", var_name="target_name", value_name="actual")


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    grouped = detail.groupby(["target_name", "scenario"], as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        abs_actual_sum=("abs_actual", "sum"),
        abs_error_sum=("abs_error", "sum"),
        mean_error=("error", "mean"),
    )
    grouped["rmse"] = np.sqrt(grouped.pop("mse"))
    grouped["wape"] = grouped["abs_error_sum"] / grouped["abs_actual_sum"].replace(0, np.nan)
    grouped = grouped.drop(columns=["abs_actual_sum", "abs_error_sum"])

    baseline = grouped.loc[grouped["scenario"] == "baseline", ["target_name", "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    grouped = grouped.merge(baseline, on="target_name", how="left")
    grouped["mae_improvement"] = grouped["baseline_mae"] - grouped["mae"]
    grouped["mae_improvement_pct"] = (
        grouped["mae_improvement"] / grouped["baseline_mae"].replace(0, np.nan) * 100
    )
    grouped["weather_validation_mode"] = np.where(
        grouped["scenario"].isin(["weather", "all_covariates"]),
        "historical weather values; not archived forecast snapshots",
        "not_applicable",
    )
    return grouped.sort_values(["target_name", "mae"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=180)
    parser.add_argument("--min-history-hours", type=int, default=24 * 14)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=SCENARIOS,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.num_cutoffs < 1 or args.spacing_hours < 1:
        raise ValueError("horizon, num-cutoffs, and spacing-hours must all be positive")

    flow = complete_flow_history(pd.read_csv(FLOW_URL))
    staffing = build_staffing_features(pd.read_csv(SHIFT_URL))
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
    print(f"Backtest cutoffs ({len(cutoffs)}): {cutoffs}")
    print(f"Targets: {FLOW_TARGETS}")
    print(f"Scenarios: {args.scenarios}")
    if any(s in {"weather", "all_covariates"} for s in args.scenarios):
        print(
            "WARNING: weather.csv is not an archived forecast-snapshot dataset; "
            "weather scenario results may be optimistic."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    detail_frames: list[pd.DataFrame] = []
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
            forecast = run_forecast(
                pipeline, history, future, horizon=args.horizon
            ).rename(columns={"predictions": "prediction"})
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            joined["cutoff"] = cutoff
            joined["scenario"] = scenario
            joined["horizon_hour"] = (
                (joined["ds"] - cutoff) / pd.Timedelta(hours=1)
            ).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            detail_frames.append(joined)

    if not detail_frames:
        raise RuntimeError("Backtest produced no forecast rows")
    detail = pd.concat(detail_frames, ignore_index=True).dropna(
        subset=["prediction", "actual"]
    )
    summary = summarize(detail)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.output_dir / "covariate_ablation_backtest.csv"
    summary_path = args.output_dir / "covariate_ablation_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(f"Saved detail: {detail_path}")
    print(f"Saved summary: {summary_path}")
    print("\nMAE summary (lower is better; positive improvement beats baseline):")
    print(
        summary[
            ["target_name", "scenario", "mae", "rmse", "wape", "mae_improvement_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
