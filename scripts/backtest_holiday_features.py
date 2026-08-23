#!/usr/bin/env python3
"""Holiday-focused native Chronos-2 ablation for JGH ED inflow.

This backtest deliberately samples forecast cutoffs whose 24-hour horizons contain
holiday effects or holiday-adjacent "shoulder" days. A generic weekly rolling backtest
can contain almost no holidays and is therefore poorly powered to choose holiday
covariates.

The primary target is Inflow_Total. Additional targets can be supplied with --targets.
No AutoGluon wrapper is used: forecasts go directly through Chronos2Pipeline.predict_df
with known future covariates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from holiday_features import add_holiday_features

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
MODEL_ID = "amazon/chronos-2"
SCENARIOS = ["baseline", "legacy", "calendars", "shoulders", "rich", "closures"]
DEFAULT_TARGETS = ["Inflow_Total"]
MAX_CONTEXT = 8192

EVENT_COLUMNS = [
    "is_any_holiday",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_long_weekend_edge",
    "is_major_jewish_holiday_eve",
    "is_christmas_newyear_period",
    "is_quebec_canada_day_period",
    "is_rebound_after_long_closure",
    "is_pre_long_closure",
]


def parse_ds(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("America/Montreal").dt.tz_localize(None)
    return parsed.dt.floor("h")


def load_flow(targets: list[str]) -> pd.DataFrame:
    raw = pd.read_csv(FLOW_URL)
    raw["ds"] = parse_ds(raw["ds"])
    raw = raw.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    missing = [target for target in targets if target not in raw.columns]
    if missing:
        raise ValueError(f"Missing target(s) in hourly dataset: {', '.join(missing)}")

    index = pd.date_range(raw["ds"].min(), raw["ds"].max(), freq="h", name="ds")
    flow = raw.set_index("ds").reindex(index).reset_index()
    for target in targets:
        flow[target] = pd.to_numeric(flow[target], errors="coerce").ffill()
        if flow[target].notna().sum() == 0:
            raise ValueError(f"Target contains no numeric observations: {target}")
    return flow[["ds", *targets]]


def _event_type(row: pd.Series) -> str:
    labels = [column.removeprefix("is_") for column in EVENT_COLUMNS if int(row.get(column, 0))]
    return "|".join(labels) if labels else "none"


def select_holiday_cutoffs(
    flow: pd.DataFrame,
    *,
    horizon: int,
    context_hours: int,
    num_cutoffs: int,
) -> pd.DataFrame:
    """Choose diverse historical days where the forecast horizon exercises holiday features."""
    first_day = flow["ds"].min().normalize()
    last_day = flow["ds"].max().normalize()
    daily = pd.DataFrame({"ds": pd.date_range(first_day, last_day, freq="D")})
    daily = add_holiday_features(daily, feature_set="closures")
    daily["is_event_day"] = daily[EVENT_COLUMNS].max(axis=1).astype(bool)
    candidates = daily.loc[daily["is_event_day"]].copy()
    candidates["cutoff"] = candidates["ds"] - pd.Timedelta(hours=1)
    candidates["event_type"] = candidates.apply(_event_type, axis=1)

    earliest = flow["ds"].min() + pd.Timedelta(hours=min(context_hours, MAX_CONTEXT) - 1)
    latest = flow["ds"].max() - pd.Timedelta(hours=horizon)
    candidates = candidates.loc[
        (candidates["cutoff"] >= earliest) & (candidates["cutoff"] <= latest)
    ].reset_index(drop=True)
    if candidates.empty:
        raise ValueError("No eligible holiday-focused cutoffs found in the available data")

    if len(candidates) > num_cutoffs:
        # Even spacing across the entire history avoids selecting only one holiday season/year.
        positions = np.linspace(0, len(candidates) - 1, num=num_cutoffs)
        indices = sorted(set(int(round(value)) for value in positions))
        candidates = candidates.iloc[indices].copy()

    return candidates[["cutoff", "ds", "event_type", *EVENT_COLUMNS]].reset_index(drop=True)


def _legacy_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the existing production holiday representation exactly."""
    featured = add_holiday_features(frame, feature_set="calendars")
    featured["is_qc_holiday"] = featured["is_qc_holiday"].map({1: "yes", 0: "no"})
    featured["is_jewish_holiday"] = featured["is_jewish_holiday"].map({1: "yes", 0: "no"})
    keep = [column for column in frame.columns] + ["is_qc_holiday", "is_jewish_holiday"]
    return featured[keep]


def build_frames(
    flow: pd.DataFrame,
    targets: list[str],
    *,
    scenario: str,
    cutoff: pd.Timestamp,
    horizon: int,
    context_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    history_start = cutoff - pd.Timedelta(hours=min(context_hours, MAX_CONTEXT) - 1)
    history = flow.loc[(flow["ds"] >= history_start) & (flow["ds"] <= cutoff)].copy()
    future_hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    future = pd.DataFrame({"ds": future_hours})

    if history.empty or len(history) < 24:
        raise ValueError(f"Insufficient history at cutoff {cutoff}")
    if history[targets].isna().any().any():
        bad = history[targets].columns[history[targets].isna().any()].tolist()
        raise ValueError(f"Missing target history at cutoff {cutoff}: {bad}")

    if scenario == "legacy":
        history = _legacy_features(history)
        future = _legacy_features(future)
    elif scenario != "baseline":
        history = add_holiday_features(history, feature_set=scenario)
        future = add_holiday_features(future, feature_set=scenario)

    history["id"] = "jgh"
    if scenario == "baseline":
        return history[["id", "ds", *targets]], None

    future["id"] = "jgh"
    covariates = [
        column
        for column in future.columns
        if column not in {"id", "ds"} and column in history.columns
    ]
    for column in covariates:
        if pd.api.types.is_numeric_dtype(history[column]) or pd.api.types.is_numeric_dtype(
            future[column]
        ):
            history[column] = pd.to_numeric(history[column], errors="coerce").astype("float64")
            future[column] = pd.to_numeric(future[column], errors="coerce").astype("float64")

    return (
        history[["id", "ds", *targets, *covariates]],
        future[["id", "ds", *covariates]],
    )


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    targets: list[str],
    *,
    horizon: int,
    context_hours: int,
) -> pd.DataFrame:
    kwargs = {
        "prediction_length": horizon,
        "id_column": "id",
        "timestamp_column": "ds",
        "target": targets,
        "quantile_levels": [0.5],
        "context_length": min(context_hours, MAX_CONTEXT, len(history)),
    }
    if future is not None:
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].rename(
        columns={"predictions": "prediction"}
    )


def actuals_long(
    flow: pd.DataFrame, targets: list[str], cutoff: pd.Timestamp, horizon: int
) -> pd.DataFrame:
    hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.loc[flow["ds"].isin(hours), ["ds", *targets]].copy()
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

    baseline = summary.loc[
        summary["scenario"] == "baseline", ["target_name", "mae"]
    ].rename(columns={"mae": "baseline_mae"})
    summary = summary.merge(baseline, on="target_name", how="left")
    summary["mae_improvement"] = summary["baseline_mae"] - summary["mae"]
    summary["mae_improvement_pct"] = (
        summary["mae_improvement"] / summary["baseline_mae"].replace(0, np.nan) * 100
    )
    return summary.sort_values(["target_name", "mae"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=8)
    parser.add_argument("--context-hours", type=int, default=MAX_CONTEXT)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output"))
    parser.add_argument("--model-id", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.num_cutoffs < 1 or args.context_hours < 24:
        raise ValueError("horizon/cutoffs must be positive and context-hours must be >= 24")

    flow = load_flow(args.targets)
    cutoffs = select_holiday_cutoffs(
        flow,
        horizon=args.horizon,
        context_hours=args.context_hours,
        num_cutoffs=args.num_cutoffs,
    )

    print(f"Targets: {', '.join(args.targets)}")
    print(f"Scenarios: {', '.join(args.scenarios)}")
    print("Holiday-focused cutoffs:")
    print(cutoffs[["cutoff", "event_type"]].to_string(index=False))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        args.model_id, device_map=device
    )

    frames: list[pd.DataFrame] = []
    for cutoff_row in cutoffs.itertuples(index=False):
        cutoff = pd.Timestamp(cutoff_row.cutoff)
        actual = actuals_long(flow, args.targets, cutoff, args.horizon)
        for scenario in args.scenarios:
            print(f"Forecasting cutoff={cutoff} scenario={scenario}")
            history, future = build_frames(
                flow,
                args.targets,
                scenario=scenario,
                cutoff=cutoff,
                horizon=args.horizon,
                context_hours=args.context_hours,
            )
            forecast = run_forecast(
                pipeline,
                history,
                future,
                args.targets,
                horizon=args.horizon,
                context_hours=args.context_hours,
            )
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            if len(joined) != args.horizon * len(args.targets):
                raise ValueError(
                    f"Expected {args.horizon * len(args.targets)} scored rows at {cutoff}, "
                    f"got {len(joined)}"
                )
            joined["cutoff"] = cutoff
            joined["event_type"] = cutoff_row.event_type
            joined["scenario"] = scenario
            joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
            joined["error"] = joined["prediction"] - joined["actual"]
            joined["abs_error"] = joined["error"].abs()
            joined["squared_error"] = joined["error"] ** 2
            joined["abs_actual"] = joined["actual"].abs()
            frames.append(joined)

    detail = pd.concat(frames, ignore_index=True)
    summary = summarize(detail)
    winners = (
        summary.sort_values(["target_name", "mae"])
        .groupby("target_name", as_index=False)
        .first()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output_dir / "holiday_feature_backtest_detail.csv", index=False)
    summary.to_csv(args.output_dir / "holiday_feature_backtest_summary.csv", index=False)
    winners.to_csv(args.output_dir / "holiday_feature_backtest_winners.csv", index=False)
    cutoffs.to_csv(args.output_dir / "holiday_feature_backtest_cutoffs.csv", index=False)

    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nWinner by target:")
    print(winners[["target_name", "scenario", "mae", "mae_improvement_pct"]].to_string(index=False))


if __name__ == "__main__":
    main()
