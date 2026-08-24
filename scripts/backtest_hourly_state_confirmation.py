#!/usr/bin/env python3
"""Confirm screened Hourly Report state features against current safe hourly routes.

This is the second-stage validation after ``backtest_hourly_report_features.py``.
It deliberately reuses the exact cutoffs from ``validation/hourly-final-ablation`` so
state features are compared apples-to-apples with the current calendar/staffing routes.

For each target x horizon-band cell we:
* load the top N individual state features from the screening run;
* test cumulative state-only sets (top1, top1+top2, ...);
* test the current safe route alone;
* test the current safe route + each cumulative state set; and
* run leave-one-out confirmation on the best combined route+state set when it contains
  more than one state feature.

Hourly Report state columns are always past-only covariates. They are merged into the
historical context only and are never supplied in future_df.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

import backtest_covariate_ablation as base
import backtest_hourly_final_features as final_bt
import backtest_hourly_report_features as state_bt
import backtest_hourly_weather_features as weather_bt
import backtest_staffing_features as staffing_bt
from hourly_calendar_features import add_hourly_calendar_features
from hourly_feature_routing import SAFE_ROUTES
from staffing_features import build_schedule_feature_frames

FLOW_TARGETS = list(final_bt.FLOW_TARGETS)
MODEL_ID = final_bt.MODEL_ID
HORIZON_BANDS = {
    "h01_04": (1, 4),
    "h05_08": (5, 8),
    "h09_12": (9, 12),
    "h13_24": (13, 24),
}


@dataclass(frozen=True)
class Cell:
    target: str
    band: str


def load_fixed_cutoffs(path: Path) -> list[pd.Timestamp]:
    frame = pd.read_csv(path)
    if "cutoff" not in frame.columns:
        raise ValueError(f"Missing cutoff column in {path}")
    cutoffs = pd.to_datetime(frame["cutoff"], errors="raise").tolist()
    if not cutoffs:
        raise ValueError(f"No cutoffs in {path}")
    return [pd.Timestamp(value) for value in cutoffs]


def load_screen_candidates(path: Path, top_n: int) -> dict[Cell, list[str]]:
    frame = pd.read_csv(path)
    required = {"target_name", "horizon_band", "features", "rank"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Screening table missing {sorted(missing)}")
    frame = frame.sort_values(["target_name", "horizon_band", "rank"])
    output: dict[Cell, list[str]] = {}
    for (target, band), group in frame.groupby(["target_name", "horizon_band"], sort=False):
        cell = Cell(str(target), str(band))
        values = []
        for value in group.head(top_n)["features"].astype(str):
            if value and value not in values:
                values.append(value)
        output[cell] = values
    expected = {Cell(target, band) for target in FLOW_TARGETS for band in HORIZON_BANDS}
    missing_cells = expected - set(output)
    if missing_cells:
        raise ValueError(f"Missing screened candidates for {sorted(missing_cells, key=lambda c: (c.target, c.band))}")
    return output


def cumulative_sets(features: list[str]) -> list[tuple[str, ...]]:
    return [tuple(features[:idx]) for idx in range(1, len(features) + 1)]


def add_past_state(history: pd.DataFrame, flow: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    if not features:
        return history.copy()
    missing = [feature for feature in features if feature not in flow.columns]
    if missing:
        raise ValueError(f"Missing state features in flow table: {missing}")
    state = flow[["ds", *features]].copy()
    out = history.merge(state, on="ds", how="left", validate="one_to_one")
    for feature in features:
        out[feature] = pd.to_numeric(out[feature], errors="coerce").astype("float64")
        if out[feature].isna().all():
            raise ValueError(f"State feature {feature} is entirely missing in history")
    return out


def build_route_inputs(
    route: str,
    *,
    flow: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
    shifts: pd.DataFrame,
    schedule_frames,
    calendar: pd.DataFrame,
    cutoff: pd.Timestamp,
    horizon: int,
    max_history_days: int,
    effect_min_hours: int,
    effect_shrinkage_hours: float,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return final_bt.scenario_frames(
        route,
        flow=flow,
        staffing=staffing,
        weather=weather,
        shifts=shifts,
        schedule_frames=schedule_frames,
        calendar=calendar,
        cutoff=cutoff,
        horizon=horizon,
        max_history_days=max_history_days,
        effect_min_hours=effect_min_hours,
        effect_shrinkage_hours=effect_shrinkage_hours,
    )


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future: pd.DataFrame | None,
    *,
    horizon: int,
) -> pd.DataFrame:
    return final_bt.run_forecast(pipeline, history, future, horizon=horizon)


def horizon_band_for_hour(hour: int) -> str:
    for band, (lo, hi) in HORIZON_BANDS.items():
        if lo <= hour <= hi:
            return band
    raise ValueError(hour)


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["abs_actual"] = out["actual"].abs()
    return out


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["target_name", "horizon_band", "scenario", "scenario_kind", "route", "state_features"]
    table = detail.groupby(group_cols, as_index=False, dropna=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)

    baseline = table.loc[table["scenario_kind"].eq("baseline"), ["target_name", "horizon_band", "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    current = table.loc[table["scenario_kind"].eq("current_route"), ["target_name", "horizon_band", "mae"]].rename(
        columns={"mae": "current_route_mae"}
    )
    table = table.merge(baseline, on=["target_name", "horizon_band"], how="left")
    table = table.merge(current, on=["target_name", "horizon_band"], how="left")
    table["mae_improvement_vs_baseline"] = table["baseline_mae"] - table["mae"]
    table["mae_improvement_pct_vs_baseline"] = (
        table["mae_improvement_vs_baseline"] / table["baseline_mae"].replace(0, np.nan) * 100
    )
    table["mae_improvement_vs_current_route"] = table["current_route_mae"] - table["mae"]
    table["mae_improvement_pct_vs_current_route"] = (
        table["mae_improvement_vs_current_route"] / table["current_route_mae"].replace(0, np.nan) * 100
    )
    table["beats_current_route"] = table["mae_improvement_vs_current_route"].gt(0)
    return table.sort_values(["target_name", "horizon_band", "mae"])


def collect_cell_rows(
    *,
    forecast: pd.DataFrame,
    actual: pd.DataFrame,
    cutoff: pd.Timestamp,
    cell: Cell,
    scenario: str,
    scenario_kind: str,
    route: str,
    state_features: tuple[str, ...],
) -> pd.DataFrame:
    lo, hi = HORIZON_BANDS[cell.band]
    joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
    joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
    joined = joined.loc[
        joined["target_name"].eq(cell.target) & joined["horizon_hour"].between(lo, hi)
    ].copy()
    joined["cutoff"] = cutoff
    joined["horizon_band"] = cell.band
    joined["scenario"] = scenario
    joined["scenario_kind"] = scenario_kind
    joined["route"] = route
    joined["state_features"] = "|".join(state_features)
    return add_errors(joined)


def scenario_label(kind: str, route: str, features: tuple[str, ...]) -> str:
    if kind == "baseline":
        return "baseline"
    if kind == "current_route":
        return f"current_route__{route}"
    suffix = "+".join(features) if features else "none"
    if kind == "state_only":
        return f"state__{suffix}"
    if kind == "route_plus_state":
        return f"{route}__plus__{suffix}"
    if kind == "leave_one_out":
        return f"{route}__loo__{suffix}"
    return f"{kind}__{route}__{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--effect-min-hours", type=int, default=24)
    parser.add_argument("--effect-shrinkage-hours", type=float, default=72.0)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--cutoffs-file",
        type=Path,
        default=Path("validation/hourly-final-ablation/cutoffs.csv"),
    )
    parser.add_argument(
        "--screening-file",
        type=Path,
        default=Path("validation/hourly-report-state-ablation/top_individual_features_by_target_horizon_band.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation-output-hourly-state-confirmation")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.horizon != 24:
        raise ValueError("This confirmation is defined for the validated 24-hour routing horizon")
    if args.top_n < 1:
        raise ValueError("top-n must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = load_fixed_cutoffs(args.cutoffs_file)
    candidates = load_screen_candidates(args.screening_file, args.top_n)

    flow, inventory = state_bt.load_hourly_report()
    shifts = staffing_bt.load_shifts()
    schedule_frames = build_schedule_feature_frames(shifts)
    staffing = base.load_staffing()
    weather = weather_bt.load_weather(base.WEATHER_URL)
    calendar = add_hourly_calendar_features(
        pd.DataFrame({
            "ds": pd.date_range(flow["ds"].min().floor("h"), max(cutoffs) + pd.Timedelta(hours=args.horizon), freq="h")
        })
    )

    needed_features = sorted({feature for values in candidates.values() for feature in values})
    missing_features = [feature for feature in needed_features if feature not in flow.columns]
    if missing_features:
        raise ValueError(f"Screened features missing from current Hourly Report: {missing_features}")

    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "cutoffs.csv", index=False)
    inventory.to_csv(args.output_dir / "feature_inventory.csv", index=False)
    manifest_rows = []
    for cell, features in sorted(candidates.items(), key=lambda item: (item[0].target, item[0].band)):
        manifest_rows.append(
            {
                "target_name": cell.target,
                "horizon_band": cell.band,
                "current_route": SAFE_ROUTES[cell.target][cell.band],
                "screened_features": "|".join(features),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(args.output_dir / "candidate_manifest.csv", index=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    print(f"Fixed cutoffs ({len(cutoffs)}): {cutoffs}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)

    cells = sorted(candidates, key=lambda c: (c.target, c.band))
    detail_frames: list[pd.DataFrame] = []

    # Phase 1: current routes versus cumulative screened state sets and combinations.
    for cutoff in cutoffs:
        print(f"Phase 1 cutoff={cutoff}")
        actual = base.actuals_long(flow, cutoff, args.horizon)
        forecast_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}

        def get_forecast(route: str, features: tuple[str, ...]) -> pd.DataFrame:
            key = (route, features)
            if key in forecast_cache:
                return forecast_cache[key]
            history, future = build_route_inputs(
                route,
                flow=flow,
                staffing=staffing,
                weather=weather,
                shifts=shifts,
                schedule_frames=schedule_frames,
                calendar=calendar,
                cutoff=cutoff,
                horizon=args.horizon,
                max_history_days=args.max_history_days,
                effect_min_hours=args.effect_min_hours,
                effect_shrinkage_hours=args.effect_shrinkage_hours,
            )
            history = add_past_state(history, flow, features)
            result = run_forecast(pipeline, history, future, horizon=args.horizon)
            forecast_cache[key] = result
            return result

        baseline_forecast = get_forecast("baseline", ())
        for cell in cells:
            route = SAFE_ROUTES[cell.target][cell.band]
            detail_frames.append(
                collect_cell_rows(
                    forecast=baseline_forecast,
                    actual=actual,
                    cutoff=cutoff,
                    cell=cell,
                    scenario="baseline",
                    scenario_kind="baseline",
                    route="baseline",
                    state_features=(),
                )
            )
            current_forecast = get_forecast(route, ())
            detail_frames.append(
                collect_cell_rows(
                    forecast=current_forecast,
                    actual=actual,
                    cutoff=cutoff,
                    cell=cell,
                    scenario=scenario_label("current_route", route, ()),
                    scenario_kind="current_route",
                    route=route,
                    state_features=(),
                )
            )
            for feature_set in cumulative_sets(candidates[cell]):
                state_forecast = get_forecast("baseline", feature_set)
                detail_frames.append(
                    collect_cell_rows(
                        forecast=state_forecast,
                        actual=actual,
                        cutoff=cutoff,
                        cell=cell,
                        scenario=scenario_label("state_only", "baseline", feature_set),
                        scenario_kind="state_only",
                        route="baseline",
                        state_features=feature_set,
                    )
                )
                combo_forecast = get_forecast(route, feature_set)
                detail_frames.append(
                    collect_cell_rows(
                        forecast=combo_forecast,
                        actual=actual,
                        cutoff=cutoff,
                        cell=cell,
                        scenario=scenario_label("route_plus_state", route, feature_set),
                        scenario_kind="route_plus_state",
                        route=route,
                        state_features=feature_set,
                    )
                )

        pd.concat(detail_frames, ignore_index=True).to_csv(args.output_dir / "detail.phase1.partial.csv", index=False)

    phase1_detail = pd.concat(detail_frames, ignore_index=True)
    phase1_summary = summarize(phase1_detail)
    phase1_detail.to_csv(args.output_dir / "detail.phase1.csv", index=False)
    phase1_summary.to_csv(args.output_dir / "confirmation_by_target_horizon.csv", index=False)

    # Choose the best current-route-or-combined scenario per cell. State-only is reported
    # but cannot be promoted over an already validated route without this direct comparison.
    promotable = phase1_summary.loc[
        phase1_summary["scenario_kind"].isin(["current_route", "route_plus_state"])
    ].copy()
    winner_idx = promotable.groupby(["target_name", "horizon_band"])["mae"].idxmin()
    winners = promotable.loc[winner_idx].sort_values(["target_name", "horizon_band"]).reset_index(drop=True)
    winners.to_csv(args.output_dir / "phase1_winners.csv", index=False)

    # Phase 2: leave-one-out only for cells where a multi-state combined route wins.
    loo_requests: dict[Cell, list[tuple[str, ...]]] = {}
    for row in winners.itertuples(index=False):
        if row.scenario_kind != "route_plus_state":
            continue
        features = tuple(filter(None, str(row.state_features).split("|")))
        if len(features) <= 1:
            continue
        cell = Cell(str(row.target_name), str(row.horizon_band))
        loo_requests[cell] = [features[:idx] + features[idx + 1 :] for idx in range(len(features))]

    loo_frames: list[pd.DataFrame] = []
    if loo_requests:
        for cutoff in cutoffs:
            print(f"Phase 2 leave-one-out cutoff={cutoff}")
            actual = base.actuals_long(flow, cutoff, args.horizon)
            cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
            for cell, feature_sets in loo_requests.items():
                route = SAFE_ROUTES[cell.target][cell.band]
                for feature_set in feature_sets:
                    key = (route, feature_set)
                    if key not in cache:
                        history, future = build_route_inputs(
                            route,
                            flow=flow,
                            staffing=staffing,
                            weather=weather,
                            shifts=shifts,
                            schedule_frames=schedule_frames,
                            calendar=calendar,
                            cutoff=cutoff,
                            horizon=args.horizon,
                            max_history_days=args.max_history_days,
                            effect_min_hours=args.effect_min_hours,
                            effect_shrinkage_hours=args.effect_shrinkage_hours,
                        )
                        history = add_past_state(history, flow, feature_set)
                        cache[key] = run_forecast(pipeline, history, future, horizon=args.horizon)
                    loo_frames.append(
                        collect_cell_rows(
                            forecast=cache[key],
                            actual=actual,
                            cutoff=cutoff,
                            cell=cell,
                            scenario=scenario_label("leave_one_out", route, feature_set),
                            scenario_kind="leave_one_out",
                            route=route,
                            state_features=feature_set,
                        )
                    )

    if loo_frames:
        loo_detail = pd.concat(loo_frames, ignore_index=True)
        # Add baseline/current rows solely so summarize can compute reference columns.
        reference = phase1_detail.loc[
            phase1_detail["scenario_kind"].isin(["baseline", "current_route"])
            & phase1_detail.apply(lambda row: Cell(str(row["target_name"]), str(row["horizon_band"])) in loo_requests, axis=1)
        ].copy()
        loo_summary = summarize(pd.concat([reference, loo_detail], ignore_index=True))
        loo_summary = loo_summary.loc[loo_summary["scenario_kind"].eq("leave_one_out")].copy()
        loo_detail.to_csv(args.output_dir / "leave_one_out_detail.csv", index=False)
        loo_summary.to_csv(args.output_dir / "leave_one_out.csv", index=False)
    else:
        pd.DataFrame().to_csv(args.output_dir / "leave_one_out.csv", index=False)

    print("\n=== Phase 1 promotable winners ===")
    print(
        winners[
            [
                "target_name",
                "horizon_band",
                "route",
                "state_features",
                "mae",
                "mae_improvement_pct_vs_current_route",
                "scenario_kind",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
