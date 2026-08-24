#!/usr/bin/env python3
"""Leakage-safe Chronos-2 ablation of Hourly Report state columns.

Chronos-2 treats columns in the historical context that are not targets as past-only
covariates. This lets us test whether the ED's observed state at and before the forecast
cutoff improves the six canonical operational forecasts without supplying any realized
future Hourly Report values.

The backtest evaluates:
* baseline: the six canonical multivariate targets only;
* every eligible numeric Hourly Report column individually;
* clinically/operationally coherent feature families; and
* all eligible Hourly Report state columns together.

Date/time metadata and columns that duplicate one of the forecast targets are excluded.
All comparisons use identical rolling cutoffs and the same native amazon/chronos-2 model.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from forecast_oncall_impact import derive_flow_metrics
from staffing_features import parse_hour

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
MODEL_ID = "amazon/chronos-2"
SERIES_ID = "jgh"
FLOW_TARGETS = [
    "Total_TBS",
    "POD_TBS",
    "Vertical_TBS",
    "TTStr",
    "Overflow",
    "WAITINGADM",
]

# These are metadata or exact aliases/components already promoted to canonical targets.
# Component state columns such as POD_GREEN_TBS remain eligible: their *past* values may
# legitimately help forecast future target values.
EXCLUDED_STATE_COLUMNS = {
    "ds",
    "dateflg",
    "timeflg",
    *FLOW_TARGETS,
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "overflow",
}

FEATURE_FAMILY_COLUMNS: dict[str, set[str]] = {
    "arrivals_inflow": {
        "INFLOW_STRETCHER",
        "Infl_Stretcher_cum",
        "INFLOW_AMBULATORY",
        "Infl_Ambulatory_cum",
        "Inflow_Total",
        "Inflow_Cum_Total",
        "INFLOW_AMBULANCES",
        "Infl_Ambulances_cum",
        "reoriented_cum",
        "reoriented_cum_MD",
    },
    "admission_pressure": {
        "FLS",
        "CUM_ADMREQ",
        "CUM_BA1",
        "PSYCH_WAITINGADM",
    },
    "hallway_overflow_state": {
        "TRG_HALLWAY1",
        "TRG_HALLWAY_TBS",
        "POST_POD1",
    },
    "pod_resus_qtrack_state": {
        "QTRACK1",
        "RESUS",
        "Pod_T",
        "POD_GREEN",
        "POD_GREEN_TBS",
        "POD_YELLOW",
        "POD_YELLOW_TBS",
        "POD_ORANGE",
        "POD_ORANGE_TBS",
    },
    "vertical_raz_state": {
        "VERTSTRET",
        "RAZ_TBS",
        "RAZ_LAZYBOY",
        "RAZ_WAITINGREZ",
        "AMBVERT1",
        "AMBVERTTBS",
        "QTrack_TBS",
        "Garage_TBS",
    },
    "process_backlog": {
        "POD_CONS_MORE2H",
        "POD_IMCONS_MORE4H",
        "POD_XRAY_MORE2H",
        "POD_CT_MORE2H",
        "RAZ_CONS_MORE2H",
        "RAZ_IMCONS_MORE4H",
        "RAZ_XRAY_MORE2H",
        "RAZ_CT_MORE2H1",
    },
    "psych_state": {
        "PSYCH1",
        "PSYCH_WAITINGADM",
    },
}


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def classify_feature_family(column: str) -> str:
    matches = [name for name, columns in FEATURE_FAMILY_COLUMNS.items() if column in columns]
    return matches[0] if matches else "other_state"


def _canonical_targets(derived: pd.DataFrame) -> pd.DataFrame:
    source_map = {
        "Total_TBS": "total_tbs",
        "POD_TBS": "pod_tbs",
        "Vertical_TBS": "vertical_tbs",
        "Overflow": "overflow",
    }
    out = derived.copy()
    for target, source in source_map.items():
        if target not in out.columns:
            if source not in out.columns:
                raise ValueError(f"Could not derive required target {target} from {source}")
            out[target] = pd.to_numeric(out[source], errors="coerce")
    missing = [target for target in FLOW_TARGETS if target not in out.columns]
    if missing:
        raise ValueError(f"Missing required flow targets: {', '.join(missing)}")
    return out


def load_hourly_report(url: str = FLOW_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load/regularize targets plus raw Hourly Report columns and return an inventory."""
    raw = pd.read_csv(url)
    if "ds" not in raw.columns:
        raise ValueError("Hourly Report source is missing ds")
    raw_columns = list(raw.columns)
    raw["ds"] = parse_hour(raw["ds"])
    raw = raw.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    derived, _ = derive_flow_metrics(raw)
    derived = _canonical_targets(derived)

    candidate_columns = [c for c in raw_columns if c not in EXCLUDED_STATE_COLUMNS]
    inventory_rows: list[dict[str, object]] = []
    numeric_candidates: list[str] = []
    for column in candidate_columns:
        numeric = pd.to_numeric(derived[column], errors="coerce")
        non_null = int(numeric.notna().sum())
        unique = int(numeric.nunique(dropna=True))
        eligible = non_null > 0 and unique > 1
        inventory_rows.append(
            {
                "column": column,
                "source": "hourly_report",
                "family": classify_feature_family(column),
                "non_null_raw": non_null,
                "unique_numeric_raw": unique,
                "numeric_eligible": eligible,
                "excluded_reason": "" if eligible else "non_numeric_or_constant_or_empty",
            }
        )
        if eligible:
            derived[column] = numeric.astype("float64")
            numeric_candidates.append(column)

    for column in FLOW_TARGETS:
        derived[column] = pd.to_numeric(derived[column], errors="coerce").astype("float64")

    keep = ["ds", *FLOW_TARGETS, *numeric_candidates]
    flow = derived[keep].copy()
    index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    flow = flow.set_index("ds").reindex(index).reset_index()

    # Forward-fill only. Filling a missing historical row from a later observation would
    # leak future ED state into the cutoff, so back-fill is intentionally never used.
    for column in [*FLOW_TARGETS, *numeric_candidates]:
        flow[column] = flow[column].ffill()

    inventory = pd.DataFrame(inventory_rows)
    excluded_metadata = [c for c in raw_columns if c in EXCLUDED_STATE_COLUMNS and c != "ds"]
    if excluded_metadata:
        extra = pd.DataFrame(
            {
                "column": excluded_metadata,
                "source": "hourly_report",
                "family": "excluded",
                "non_null_raw": np.nan,
                "unique_numeric_raw": np.nan,
                "numeric_eligible": False,
                "excluded_reason": "metadata_or_target_duplicate",
            }
        )
        inventory = pd.concat([inventory, extra], ignore_index=True)
    return flow, inventory.sort_values(["numeric_eligible", "family", "column"], ascending=[False, True, True])


def select_cutoffs(
    flow: pd.DataFrame,
    *,
    horizon: int,
    num_cutoffs: int,
    spacing_hours: int,
    min_history_hours: int,
) -> list[pd.Timestamp]:
    common_start = flow["ds"].min() + pd.Timedelta(hours=min_history_hours)
    common_end = flow["ds"].max() - pd.Timedelta(hours=horizon)
    if common_end < common_start:
        raise ValueError(f"No eligible backtest window: {common_start} to {common_end}")

    indexed = flow.set_index("ds")
    cutoffs: list[pd.Timestamp] = []
    current = common_end.floor("h")
    while current >= common_start and len(cutoffs) < num_cutoffs:
        future_hours = pd.date_range(current + pd.Timedelta(hours=1), periods=horizon, freq="h")
        actual = indexed.reindex(future_hours)[FLOW_TARGETS]
        if len(actual) == horizon and not actual.isna().any().any():
            cutoffs.append(current)
        current -= pd.Timedelta(hours=spacing_hours)
    if not cutoffs:
        raise ValueError("No eligible historical cutoffs found")
    return sorted(cutoffs)


def eligible_features_for_window(
    flow: pd.DataFrame,
    inventory: pd.DataFrame,
    cutoffs: list[pd.Timestamp],
    *,
    max_history_days: int,
    min_feature_observations: int,
) -> tuple[list[str], pd.DataFrame]:
    start = min(cutoffs) - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    end = max(cutoffs)
    window = flow.loc[flow["ds"].between(start, end)]
    candidates = inventory.loc[inventory["numeric_eligible"].eq(True), "column"].tolist()
    stats: list[dict[str, object]] = []
    eligible: list[str] = []
    for column in candidates:
        if column not in window.columns:
            continue
        series = pd.to_numeric(window[column], errors="coerce")
        non_null = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))
        ok = non_null >= min_feature_observations and unique > 1
        stats.append(
            {
                "column": column,
                "window_non_null": non_null,
                "window_unique": unique,
                "window_eligible": ok,
                "window_excluded_reason": "" if ok else "insufficient_window_history_or_constant",
            }
        )
        if ok:
            eligible.append(column)
    updated = inventory.merge(pd.DataFrame(stats), on="column", how="left")
    updated["window_eligible"] = updated["window_eligible"].fillna(False).astype(bool)
    return eligible, updated


def family_feature_sets(features: Iterable[str]) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for feature in features:
        family = classify_feature_family(feature)
        output.setdefault(family, []).append(feature)
    return {family: sorted(columns) for family, columns in sorted(output.items()) if columns}


def build_scenarios(features: list[str], *, include_individual: bool = True) -> dict[str, list[str]]:
    scenarios: dict[str, list[str]] = {"baseline": []}
    if include_individual:
        for feature in sorted(features):
            scenarios[f"feature__{slug(feature)}"] = [feature]
    for family, columns in family_feature_sets(features).items():
        scenarios[f"family__{slug(family)}"] = columns
    if features:
        scenarios["all_state"] = sorted(features)
    return scenarios


def scenario_kind(name: str) -> str:
    if name == "baseline":
        return "baseline"
    if name == "all_state":
        return "all_state"
    if name.startswith("feature__"):
        return "individual"
    if name.startswith("family__"):
        return "family"
    return "other"


def scenario_history(
    flow: pd.DataFrame,
    features: list[str],
    *,
    cutoff: pd.Timestamp,
    max_history_days: int,
) -> pd.DataFrame:
    history_start = cutoff - pd.Timedelta(days=max_history_days) + pd.Timedelta(hours=1)
    columns = ["ds", *FLOW_TARGETS, *features]
    history = flow.loc[flow["ds"].between(history_start, cutoff), columns].copy()
    if history.empty:
        raise ValueError(f"No history at cutoff {cutoff}")
    if history[FLOW_TARGETS].isna().any().any():
        bad = history[FLOW_TARGETS].columns[history[FLOW_TARGETS].isna().any()].tolist()
        raise ValueError(f"Missing target history at cutoff {cutoff}: {bad}")
    for feature in features:
        history[feature] = pd.to_numeric(history[feature], errors="coerce").astype("float64")
    history["id"] = SERIES_ID
    return history[["id", "ds", *FLOW_TARGETS, *features]]


def run_forecast(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    # No future_df is passed. Non-target columns are therefore past-only covariates.
    result = pipeline.predict_df(
        history,
        prediction_length=horizon,
        id_column="id",
        timestamp_column="ds",
        target=FLOW_TARGETS,
        quantile_levels=[0.5],
    )
    required = {"ds", "target_name", "predictions"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Unexpected Chronos output; missing {sorted(missing)}")
    return result[["ds", "target_name", "predictions"]].rename(columns={"predictions": "prediction"})


def actuals_long(flow: pd.DataFrame, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hours = pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")
    actual = flow.loc[flow["ds"].isin(hours), ["ds", *FLOW_TARGETS]].copy()
    return actual.melt(id_vars="ds", var_name="target_name", value_name="actual")


def add_errors(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["error"] = out["prediction"] - out["actual"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["abs_actual"] = out["actual"].abs()
    return out


def metrics(detail: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    table = detail.groupby(group_cols, as_index=False).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        mean_error=("error", "mean"),
        abs_error_sum=("abs_error", "sum"),
        abs_actual_sum=("abs_actual", "sum"),
    )
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)
    return table


def add_baseline_comparison(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    baseline = table.loc[table["scenario"].eq("baseline"), [*keys, "mae"]].rename(
        columns={"mae": "baseline_mae"}
    )
    out = table.merge(baseline, on=keys, how="left")
    out["mae_improvement"] = out["baseline_mae"] - out["mae"]
    out["mae_improvement_pct"] = (
        out["mae_improvement"] / out["baseline_mae"].replace(0, np.nan) * 100
    )
    out["beats_baseline"] = out["mae_improvement"].gt(0)
    return out


def attach_manifest(table: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    return table.merge(manifest, on="scenario", how="left")


def top_individuals(by_band: pd.DataFrame, *, top_k: int) -> pd.DataFrame:
    individual = by_band.loc[by_band["scenario_kind"].eq("individual")].copy()
    individual = individual.sort_values(
        ["target_name", "horizon_band", "mae_improvement_pct"], ascending=[True, True, False]
    )
    individual["rank"] = individual.groupby(["target_name", "horizon_band"]).cumcount() + 1
    return individual.loc[individual["rank"] <= top_k].reset_index(drop=True)


def write_summaries(detail: pd.DataFrame, manifest: pd.DataFrame, output_dir: Path, *, top_k: int) -> None:
    detail = detail.copy()
    detail["horizon_band"] = pd.cut(
        detail["horizon_hour"],
        bins=[0, 4, 8, 12, 24],
        labels=["h01_04", "h05_08", "h09_12", "h13_24"],
        include_lowest=True,
    ).astype(str)
    detail.to_csv(output_dir / "detail.csv", index=False)

    overall = add_baseline_comparison(metrics(detail, ["target_name", "scenario"]), ["target_name"])
    overall = attach_manifest(overall, manifest).sort_values(["target_name", "mae"])
    overall.to_csv(output_dir / "summary.csv", index=False)

    by_band = add_baseline_comparison(
        metrics(detail, ["target_name", "horizon_band", "scenario"]),
        ["target_name", "horizon_band"],
    )
    by_band = attach_manifest(by_band, manifest).sort_values(["target_name", "horizon_band", "mae"])
    by_band.to_csv(output_dir / "by_horizon_band.csv", index=False)

    by_hour = add_baseline_comparison(
        metrics(detail, ["target_name", "horizon_hour", "scenario"]),
        ["target_name", "horizon_hour"],
    )
    by_hour = attach_manifest(by_hour, manifest).sort_values(["target_name", "horizon_hour", "mae"])
    by_hour.to_csv(output_dir / "by_horizon_hour.csv", index=False)

    by_band.loc[by_band["scenario_kind"].eq("individual")].to_csv(
        output_dir / "individual_features_by_horizon_band.csv", index=False
    )
    top_individuals(by_band, top_k=top_k).to_csv(
        output_dir / "top_individual_features_by_target_horizon_band.csv", index=False
    )
    by_band.loc[by_band["scenario_kind"].eq("family")].to_csv(
        output_dir / "feature_families_by_horizon_band.csv", index=False
    )
    by_band.loc[by_band["scenario_kind"].eq("all_state")].to_csv(
        output_dir / "all_state_by_horizon_band.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-cutoffs", type=int, default=6)
    parser.add_argument("--spacing-hours", type=int, default=168)
    parser.add_argument("--max-history-days", type=int, default=365)
    parser.add_argument("--min-history-hours", type=int, default=24 * 28)
    parser.add_argument("--min-feature-observations", type=int, default=24 * 28)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--flow-url", default=FLOW_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-hourly-report"))
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Load the current Hourly Report schema and write the feature inventory without loading Chronos-2.",
    )
    parser.add_argument(
        "--skip-individual",
        action="store_true",
        help="Run only family/all-state scenarios (plus baseline).",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="Optional exact Hourly Report columns to restrict the ablation to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    positive = [
        args.horizon,
        args.num_cutoffs,
        args.spacing_hours,
        args.max_history_days,
        args.min_history_hours,
        args.min_feature_observations,
        args.top_k,
        args.checkpoint_every,
    ]
    if min(positive) < 1:
        raise ValueError("Backtest sizes and thresholds must be positive")

    flow, inventory = load_hourly_report(args.flow_url)
    inventory.to_csv(args.output_dir / "feature_inventory_raw.csv", index=False)
    if args.inventory_only:
        print(inventory.to_string(index=False))
        return

    cutoffs = select_cutoffs(
        flow,
        horizon=args.horizon,
        num_cutoffs=args.num_cutoffs,
        spacing_hours=args.spacing_hours,
        min_history_hours=args.min_history_hours,
    )
    pd.DataFrame({"cutoff": cutoffs}).to_csv(args.output_dir / "cutoffs.csv", index=False)

    features, inventory = eligible_features_for_window(
        flow,
        inventory,
        cutoffs,
        max_history_days=args.max_history_days,
        min_feature_observations=args.min_feature_observations,
    )
    if args.features is not None:
        requested = list(dict.fromkeys(args.features))
        unknown = sorted(set(requested) - set(features))
        if unknown:
            raise ValueError(f"Requested features are not eligible in the backtest window: {unknown}")
        features = requested
    inventory["selected_for_run"] = inventory["column"].isin(features)
    inventory.to_csv(args.output_dir / "feature_inventory.csv", index=False)
    if not features:
        raise ValueError("No eligible Hourly Report state features found")

    scenarios = build_scenarios(features, include_individual=not args.skip_individual)
    manifest = pd.DataFrame(
        [
            {
                "scenario": name,
                "scenario_kind": scenario_kind(name),
                "feature_count": len(columns),
                "features": "|".join(columns),
            }
            for name, columns in scenarios.items()
        ]
    )
    manifest.to_csv(args.output_dir / "scenario_manifest.csv", index=False)
    (args.output_dir / "scenario_manifest.json").write_text(
        json.dumps(scenarios, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(f"Eligible Hourly Report features ({len(features)}): {', '.join(features)}")
    print(f"Scenarios ({len(scenarios)}): {', '.join(scenarios)}")
    print(f"Cutoffs ({len(cutoffs)}): {cutoffs}")
    print("Leakage guard: Hourly Report state is passed only as past-only covariates; future_df=None.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device}")
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map=device)

    frames: list[pd.DataFrame] = []
    completed = 0
    for cutoff in cutoffs:
        actual = actuals_long(flow, cutoff, args.horizon)
        for name, columns in scenarios.items():
            print(f"Forecasting cutoff={cutoff} scenario={name} feature_count={len(columns)}")
            history = scenario_history(
                flow,
                columns,
                cutoff=cutoff,
                max_history_days=args.max_history_days,
            )
            forecast = run_forecast(pipeline, history, horizon=args.horizon)
            joined = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            joined["cutoff"] = cutoff
            joined["scenario"] = name
            joined["scenario_kind"] = scenario_kind(name)
            joined["horizon_hour"] = ((joined["ds"] - cutoff) / pd.Timedelta(hours=1)).astype(int)
            frames.append(add_errors(joined))
            completed += 1
            if completed % args.checkpoint_every == 0:
                pd.concat(frames, ignore_index=True).to_csv(args.output_dir / "detail.partial.csv", index=False)

    detail = pd.concat(frames, ignore_index=True)
    write_summaries(detail, manifest, args.output_dir, top_k=args.top_k)
    if (args.output_dir / "detail.partial.csv").exists():
        (args.output_dir / "detail.partial.csv").unlink()

    top = pd.read_csv(args.output_dir / "top_individual_features_by_target_horizon_band.csv")
    print("\n=== Top individual Hourly Report state features by target/horizon ===")
    if top.empty:
        print("No individual-feature results.")
    else:
        print(
            top[["target_name", "horizon_band", "features", "mae_improvement_pct", "beats_baseline", "rank"]]
            .head(80)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
