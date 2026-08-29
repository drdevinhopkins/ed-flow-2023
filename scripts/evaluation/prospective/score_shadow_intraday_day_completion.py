#!/usr/bin/env python3
"""Score immutable intraday shadow forecasts once their Montreal day is complete."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKTEST_DIR = Path(__file__).resolve().parents[1] / "backtests"
sys.path.insert(0, str(BACKTEST_DIR))

from backtest_intraday_day_completion import FLOW_URL, load_hourly_flow  # noqa: E402

MIN_PROSPECTIVE_DAYS = 28
MIN_SAMPLES_PER_OPERATIONAL_HOUR = 20
OPERATIONAL_HOURS = tuple(range(11, 19))


def score_forecasts(forecasts: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    forecasts = forecasts.copy()
    if "status" in forecasts:
        forecasts = forecasts.loc[forecasts["status"].eq("shadow_only")].copy()
    if "model_fingerprint" in forecasts:
        order = forecasts.sort_values("generated_at_utc").copy()
        group = ["model_version", "source_hash", "training_end"]
        reference = (
            order.loc[order["model_fingerprint"].notna()]
            .groupby(group, dropna=False)["model_fingerprint"]
            .first()
            .rename("reference_fingerprint")
        )
        order = order.merge(reference, on=group, how="left")
        consistent = (
            order["model_fingerprint"].isna()
            | order["reference_fingerprint"].isna()
            | order["model_fingerprint"].eq(order["reference_fingerprint"])
        )
        forecasts = order.loc[consistent].drop(columns="reference_fingerprint")
    forecasts["forecast_day"] = pd.to_datetime(forecasts["forecast_day"]).dt.normalize()
    complete = flow.loc[flow["is_complete_day"]].copy()
    totals = complete.groupby("day", as_index=False)["Inflow_Total"].sum().rename(
        columns={"day": "forecast_day", "Inflow_Total": "actual_total"}
    )
    scored = forecasts.merge(totals, on="forecast_day", how="inner", validate="many_to_one")
    if scored.empty:
        return scored
    scored["error"] = scored["predicted_total"] - scored["actual_total"]
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"].pow(2)
    scored["p80_covered"] = scored["actual_total"].between(
        scored["p10_total"], scored["p90_total"]
    )
    scored["baseline_error"] = scored["prior_update_baseline"] - scored["actual_total"]
    scored["baseline_absolute_error"] = scored["baseline_error"].abs()
    scored["forecast_day"] = scored["forecast_day"].dt.date.astype(str)
    return scored.sort_values(["forecast_day", "cutoff_hour", "model_version"])


def summarize_scores(scored: pd.DataFrame) -> dict[str, object]:
    if scored.empty:
        return {
            "scored_forecasts": 0,
            "prospective_days": 0,
            "metrics": None,
            "by_hour": [],
        }

    error = scored["error"].astype(float)
    baseline_error = scored["baseline_error"].astype(float)
    by_hour = (
        scored.groupby("cutoff_hour")
        .agg(
            n=("error", "size"),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            p80_coverage=("p80_covered", "mean"),
            baseline_mae=("baseline_absolute_error", "mean"),
        )
        .reset_index()
    )
    return {
        "scored_forecasts": int(len(scored)),
        "prospective_days": int(scored["forecast_day"].nunique()),
        "metrics": {
            "mae": float(error.abs().mean()),
            "rmse": float(np.sqrt(error.pow(2).mean())),
            "bias": float(error.mean()),
            "p80_coverage": float(scored["p80_covered"].mean()),
            "baseline_mae": float(baseline_error.abs().mean()),
            "mae_improvement_fraction": float(
                1.0 - error.abs().mean() / baseline_error.abs().mean()
            ),
        },
        "by_hour": by_hour.to_dict(orient="records"),
    }


def evaluate_prospective_readiness(summary: dict[str, object]) -> dict[str, object]:
    metrics = summary["metrics"]
    by_hour = pd.DataFrame(summary["by_hour"])
    if by_hour.empty:
        operational = by_hour
        hour_counts = {str(hour): 0 for hour in OPERATIONAL_HOURS}
    else:
        operational = by_hour.loc[by_hour["cutoff_hour"].isin(OPERATIONAL_HOURS)]
        hour_counts = {
            str(hour): int(
                operational.loc[operational["cutoff_hour"].eq(hour), "n"].sum()
            )
            for hour in OPERATIONAL_HOURS
        }
    enough_hours = all(count >= MIN_SAMPLES_PER_OPERATIONAL_HOUR for count in hour_counts.values())
    max_hour_bias = (
        float(operational["bias"].abs().max()) if enough_hours and not operational.empty else None
    )
    gates = {
        "prospective_days_at_least_28": summary["prospective_days"] >= MIN_PROSPECTIVE_DAYS,
        "at_least_20_samples_per_operational_hour": enough_hours,
        "mae_improvement_at_least_5pct": bool(
            metrics is not None and metrics["mae_improvement_fraction"] >= 0.05
        ),
        "absolute_bias_at_most_2": bool(metrics is not None and abs(metrics["bias"]) <= 2.0),
        "p80_coverage_between_75_and_85pct": bool(
            metrics is not None and 0.75 <= metrics["p80_coverage"] <= 0.85
        ),
        "max_operational_hour_bias_at_most_3": bool(
            max_hour_bias is not None and max_hour_bias <= 3.0
        ),
    }
    return {
        "prospective_days": summary["prospective_days"],
        "scored_forecasts": summary["scored_forecasts"],
        "operational_hour_counts": hour_counts,
        "max_absolute_operational_hour_bias": max_hour_bias,
        "gates": gates,
        "prospective_ready": bool(all(gates.values())),
        "production_ready": False,
        "remaining_requirements": [
            "Prospective gates must all pass.",
            "A reviewed scheduler, serialized model artifacts, fallback, and runbook are required.",
            "An explicit go/no-go review is required before operational publishing.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-csv", default=FLOW_URL)
    parser.add_argument("--forecasts-csv", type=Path, required=True)
    parser.add_argument("--scores-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--readiness-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecasts = pd.read_csv(args.forecasts_csv) if args.forecasts_csv.exists() else pd.DataFrame()
    if forecasts.empty:
        scored = forecasts
    else:
        scored = score_forecasts(forecasts, load_hourly_flow(args.flow_csv))
    summary = summarize_scores(scored)
    readiness = evaluate_prospective_readiness(summary)
    for path in (args.scores_csv, args.summary_json, args.readiness_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scores_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.readiness_json.write_text(json.dumps(readiness, indent=2) + "\n")
    print(json.dumps(readiness, indent=2), flush=True)


if __name__ == "__main__":
    main()
