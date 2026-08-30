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
MIN_RECENT_CLEAN_COLLECTION_DAYS = 7
MIN_DAYS_BEFORE_RECALIBRATION_REVIEW = 7


def _quarantine_functional_drift(forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candidate rows into fingerprint-consistent and quarantined forecasts."""
    sort_columns = [
        column
        for column in ("generated_at_utc", "forecast_day", "cutoff_hour")
        if column in forecasts
    ]
    order = forecasts.sort_values(sort_columns).copy() if sort_columns else forecasts.copy()
    if "model_fingerprint" not in order:
        return order, order.iloc[0:0].copy()
    group = ["model_version", "training_end"]
    if "model_fingerprint_version" in order:
        group.append("model_fingerprint_version")
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
    return (
        order.loc[consistent].drop(columns="reference_fingerprint"),
        order.loc[~consistent].drop(columns="reference_fingerprint"),
    )


def score_forecasts(forecasts: pd.DataFrame, flow: pd.DataFrame) -> pd.DataFrame:
    forecasts = forecasts.copy()
    if "status" in forecasts:
        forecasts = forecasts.loc[forecasts["status"].eq("shadow_only")].copy()
    forecasts, _ = _quarantine_functional_drift(forecasts)
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
            "by_day": [],
            "early_diagnostic": {
                "minimum_days_before_recalibration_review": MIN_DAYS_BEFORE_RECALIBRATION_REVIEW,
                "days_observed": 0,
                "bias_sign_reversal_observed": False,
                "recommendation": "collect_without_recalibration",
            },
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
    by_day = (
        scored.groupby("forecast_day")
        .agg(
            n=("error", "size"),
            mae=("absolute_error", "mean"),
            bias=("error", "mean"),
            p80_coverage=("p80_covered", "mean"),
            baseline_mae=("baseline_absolute_error", "mean"),
        )
        .reset_index()
        .sort_values("forecast_day")
    )
    day_bias = by_day["bias"].astype(float)
    bias_sign_reversal = bool((day_bias.lt(0).any()) and (day_bias.gt(0).any()))
    days_observed = int(len(by_day))
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
        "by_day": by_day.to_dict(orient="records"),
        "early_diagnostic": {
            "minimum_days_before_recalibration_review": MIN_DAYS_BEFORE_RECALIBRATION_REVIEW,
            "days_observed": days_observed,
            "bias_sign_reversal_observed": bias_sign_reversal,
            "recommendation": (
                "review_calibration_without_tuning_on_prospective_data"
                if days_observed >= MIN_DAYS_BEFORE_RECALIBRATION_REVIEW
                else "collect_without_recalibration"
            ),
        },
    }


def summarize_collection_reliability(
    forecasts: pd.DataFrame,
    *,
    through_day: pd.Timestamp | str | None,
) -> dict[str, object]:
    """Require complete, unquarantined cutoff collection on seven recent completed days."""
    candidate = forecasts.copy()
    if not candidate.empty and "status" in candidate:
        candidate = candidate.loc[candidate["status"].eq("shadow_only")].copy()
    if candidate.empty or through_day is None:
        return {
            "required_recent_clean_days": MIN_RECENT_CLEAN_COLLECTION_DAYS,
            "recent_days": [],
            "clean_recent_days": 0,
            "quarantined_forecasts": 0,
            "recent_complete_clean_collection": False,
        }

    clean, quarantined = _quarantine_functional_drift(candidate)
    clean["forecast_day"] = pd.to_datetime(clean["forecast_day"]).dt.normalize()
    quarantined["forecast_day"] = pd.to_datetime(quarantined["forecast_day"]).dt.normalize()
    end = pd.Timestamp(through_day).normalize()
    recent = pd.date_range(end=end, periods=MIN_RECENT_CLEAN_COLLECTION_DAYS, freq="D")
    clean_days = 0
    details = []
    for day in recent:
        hours = set(
            clean.loc[clean["forecast_day"].eq(day), "cutoff_hour"].astype(int).tolist()
        )
        quarantined_count = int(quarantined["forecast_day"].eq(day).sum())
        complete = hours == set(OPERATIONAL_HOURS) and quarantined_count == 0
        clean_days += int(complete)
        details.append(
            {
                "day": day.date().isoformat(),
                "eligible_cutoff_count": len(hours & set(OPERATIONAL_HOURS)),
                "quarantined_forecasts": quarantined_count,
                "complete_clean_collection": complete,
            }
        )
    return {
        "required_recent_clean_days": MIN_RECENT_CLEAN_COLLECTION_DAYS,
        "recent_days": details,
        "clean_recent_days": clean_days,
        "quarantined_forecasts": int(len(quarantined)),
        "recent_complete_clean_collection": clean_days == MIN_RECENT_CLEAN_COLLECTION_DAYS,
    }


def evaluate_prospective_readiness(
    summary: dict[str, object],
    collection: dict[str, object] | None = None,
) -> dict[str, object]:
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
        "seven_recent_complete_clean_collection_days": bool(
            collection is not None and collection["recent_complete_clean_collection"]
        ),
    }
    return {
        "prospective_days": summary["prospective_days"],
        "scored_forecasts": summary["scored_forecasts"],
        "operational_hour_counts": hour_counts,
        "max_absolute_operational_hour_bias": max_hour_bias,
        "collection_reliability": collection,
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
    flow = load_hourly_flow(args.flow_csv)
    if forecasts.empty:
        scored = forecasts
    else:
        scored = score_forecasts(forecasts, flow)
    summary = summarize_scores(scored)
    complete_days = flow.loc[flow["is_complete_day"], "day"]
    through_day = complete_days.max() if not complete_days.empty else None
    collection = summarize_collection_reliability(forecasts, through_day=through_day)
    readiness = evaluate_prospective_readiness(summary, collection)
    for path in (args.scores_csv, args.summary_json, args.readiness_json):
        path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.scores_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.readiness_json.write_text(json.dumps(readiness, indent=2) + "\n")
    print(json.dumps(readiness, indent=2), flush=True)


if __name__ == "__main__":
    main()
