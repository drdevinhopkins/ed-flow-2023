#!/usr/bin/env python3
"""Interval-day research with local data inputs and no production publication.

Raw hourly endpoint timestamps remain unchanged. All generated artifacts use
separate v2 names and a new target/model version; forecasts remain experimental.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arrival_day_policy import (
    INTERVAL_MODEL_VERSION, INTERVAL_QUALITY_VERSION, INTERVAL_TARGET_VERSION,
    build_interval_daily_outputs, select_interval_issues,
)
from daily_arrival_quality import (
    build_daily_inflow_outputs, require_latest_completed_day,
    target_history_fingerprint, verify_explanation_context,
)

OUTPUT_NAMES = ["daily_inflow_interval_v2.csv", "daily_inflow_interval_quality_v2.csv",
                "target_comparison_v2.csv", "run_summary_v2.json", "daily_forecast_interval_v2.csv",
                "daily_weather_snapshot_interval_v2.csv", "daily_explained_interval_v2.csv",
                "daily_effects_interval_v2.csv", "intraday_forecast_interval_v2.csv",
                "intraday_status_interval_v2.json", "backtest_predictions_interval_v2.csv",
                "backtest_metrics_interval_v2.csv", "backtest_features_interval_v2.csv",
                "backtest_readiness_interval_v2.json", "daily_scores_interval_v2.csv",
                "daily_score_summary_interval_v2.csv", "intraday_scores_interval_v2.csv",
                "intraday_metrics_interval_v2.csv", "intraday_coverage_interval_v2.csv",
                "intraday_inventory_interval_v2.csv", "intraday_exclusions_interval_v2.csv"]


def validate_paths(args):
    """Reject remote inputs, source overwrites, and replacement of existing issues."""
    sources = [p for p in [args.hourly, args.weather_csv, args.artifacts, args.runs] if p]
    if args.archive_dir:
        sources.extend(args.archive_dir.glob("*.csv"))
    if args.zip_dir:
        sources.extend(args.zip_dir.glob("*.zip"))
    if any(not p.is_file() for p in sources):
        raise ValueError("All sources must be existing local files")
    inputs = {p.resolve() for p in sources}
    outputs = {(args.output_dir / name).resolve() for name in OUTPUT_NAMES}
    if inputs & outputs or any(p.exists() for p in outputs):
        raise ValueError("Use a fresh output directory; never overwrite inputs or existing research issues")
    if not args.now or pd.Timestamp(args.now).tzinfo is None:
        raise ValueError("A timezone-aware --now is required")
    if getattr(args, "min_history_days", 28) < 28:
        raise ValueError("The 28-day minimum cannot be lowered by this migration")
    for key in ["context_days", "max_iter", "n_folds", "test_days"]:
        if getattr(args, key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    if getattr(args, "context_days", 1095) < getattr(args, "min_history_days", 28):
        raise ValueError("context_days must cover min_history_days")


def score_interval_daily(archive, actuals):
    from evaluation.prospective.score_daily_visits_forecast import score_archive, select_earliest_issues
    if actuals.attrs.get("target_quality_version") != INTERVAL_QUALITY_VERSION:
        raise ValueError("Interval-day scoring requires v2 actuals")
    # Filter before earliest-issue selection: legacy issues never consume a new
    # version's original cutoff. Do not pool or relabel legacy forecasts.
    issues = select_earliest_issues(select_interval_issues(archive))
    return score_archive(issues, actuals)


def run_daily(args, daily):
    """Fresh Chronos inference and explanation from exactly the same v2 frames."""
    import torch
    import forecast_daily_visits as forecast
    import explain_daily_visits_forecast as explain
    from chronos import BaseChronosPipeline

    if not args.weather_csv:
        raise ValueError("daily requires local --weather-csv, including horizon weather")
    require_latest_completed_day(daily, now=args.now)
    model_daily = daily.rename(columns={"Daily_Inflow_Total": forecast.TARGET})
    cutoff, history, future = forecast.build_forecast_frames(
        model_daily, pd.read_csv(args.weather_csv), context_days=args.context_days,
        min_history_days=args.min_history_days, horizon_days=7)
    pipeline = BaseChronosPipeline.from_pretrained(
        forecast.MODEL_ID, device_map="cuda" if torch.cuda.is_available() else "cpu")
    prediction = forecast.run_daily_forecast(pipeline, history, future, horizon_days=7,
                                              context_days=args.context_days)
    formatted = forecast.format_output(prediction, future, cutoff=cutoff, history_days=len(history),
                                       generated_at=pd.Timestamp(args.now))
    formatted["target_quality_version"] = INTERVAL_QUALITY_VERSION
    formatted["target_definition_version"] = INTERVAL_TARGET_VERSION
    formatted["target_history_sha256"] = target_history_fingerprint(history)
    formatted["status"] = "experimental_interval_day_forecast"
    verify_explanation_context(formatted, history, quality_version=INTERVAL_QUALITY_VERSION)
    weather = forecast.build_weather_snapshot(future, cutoff=cutoff, generated_at=pd.Timestamp(args.now))
    effects = explain.build_explanation_rows(pipeline, history, future, formatted,
                                             horizon_days=7, context_days=args.context_days)
    enriched = explain.enrich_forecast(formatted, history, effects, baseline_history=model_daily)
    for name, frame in [("daily_forecast_interval_v2.csv", formatted),
                        ("daily_weather_snapshot_interval_v2.csv", weather),
                        ("daily_explained_interval_v2.csv", enriched),
                        ("daily_effects_interval_v2.csv", effects)]:
        frame.to_csv(args.output_dir / name, index=False)
    return {"history_days": len(history), "target_history_sha256": target_history_fingerprint(history),
            "forecast_rows": len(formatted), "explanation_rows": len(effects)}


def run_intraday(args):
    from forecast_intraday_daily_inflow import DataQualityError, build_intraday_forecast
    if not args.weather_csv:
        raise ValueError("intraday requires local --weather-csv")
    try:
        row = build_intraday_forecast(flow_source=args.hourly, weather_source=args.weather_csv,
                                     generated_at=pd.Timestamp(args.now), interval_day=True,
                                     max_iter=args.max_iter)
        pd.DataFrame([row]).to_csv(args.output_dir / "intraday_forecast_interval_v2.csv", index=False)
        status = {"status": "forecast_written", "generated_at_utc": row["generated_at_utc"], "forecast": row}
    except DataQualityError as exc:
        status = {"status": "suppressed_data_quality", "reason": str(exc),
                  "generated_at_utc": pd.Timestamp(args.now).isoformat(), "model_version": INTERVAL_MODEL_VERSION}
    (args.output_dir / "intraday_status_interval_v2.json").write_text(json.dumps(status, indent=2)+"\n")
    return status


def run_backtest(args):
    import intraday_day_completion_model as model
    flow = model.load_hourly_flow(args.hourly, interval_day=True, now=args.now)
    weather = model.build_weather_features(args.weather_csv) if args.weather_csv else None
    snapshots = model.build_snapshots(flow, calendar_mode=args.calendar_context, weather=weather)
    predictions, metrics, features = model.run_backtest(
        snapshots, cutoff_hours=list(range(6, 23)), n_folds=args.n_folds, test_days=args.test_days,
        min_train_days=365, max_iter=args.max_iter, random_state=42,
        calibration_days=56, calibration_shrinkage_days=28)
    for frame in [predictions, metrics, features]:
        frame["target_definition_version"] = INTERVAL_TARGET_VERSION
        frame["target_quality_version"] = INTERVAL_QUALITY_VERSION
    for name, frame in [("backtest_predictions_interval_v2.csv", predictions),
                        ("backtest_metrics_interval_v2.csv", metrics),
                        ("backtest_features_interval_v2.csv", features)]:
        frame.to_csv(args.output_dir / name, index=False)
    readiness = model.evaluate_readiness(predictions)
    readiness["operational_promotion"] = False
    readiness["reason"] = "Retrospective research only; new-target prospective collection required"
    (args.output_dir / "backtest_readiness_interval_v2.json").write_text(json.dumps(readiness, indent=2)+"\n")
    return {"complete_training_days": int(flow.loc[flow.is_complete_day, "day"].nunique()),
            "predictions": len(predictions), "readiness": readiness}


def run_daily_score(args, daily):
    from evaluation.prospective import score_daily_visits_forecast as score
    if not args.archive_dir:
        raise ValueError("score-daily requires --archive-dir")
    frames = [score.normalize_snapshot(pd.read_csv(p), snapshot_name=p.name)
              for p in sorted(args.archive_dir.glob("*.csv"))]
    if not frames:
        raise ValueError("No archived local forecast files")
    archive = pd.concat(frames, ignore_index=True)
    actuals = daily.rename(columns={"Daily_Inflow_Total": "actual"})
    detail = score_interval_daily(archive, actuals)
    summary = score.summarize_by_horizon(detail, quality_version=INTERVAL_QUALITY_VERSION)
    detail.to_csv(args.output_dir / "daily_scores_interval_v2.csv", index=False)
    summary.to_csv(args.output_dir / "daily_score_summary_interval_v2.csv", index=False)
    return {"archived_rows": len(archive), "v2_issue_rows": len(select_interval_issues(archive)),
            "scored_rows": len(detail), "legacy_rows_not_relabelled": len(archive)-len(select_interval_issues(archive))}


def run_intraday_score(args, hourly):
    from evaluation.prospective import audit_intraday_forecasts as audit
    if not all([args.artifacts, args.runs, args.zip_dir]):
        raise ValueError("score-intraday requires --artifacts, --runs and --zip-dir")
    forecasts, inventory, exclusions = audit.read_artifacts(json.loads(args.artifacts.read_text()),
        json.loads(args.runs.read_text()), args.zip_dir, target_definition=INTERVAL_TARGET_VERSION)
    scores, _, dropped = audit.score_forecasts(forecasts, hourly, now=args.now,
                                              daily_builder=build_interval_daily_outputs)
    start = forecasts.forecast_day.min()
    end = pd.Timestamp(args.now).tz_convert(audit.TZ).tz_localize(None).normalize()-pd.Timedelta(days=1)
    coverage = audit.coverage_calendar(scores, start, end)
    metrics, gates = audit.summarize(scores, coverage)
    for name, frame in [("intraday_scores_interval_v2.csv", scores),
                        ("intraday_metrics_interval_v2.csv", metrics),
                        ("intraday_coverage_interval_v2.csv", coverage),
                        ("intraday_inventory_interval_v2.csv", inventory),
                        ("intraday_exclusions_interval_v2.csv", pd.DataFrame(exclusions+dropped))]:
        frame.to_csv(args.output_dir/name, index=False)
    return {"scored_rows": len(scores), "gates": gates, "operational_promotion": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["targets", "daily", "intraday", "backtest", "score-daily", "score-intraday"])
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--now", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for option in ["weather-csv", "archive-dir", "artifacts", "runs", "zip-dir"]:
        parser.add_argument("--"+option, type=Path)
    parser.add_argument("--context-days", type=int, default=1095)
    parser.add_argument("--min-history-days", type=int, default=28)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--calendar-context", choices=["basic", "rich"], default="rich")
    args = parser.parse_args()
    validate_paths(args)
    hourly = pd.read_csv(args.hourly)
    daily, quality = build_interval_daily_outputs(hourly, now=args.now)
    old, _ = build_daily_inflow_outputs(hourly, now=args.now)
    comparison = old.rename(columns={"Daily_Inflow_Total": "stored_calendar_total"}).merge(
        daily.rename(columns={"Daily_Inflow_Total": "interval_day_total"}), on="ds", how="outer")
    comparison["interval_minus_stored"] = comparison.interval_day_total-comparison.stored_calendar_total
    args.output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output_dir/"daily_inflow_interval_v2.csv", index=False)
    quality.to_csv(args.output_dir/"daily_inflow_interval_quality_v2.csv", index=False)
    comparison.to_csv(args.output_dir/"target_comparison_v2.csv", index=False)
    summary = {"target_definition_version": INTERVAL_TARGET_VERSION,
               "target_quality_version": INTERVAL_QUALITY_VERSION,
               "audit_at_utc": pd.Timestamp(args.now).tz_convert("UTC").isoformat(),
               "source_sha256": hashlib.sha256(args.hourly.read_bytes()).hexdigest(),
               "mode": args.mode, "verified_days": int(daily.Daily_Inflow_Total.notna().sum()),
               "unverified_days": int(daily.Daily_Inflow_Total.isna().sum()),
               "production_changed": False, "operational_promotion": False}
    summary["configuration"] = {key: str(value) if isinstance(value, Path) else value
                                for key, value in vars(args).items()}
    summary["source_sha256s"] = {key: hashlib.sha256(value.read_bytes()).hexdigest()
                                 for key, value in [("hourly", args.hourly), ("weather", args.weather_csv),
                                                    ("artifacts", args.artifacts), ("runs", args.runs)] if value}
    if args.archive_dir:
        summary["archive_sha256s"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(args.archive_dir.glob("*.csv"))}
    if args.mode == "daily":
        summary["result"] = run_daily(args, daily)
    elif args.mode == "intraday":
        summary["result"] = run_intraday(args)
    elif args.mode == "backtest":
        summary["result"] = run_backtest(args)
    elif args.mode == "score-daily":
        summary["result"] = run_daily_score(args, daily)
    elif args.mode == "score-intraday":
        summary["result"] = run_intraday_score(args, hourly)
    (args.output_dir/"run_summary_v2.json").write_text(json.dumps(summary, indent=2, default=str)+"\n")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
