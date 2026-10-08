#!/usr/bin/env python3
"""Read-only scoring of original intraday GitHub artifacts; never refit forecasts.

Score the model's stored-ds calendar target. Report the interval-end/cumulative
counter discrepancy separately; neither silently relabel nor change production.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from daily_arrival_quality import build_daily_inflow_outputs

TZ = "America/Montreal"
NUMBERS = ["observed_arrivals", "predicted_total", "p10_total", "p90_total",
           "expected_additional_arrivals", "prior_update_baseline"]
OUTPUTS = ["summary.json", "scores.csv", "metrics.csv", "coverage.csv",
           "artifact_inventory.csv", "exclusions.csv", "day_boundary.csv"]


def validate_forecast(row, created_at):
    """Reject malformed, late or inconsistent issues before selecting originals."""
    cutoff = pd.Timestamp(row["cutoff_ds_local"])
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert(TZ).tz_localize(None)
    day = pd.Timestamp(row["forecast_day"])
    generated = pd.Timestamp(row["generated_at_utc"])
    created = pd.Timestamp(created_at)
    if generated.tzinfo is None or created.tzinfo is None:
        raise ValueError("issue timestamps must be timezone-aware")
    if day.tzinfo is not None or day != day.normalize() or cutoff != cutoff.floor("h"):
        raise ValueError("invalid local calendar/cutoff")
    hour = float(row["cutoff_hour"])
    if hour != cutoff.hour or cutoff.normalize() != day or not 6 <= hour <= 22:
        raise ValueError("cutoff and forecast day disagree")
    lag = (generated.tz_convert(TZ).tz_localize(None) - cutoff).total_seconds() / 60
    if not 0 <= lag <= 90:
        raise ValueError("issue outside the original 90-minute freshness limit")
    if created < generated or created.tz_convert(TZ).date() != day.date():
        raise ValueError("artifact was not retained before the outcome day ended")
    values = {key: float(row[key]) for key in NUMBERS}
    if not all(np.isfinite(v) and v >= 0 for v in values.values()):
        raise ValueError("nonfinite or negative forecast values")
    if not (values["observed_arrivals"] <= values["p10_total"]
            <= values["predicted_total"] <= values["p90_total"]):
        raise ValueError("forecast interval invariant failed")
    if not np.isclose(values["expected_additional_arrivals"],
                      values["predicted_total"] - values["observed_arrivals"], atol=1e-8, rtol=0):
        raise ValueError("remaining arrivals disagree")
    if values["prior_update_baseline"] < values["observed_arrivals"]:
        raise ValueError("baseline below observed arrivals")
    if not isinstance(row.get("model_version"), str) or not row["model_version"]:
        raise ValueError("missing model version")
    if row.get("status") != "experimental_forecast":
        raise ValueError("unexpected forecast status")
    if not isinstance(row.get("within_prospective_window"), bool) or bool(
            row["within_prospective_window"]) != (11 <= hour <= 18):
        raise ValueError("prospective-window flag disagrees")
    return {**values, "forecast_day": day, "cutoff_ds_local": cutoff,
            "cutoff_hour": int(hour), "generated_at_utc": generated.tz_convert("UTC"),
            "model_version": row["model_version"]}


def read_artifacts(artifacts, runs, zip_dir, *, target_definition=None):
    inventory, forecasts, excluded = [], [], []
    run_ids = {r["id"] for r in runs}
    for artifact in artifacts:
        if artifact["workflow_run"]["head_branch"] != "main" or artifact["workflow_run"]["id"] not in run_ids:
            continue
        record = {"artifact_id": artifact["id"], "run_id": artifact["workflow_run"]["id"],
                  "head_sha": artifact["workflow_run"]["head_sha"],
                  "created_at": artifact["created_at"], "expected_digest": artifact.get("digest"),
                  "status": "unread", "reason": ""}
        path = zip_dir / f"{artifact['id']}.zip"
        try:
            data = path.read_bytes()
            record["sha256"] = hashlib.sha256(data).hexdigest()
            if artifact.get("expired") or artifact.get("digest") != "sha256:" + record["sha256"]:
                raise ValueError("expired artifact or digest mismatch")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                status = json.loads(archive.read("intraday-daily-inflow-status.json"))
                record["status"] = status["status"]
                record["reason"] = status.get("reason", "")
                # A leftover latest CSV never overrides an explicit suppression.
                if status["status"] in ["suppressed_data_quality", "suppressed_model_error"]:
                    continue
                if status["status"] != "forecast_written":
                    raise ValueError("unknown status")
                row = status["forecast"]
                # Target changes are separate collections. The legacy audit must
                # never score a new interval-day model against stored-calendar actuals.
                if row.get("target_definition_version") != target_definition:
                    record["status"] = "excluded_target_definition"
                    record["reason"] = "forecast target differs from this audit collection"
                    excluded.append({"artifact_id": artifact["id"], "reason": record["reason"]})
                    continue
                if target_definition is not None:
                    from arrival_day_policy import INTERVAL_QUALITY_VERSION, INTERVAL_TARGET_VERSION
                    if target_definition != INTERVAL_TARGET_VERSION or row.get("target_quality_version") != INTERVAL_QUALITY_VERSION:
                        raise ValueError("unrecognized target definition or quality version")
                csv = pd.read_csv(io.BytesIO(archive.read("intraday-daily-inflow-forecast.csv")))
                if len(csv) != 1:
                    raise ValueError("forecast CSV must have exactly one row")
                parsed = validate_forecast(row, artifact["created_at"])
                if pd.Timestamp(status["generated_at_utc"]) != parsed["generated_at_utc"]:
                    raise ValueError("status and forecast issue timestamps disagree")
                for key in NUMBERS:
                    if not np.isclose(float(csv.iloc[0][key]), parsed[key], rtol=0, atol=1e-8):
                        raise ValueError(f"CSV and status disagree: {key}")
                for key in ["forecast_day", "cutoff_ds_local", "generated_at_utc"]:
                    if pd.Timestamp(csv.iloc[0][key]) != pd.Timestamp(row[key]):
                        raise ValueError(f"CSV and status disagree: {key}")
                if csv.iloc[0]["model_version"] != parsed["model_version"]:
                    raise ValueError("CSV and status model versions disagree")
                if float(csv.iloc[0]["cutoff_hour"]) != parsed["cutoff_hour"]:
                    raise ValueError("CSV and status cutoff hours disagree")
                if csv.iloc[0]["status"] != row["status"] or bool(csv.iloc[0]["within_prospective_window"]) != row["within_prospective_window"]:
                    raise ValueError("CSV and status labels disagree")
                if target_definition is not None:
                    for key in ["target_definition_version", "target_quality_version"]:
                        if csv.iloc[0][key] != row[key]:
                            raise ValueError(f"CSV and status disagree: {key}")
                        parsed[key] = row[key]
                forecasts.append(parsed | {"artifact_id": artifact["id"], "run_id": record["run_id"]})
        except (ValueError, KeyError, TypeError, OSError, zipfile.BadZipFile) as exc:
            record["status"] = "quarantined_artifact"
            record["reason"] = str(exc)
            excluded.append({"artifact_id": artifact["id"], "reason": str(exc)})
        finally:
            inventory.append(record)
    if not forecasts:
        raise ValueError("No valid original forecast artifacts")
    frame = pd.DataFrame(forecasts).sort_values(["generated_at_utc", "artifact_id"])
    keys = ["forecast_day", "cutoff_hour", "model_version"]
    duplicates = frame.duplicated(keys, keep="first")
    excluded.extend({"artifact_id": r.artifact_id, "reason": "later repeated issue"}
                    for r in frame.loc[duplicates].itertuples())
    return frame.loc[~duplicates].copy(), pd.DataFrame(inventory), excluded


def score_forecasts(forecasts, hourly, *, now, daily_builder=build_daily_inflow_outputs):
    daily, quality = daily_builder(hourly, now=now)
    actual = daily.rename(columns={"ds": "forecast_day", "Daily_Inflow_Total": "actual"})
    rows = forecasts.merge(actual, on="forecast_day", how="left", validate="many_to_one")
    today = pd.Timestamp(now).tz_convert(TZ).tz_localize(None).normalize()
    exclusions = [{"artifact_id": r.artifact_id, "reason": "not yet matured" if r.forecast_day >= today
                   else "unverified actual"} for r in rows.loc[rows.actual.isna()].itertuples()]
    rows = rows.loc[rows.actual.notna()].copy()
    if rows.empty:
        raise ValueError("No matured verified forecast outcomes")
    rows["error"] = rows.predicted_total - rows.actual
    rows["baseline_error"] = rows.prior_update_baseline - rows.actual
    rows["covered"] = rows.actual.between(rows.p10_total, rows.p90_total)
    rows["interval_width"] = rows.p90_total - rows.p10_total
    return rows, quality, exclusions


def coverage_calendar(scores, start, end):
    """Missing calendar days break both longest and latest collection streaks."""
    rows = []
    for version in sorted(scores.model_version.unique()):
        group = scores.loc[scores.model_version.eq(version) & scores.cutoff_hour.between(11, 18)]
        for day in pd.date_range(start, end):
            hours = set(group.loc[group.forecast_day.eq(day), "cutoff_hour"].astype(int))
            missing = sorted(set(range(11, 19)) - hours)
            rows.append({"model_version": version, "day": day, "cutoffs_present": len(hours),
                         "missing_cutoffs": ";".join(map(str, missing)), "complete": not missing})
    return pd.DataFrame(rows)


def longest_streak(flags):
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest, current


def metric_row(frame, scope, version, hour=-1):
    mae = float(frame.error.abs().mean())
    baseline = float(frame.baseline_error.abs().mean())
    return {"model_version": version, "scope": scope, "cutoff_hour": hour,
            "n": len(frame), "days": frame.forecast_day.nunique(), "mae": mae,
            "baseline_mae": baseline, "improvement_pct": 100 * (1-mae/baseline) if baseline else np.nan,
            "bias": float(frame.error.mean()), "rmse": float(np.sqrt((frame.error**2).mean())),
            "p80_coverage": float(frame.covered.mean()), "mean_width": float(frame.interval_width.mean())}


def summarize(scores, coverage):
    metrics, gates = [], []
    for version, group in scores.groupby("model_version"):
        op = group.loc[group.cutoff_hour.between(11, 18)]
        full_days = coverage.loc[coverage.model_version.eq(version) & coverage.complete, "day"]
        overall = metric_row(group, "all_model_hours", version)
        operational = metric_row(op, "operational_available", version)
        metrics.extend([overall, operational, metric_row(op.loc[op.forecast_day.isin(full_days)],
                                                        "operational_complete_days", version)])
        by_hour = [metric_row(g, "cutoff_hour", version, int(h)) for h, g in group.groupby("cutoff_hour")]
        metrics.extend(by_hour)
        longest, latest = longest_streak(coverage.loc[coverage.model_version.eq(version), "complete"])
        operational_hours = [r for r in by_hour if 11 <= r["cutoff_hour"] <= 18]
        gates.append({"model_version": version, "complete_operational_days": len(full_days),
                      "longest_clean_streak": longest, "latest_clean_streak": latest,
                      "minimum_28_complete_days": len(full_days) >= 28,
                      "seven_consecutive_clean_days_observed": longest >= 7,
                      "latest_seven_days_clean": latest >= 7,
                      "overall_mae_improvement_at_least_5pct": overall["improvement_pct"] >= 5,
                      "operational_mae_improvement_at_least_5pct": operational["improvement_pct"] >= 5,
                      "overall_absolute_bias_at_most_2": abs(overall["bias"]) <= 2,
                      "every_operational_hour_absolute_bias_at_most_3": len(operational_hours) == 8
                      and all(abs(r["bias"]) <= 3 for r in operational_hours),
                      "overall_p80_coverage_75_to_85pct": .75 <= overall["p80_coverage"] <= .85,
                      "operational_p80_coverage_75_to_85pct": .75 <= operational["p80_coverage"] <= .85})
    return pd.DataFrame(metrics), gates


def day_boundary_audit(hourly, quality, start, end):
    """Compare stored-calendar and 01:00..next-00:00 sums without choosing a new target."""
    h = hourly.copy()
    h["ds"] = pd.to_datetime(h.ds)
    h = h.set_index("ds").sort_index()
    records = []
    eligible = quality.loc[quality.audit_eligible & quality.ds.between(start, end)]
    for row in eligible.itertuples():
        end_stamp = row.ds + pd.Timedelta(days=1)
        stamps = pd.date_range(row.ds + pd.Timedelta(hours=1), end_stamp, freq="h")
        if not stamps.isin(h.index).all():
            continue
        block = h.loc[stamps]
        inflow = pd.to_numeric(block.Inflow_Total, errors="coerce")
        if len(block) != 24 or not (np.isfinite(inflow) & inflow.ge(0)).all():
            continue
        counter = pd.to_numeric(block.Inflow_Cum_Total, errors="coerce") if "Inflow_Cum_Total" in block else pd.Series(np.nan, index=stamps)
        chain_matches = bool(np.isclose(inflow.cumsum(), counter, atol=1e-9, rtol=0).all())
        records.append({"day": row.ds, "stored_calendar_sum": row.observed_inflow_sum,
                        "shifted_hour_sum": float(inflow.sum()), "next_midnight_counter": counter.iloc[-1],
                        "entire_counter_chain_matches_shifted_sum": chain_matches,
                        "calendar_minus_shifted": row.observed_inflow_sum - float(inflow.sum())})
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["artifacts", "runs", "zip-dir", "hourly", "output-dir"]:
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--now", required=True, help="Aware timestamp fixing outcome maturity")
    args = parser.parse_args()
    inputs = {args.artifacts.resolve(), args.runs.resolve(), args.hourly.resolve()}
    inputs.update(p.resolve() for p in args.zip_dir.glob("*.zip"))
    if inputs.intersection((args.output_dir/name).resolve() for name in OUTPUTS):
        raise ValueError("Outputs must not overwrite source files")
    artifacts = json.loads(args.artifacts.read_text())
    runs = json.loads(args.runs.read_text())
    hourly = pd.read_csv(args.hourly)
    forecasts, inventory, exclusions = read_artifacts(artifacts, runs, args.zip_dir)
    scores, quality, score_exclusions = score_forecasts(forecasts, hourly, now=args.now)
    start = pd.to_datetime(inventory.created_at, utc=True).min().tz_convert(TZ).tz_localize(None).normalize()
    end = pd.Timestamp(args.now).tz_convert(TZ).tz_localize(None).normalize() - pd.Timedelta(days=1)
    coverage = coverage_calendar(scores, start, end)
    metrics, gates = summarize(scores, coverage)
    boundary = day_boundary_audit(hourly, quality, start, end)
    versions = sorted(scores.model_version.unique())
    summary = {"audit_at_utc": pd.Timestamp(args.now).tz_convert("UTC").isoformat(),
               "target": "sum Inflow_Total by stored naive Montreal ds date, including 00:00",
               "window_start": str(start.date()), "window_end": str(end.date()),
               "artifacts": len(inventory), "artifact_status_counts": inventory.status.value_counts().to_dict(),
               "original_forecasts": int(inventory.status.eq("forecast_written").sum()),
               "distinct_issues": len(forecasts), "scored_rows": len(scores),
               "operational_rows": int(scores.cutoff_hour.between(11,18).sum()),
               "model_versions": versions, "gates": gates,
               "promotion_decision": "no-go; explicit manual review and target-boundary resolution required",
               "boundary_days_compared": len(boundary),
               "boundary_counter_chain_matches": int(boundary.entire_counter_chain_matches_shifted_sum.sum()) if len(boundary) else 0,
               "boundary_days_with_different_totals": int(boundary.calendar_minus_shifted.ne(0).sum()) if len(boundary) else 0,
               "source_sha256": {name:hashlib.sha256(path.read_bytes()).hexdigest()
                                 for name,path in [("hourly",args.hourly),("artifacts",args.artifacts),("runs",args.runs)]},
               "limitations": ["Coverage applies to retained artifacts in the supplied run inventory, not inaccessible hospital logs or older unretained issues.",
                               "Missing or suppressed issues are not retrospectively recreated.",
                               "Actuals may include authoritative backfills; original forecast values and recorded baselines are unchanged.",
                               "The first archive day is left-censored; it cannot count as a complete day.",
                               "Timestamp/counter agreement is diagnostic evidence, not independent upstream confirmation."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name,frame in [("scores.csv",scores),("metrics.csv",metrics),("coverage.csv",coverage),
                       ("artifact_inventory.csv",inventory),("exclusions.csv",pd.DataFrame(exclusions+score_exclusions)),
                       ("day_boundary.csv",boundary)]:
        frame.to_csv(args.output_dir/name,index=False)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2,default=str)+"\n")
    print(json.dumps(summary,indent=2,default=str))


if __name__ == "__main__":
    main()
