#!/usr/bin/env python3
"""Prospectively score archived D+1..D+7 daily ED arrival forecasts.

The operational daily forecast already archives each issued ``daily_visits_forecast.csv``
under ``/daily_visits_forecast_snapshots`` in Dropbox. This scorer treats those immutable
forecast-time files as the prospective record, joins only target dates that have since
matured in ``/daily_inflow.csv``, and reports accuracy separately for horizon days 1..7.

For each data cutoff, the earliest archived forecast is retained so manual/repeated runs
cannot improve the score after seeing later information. A leakage-free eight-week
same-weekday baseline is rebuilt using only actual arrival totals available on or before
that forecast's data cutoff.

Outputs:
- ``daily_visits_prospective_detail.csv``: one matured forecast-date row per issue/horizon.
- ``daily_visits_prospective_horizon_summary.csv``: D+1..D+7 plus an all-horizons row.
- ``daily_visits_prospective_quality_summary.csv``: separate input-quality collections.

Actuals and comparator dates require verified hourly coverage. Historical forecasts
remain unchanged and are labeled legacy when they lack input-quality metadata.
``evidence_ready`` requires a sufficient collection with verified actuals and the
corrected forecast-input version; it does not itself imply an accuracy go decision.

Morning and evening issues additionally retain the earliest issue per scheduled window,
with per-slot summaries and matched-target comparisons. Legacy/manual runs are never
assigned a scheduled slot. Existing earliest-per-cutoff headline scores remain unchanged.

This script is evaluation only; it never changes the operational forecast itself.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import dropbox
import numpy as np
import pandas as pd
import requests

SCRIPTS = Path(__file__).resolve().parents[2]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from daily_arrival_quality import QUALITY_VERSION, load_verified_daily, verified_daily_targets
from daily_forecast_issues import issue_metadata, LOCAL_TZ

SNAPSHOT_FOLDER = "/daily_visits_forecast_snapshots"
DAILY_INFLOW_PATH = "/daily_inflow.csv"
DEFAULT_DETAIL_OUTPUT = Path("daily_visits_prospective_detail.csv")
DEFAULT_SUMMARY_OUTPUT = Path("daily_visits_prospective_horizon_summary.csv")
DEFAULT_QUALITY_SUMMARY_OUTPUT = Path("daily_visits_prospective_quality_summary.csv")
DEFAULT_ISSUE_DETAIL_OUTPUT = Path("daily_visits_prospective_issue_detail.csv")
DEFAULT_ISSUE_SUMMARY_OUTPUT = Path("daily_visits_prospective_issue_summary.csv")
DEFAULT_REFRESH_COMPARISON_OUTPUT = Path("daily_visits_prospective_refresh_comparison.csv")
MIN_ISSUE_DATES = 28
MIN_PROSPECTIVE_SPAN_DAYS = 28
BASELINE_WEEKS = 8

REQUIRED_FORECAST_COLUMNS = {
    "ds",
    "daily_visits_prediction",
    "0.1",
    "0.9",
    "data_cutoff",
    "horizon_day",
    "forecast_generated_at_utc",
}


def _dropbox_client() -> dropbox.Dropbox:
    key = os.environ.get("DROPBOX_APP_KEY")
    secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([key, secret, refresh]):
        raise RuntimeError(
            "Dropbox scoring requires DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and "
            "DROPBOX_REFRESH_TOKEN"
        )
    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh,
            "client_id": key,
            "client_secret": secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return dropbox.Dropbox(response.json()["access_token"])


def _download_csv(dbx: dropbox.Dropbox, path: str) -> pd.DataFrame:
    _metadata, response = dbx.files_download(path)
    return pd.read_csv(io.BytesIO(response.content))


def load_actual_daily(dbx: dropbox.Dropbox) -> pd.DataFrame:
    raw, _quality = load_verified_daily(dbx)
    out = raw.rename(columns={"Daily_Inflow_Total": "actual"})
    if out["actual"].notna().sum() == 0:
        raise ValueError(f"{DAILY_INFLOW_PATH} contains no usable actual arrivals")
    return out


def _iter_snapshot_metadata(dbx: dropbox.Dropbox):
    result = dbx.files_list_folder(SNAPSHOT_FOLDER)
    while True:
        for entry in result.entries:
            name = getattr(entry, "name", "")
            if name.startswith("daily_visits_forecast_") and name.endswith(".csv"):
                yield entry
        if not result.has_more:
            break
        result = dbx.files_list_folder_continue(result.cursor)


def normalize_snapshot(frame: pd.DataFrame, *, snapshot_name: str) -> pd.DataFrame:
    missing = REQUIRED_FORECAST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{snapshot_name} missing forecast columns: {sorted(missing)}")
    if len(frame) != 7:
        raise ValueError(f"{snapshot_name} expected 7 forecast rows, got {len(frame)}")

    out = frame.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.normalize()
    out["data_cutoff"] = pd.to_datetime(out["data_cutoff"], errors="coerce").dt.normalize()
    out["forecast_generated_at_utc"] = pd.to_datetime(
        out["forecast_generated_at_utc"], errors="coerce", utc=True
    )
    for column in ["daily_visits_prediction", "0.1", "0.9", "horizon_day"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    if out[["ds", "data_cutoff", "forecast_generated_at_utc"]].isna().any().any():
        raise ValueError(f"{snapshot_name} contains invalid forecast dates")
    if out[["daily_visits_prediction", "0.1", "0.9", "horizon_day"]].isna().any().any():
        raise ValueError(f"{snapshot_name} contains invalid forecast values")
    if out["data_cutoff"].nunique() != 1:
        raise ValueError(f"{snapshot_name} contains multiple data cutoffs")
    if out["forecast_generated_at_utc"].nunique() != 1:
        raise ValueError(f"{snapshot_name} contains multiple generated timestamps")
    if out["horizon_day"].astype(int).tolist() != list(range(1, 8)):
        raise ValueError(f"{snapshot_name} horizon_day is not D+1..D+7")
    expected_dates = pd.date_range(
        out["data_cutoff"].iloc[0] + pd.Timedelta(days=1), periods=7, freq="D"
    )
    if out["ds"].tolist() != expected_dates.tolist():
        raise ValueError(f"{snapshot_name} target dates do not match data cutoff")

    issue_columns = {"forecast_issue_slot", "forecast_issue_day_local"}
    if issue_columns.intersection(out.columns):
        if not issue_columns.issubset(out.columns):
            raise ValueError(f"{snapshot_name} has incomplete forecast issue metadata")
        if any(out[column].nunique(dropna=False) != 1 for column in issue_columns):
            raise ValueError(f"{snapshot_name} contains multiple issue labels")
        expected = issue_metadata(out["forecast_generated_at_utc"].iloc[0],
                                  cutoff=out["data_cutoff"].iloc[0],
                                  slot=out["forecast_issue_slot"].iloc[0])
        if not out["forecast_issue_day_local"].astype(str).eq(expected["forecast_issue_day_local"]).all():
            raise ValueError(f"{snapshot_name} issue date does not match actual issue time")

    out["snapshot_name"] = snapshot_name
    return out


def select_earliest_issues(archive: pd.DataFrame) -> pd.DataFrame:
    """Select the original issue per cutoff for both Dropbox and local replay."""
    issue_times = archive.groupby("data_cutoff")["forecast_generated_at_utc"].transform("min")
    return (
        archive.loc[archive["forecast_generated_at_utc"].eq(issue_times)]
        .sort_values(["data_cutoff", "forecast_generated_at_utc", "snapshot_name", "horizon_day"])
        .drop_duplicates(["data_cutoff", "horizon_day"], keep="first")
        .reset_index(drop=True)
    )


def load_forecast_archive(dbx: dropbox.Dropbox, *, all_issues=False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    for entry in _iter_snapshot_metadata(dbx):
        path = getattr(entry, "path_lower", None) or getattr(entry, "path_display", None)
        if not path:
            continue
        try:
            raw = _download_csv(dbx, path)
            frames.append(normalize_snapshot(raw, snapshot_name=entry.name))
        except Exception as exc:  # retain scoring if a legacy/bad archive file exists
            skipped.append(f"{entry.name}: {exc}")

    if not frames:
        if skipped:
            raise ValueError("No valid daily forecast snapshots; " + "; ".join(skipped[:5]))
        raise ValueError(f"No forecast snapshots found in {SNAPSHOT_FOLDER}")

    archive = pd.concat(frames, ignore_index=True)
    # One operational issue per data cutoff. Keep the earliest forecast so repeated/manual
    # runs cannot retrospectively improve the prospective score.
    if not all_issues:
        archive = select_earliest_issues(archive)
    if skipped:
        print(f"Skipped {len(skipped)} invalid/legacy snapshot(s); first: {skipped[0]}")
    return archive


def same_weekday_baseline(
    actuals: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    target_date: pd.Timestamp,
    weeks: int = BASELINE_WEEKS,
) -> float:
    eligible = actuals.loc[
        (actuals["ds"] <= cutoff)
        & (actuals["ds"].dt.weekday == target_date.weekday())
        & actuals["actual"].notna(),
        "actual",
    ].tail(weeks)
    if eligible.empty:
        eligible = actuals.loc[
            (actuals["ds"] <= cutoff) & actuals["actual"].notna(), "actual"
        ].tail(28)
    if eligible.empty:
        return float("nan")
    return float(eligible.mean())


def score_archive(archive: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    archive = archive.copy()
    if "target_quality_version" not in archive:
        archive["target_quality_version"] = "legacy_unverified"
    archive["target_quality_version"] = archive["target_quality_version"].fillna("legacy_unverified")
    actual_lookup = actuals[["ds", "actual"]].dropna(subset=["actual"])
    detail = archive.merge(actual_lookup, on="ds", how="left", validate="many_to_one")
    detail = detail.loc[detail["actual"].notna()].copy()
    if detail.empty:
        return detail

    baselines = []
    for row in detail.itertuples(index=False):
        baselines.append(
            same_weekday_baseline(
                actuals,
                cutoff=pd.Timestamp(row.data_cutoff),
                target_date=pd.Timestamp(row.ds),
            )
        )
    detail["baseline_prediction"] = baselines
    # Outcomes lacking any verified cutoff-safe baseline cannot enter comparisons.
    detail = detail.loc[np.isfinite(detail["baseline_prediction"])].copy()

    issue_day = detail["forecast_generated_at_utc"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    detail["calendar_days_ahead"] = (detail["ds"] - issue_day).dt.days
    detail["lead_hours_to_target_start"] = (
        detail["ds"].dt.tz_localize(LOCAL_TZ).dt.tz_convert("UTC")
        - detail["forecast_generated_at_utc"]
    ).dt.total_seconds() / 3600
    detail["forecast_error"] = detail["daily_visits_prediction"] - detail["actual"]
    detail["baseline_error"] = detail["baseline_prediction"] - detail["actual"]
    detail["forecast_absolute_error"] = detail["forecast_error"].abs()
    detail["baseline_absolute_error"] = detail["baseline_error"].abs()
    detail["forecast_squared_error"] = detail["forecast_error"] ** 2
    detail["baseline_squared_error"] = detail["baseline_error"] ** 2
    detail["forecast_beats_baseline"] = (
        detail["forecast_absolute_error"] < detail["baseline_absolute_error"]
    )
    detail["interval_80_covered"] = (
        (detail["actual"] >= detail["0.1"]) & (detail["actual"] <= detail["0.9"])
    )
    detail["interval_80_low_miss"] = detail["actual"] < detail["0.1"]
    detail["interval_80_high_miss"] = detail["actual"] > detail["0.9"]
    detail["interval_80_width"] = detail["0.9"] - detail["0.1"]
    detail["absolute_error_improvement"] = (
        detail["baseline_absolute_error"] - detail["forecast_absolute_error"]
    )
    detail["actual_quality_version"] = actuals.attrs.get("target_quality_version", "unverified")
    return detail.sort_values(["data_cutoff", "horizon_day"]).reset_index(drop=True)


def _summary_row(group: pd.DataFrame, horizon_label: str) -> dict[str, object]:
    n = len(group)
    n_issue_dates = int(group["data_cutoff"].nunique()) if n else 0
    first_cutoff = group["data_cutoff"].min() if n else pd.NaT
    last_cutoff = group["data_cutoff"].max() if n else pd.NaT
    span_days = (
        float((last_cutoff - first_cutoff) / pd.Timedelta(days=1)) if n_issue_dates >= 2 else 0.0
    )

    forecast_mae = float(group["forecast_absolute_error"].mean()) if n else np.nan
    baseline_mae = float(group["baseline_absolute_error"].mean()) if n else np.nan
    forecast_rmse = float(np.sqrt(group["forecast_squared_error"].mean())) if n else np.nan
    baseline_rmse = float(np.sqrt(group["baseline_squared_error"].mean())) if n else np.nan
    actual_sum = float(group["actual"].abs().sum()) if n else 0.0
    forecast_wape = float(group["forecast_absolute_error"].sum() / actual_sum) if actual_sum else np.nan
    baseline_wape = float(group["baseline_absolute_error"].sum() / actual_sum) if actual_sum else np.nan

    quality_verified = (n > 0 and "actual_quality_version" in group
                        and group["actual_quality_version"].eq(QUALITY_VERSION).all())
    forecast_verified = (n > 0 and "target_quality_version" in group
                         and group["target_quality_version"].eq(QUALITY_VERSION).all())
    versions = group["target_quality_version"].unique() if n else []
    collection_ready = n_issue_dates >= MIN_ISSUE_DATES and span_days >= MIN_PROSPECTIVE_SPAN_DAYS
    return {
        "horizon": horizon_label,
        "n": n,
        "n_issue_dates": n_issue_dates,
        "first_data_cutoff": first_cutoff,
        "last_data_cutoff": last_cutoff,
        "prospective_span_days": span_days,
        "forecast_mae": forecast_mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": baseline_mae - forecast_mae if n else np.nan,
        "mae_improvement_pct": (
            (baseline_mae - forecast_mae) / baseline_mae * 100
            if n and np.isfinite(baseline_mae) and baseline_mae != 0
            else np.nan
        ),
        "forecast_bias": float(group["forecast_error"].mean()) if n else np.nan,
        "baseline_bias": float(group["baseline_error"].mean()) if n else np.nan,
        "forecast_rmse": forecast_rmse,
        "baseline_rmse": baseline_rmse,
        "forecast_wape": forecast_wape,
        "baseline_wape": baseline_wape,
        "forecast_win_rate": float(group["forecast_beats_baseline"].mean()) if n else np.nan,
        "interval_80_coverage": float(group["interval_80_covered"].mean()) if n else np.nan,
        "interval_80_low_miss_rate": float(group["interval_80_low_miss"].mean()) if n else np.nan,
        "interval_80_high_miss_rate": float(group["interval_80_high_miss"].mean()) if n else np.nan,
        "interval_80_mean_width": float(group["interval_80_width"].mean()) if n else np.nan,
        "collection_ready": collection_ready,
        "actual_quality_verified": quality_verified,
        "actual_quality_version": QUALITY_VERSION if quality_verified else "unverified",
        "forecast_quality_version": str(versions[0]) if len(versions) == 1 else "mixed_or_empty",
        "forecast_quality_verified": forecast_verified,
        "evidence_ready": collection_ready and quality_verified and forecast_verified,
    }


def summarize_by_horizon(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon_day in range(1, 8):
        group = detail.loc[detail["horizon_day"].astype(int).eq(horizon_day)]
        rows.append(_summary_row(group, f"D+{horizon_day}"))
    rows.append(_summary_row(detail, "all"))
    return pd.DataFrame(rows)


def summarize_by_quality_version(detail: pd.DataFrame) -> pd.DataFrame:
    """Do not use legacy training semantics to promote the corrected input route."""
    if detail.empty:
        return summarize_by_horizon(detail)
    return pd.concat([summarize_by_horizon(group)
                      for _, group in detail.groupby("target_quality_version", sort=True)],
                     ignore_index=True)



def select_scheduled_issues(archive: pd.DataFrame) -> pd.DataFrame:
    """First actual morning and evening issue; no relabeling of older/manual runs."""
    if "forecast_issue_slot" not in archive:
        return archive.iloc[:0].copy()
    scheduled = archive.loc[archive.forecast_issue_slot.isin(["morning", "evening"])].copy()
    return (scheduled.sort_values(["forecast_generated_at_utc", "snapshot_name"])
            .drop_duplicates(["data_cutoff", "forecast_issue_day_local", "forecast_issue_slot", "horizon_day"])
            .reset_index(drop=True))


def summarize_by_issue_slot(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slot in ["morning", "evening"]:
        group = (detail.loc[detail.forecast_issue_slot.eq(slot)]
                 if "forecast_issue_slot" in detail else detail.iloc[:0])
        summary = summarize_by_quality_version(group)
        summary["forecast_issue_slot"] = slot
        # Existing D+ labels are relative to completed-day cutoff, so D+2 is tomorrow.
        summary["calendar_days_ahead"] = summary.horizon.map(
            {f"D+{i}": i - 1 for i in range(1, 8)})
        rows.append(summary)
    return pd.concat(rows, ignore_index=True)


def compare_refreshes(detail: pd.DataFrame) -> pd.DataFrame:
    """Compare only matched morning/evening target dates and quality versions."""
    columns = ["target_quality_version", "horizon_day", "calendar_days_ahead", "n_pairs",
               "morning_mae", "evening_mae", "mae_improvement", "evening_win_rate",
               "morning_interval_80_coverage", "evening_interval_80_coverage",
               "mean_absolute_revision", "morning_mean_lead_hours", "evening_mean_lead_hours"]
    if detail.empty or "forecast_issue_slot" not in detail:
        return pd.DataFrame(columns=columns)
    keys = ["data_cutoff", "ds", "horizon_day", "target_quality_version"]
    morning = detail.loc[detail.forecast_issue_slot.eq("morning")]
    evening = detail.loc[detail.forecast_issue_slot.eq("evening")]
    paired = morning.merge(evening, on=keys, suffixes=("_morning", "_evening"), validate="one_to_one")
    rows = []
    for (version, horizon), group in paired.groupby(["target_quality_version", "horizon_day"]):
        am = group.forecast_absolute_error_morning
        pm = group.forecast_absolute_error_evening
        rows.append(dict(zip(columns, [
            version, horizon, int(horizon) - 1, len(group), am.mean(), pm.mean(),
            (am - pm).mean(), (pm < am).mean(), group.interval_80_covered_morning.mean(),
            group.interval_80_covered_evening.mean(),
            (group.daily_visits_prediction_evening - group.daily_visits_prediction_morning).abs().mean(),
            group.lead_hours_to_target_start_morning.mean(),
            group.lead_hours_to_target_start_evening.mean(),
        ])))
    return pd.DataFrame(rows, columns=columns)

def _upload_csv(dbx: dropbox.Dropbox, local_path: Path, remote_name: str) -> None:
    with local_path.open("rb") as handle:
        dbx.files_upload(
            handle.read(),
            "/" + remote_name,
            mode=dropbox.files.WriteMode.overwrite,
            mute=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail-output", type=Path, default=DEFAULT_DETAIL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--quality-summary-output", type=Path, default=DEFAULT_QUALITY_SUMMARY_OUTPUT)
    parser.add_argument("--issue-detail-output", type=Path, default=DEFAULT_ISSUE_DETAIL_OUTPUT)
    parser.add_argument("--issue-summary-output", type=Path, default=DEFAULT_ISSUE_SUMMARY_OUTPUT)
    parser.add_argument("--refresh-comparison-output", type=Path, default=DEFAULT_REFRESH_COMPARISON_OUTPUT)
    parser.add_argument("--no-dropbox-output", action="store_true")
    parser.add_argument("--daily-csv", type=Path)
    parser.add_argument("--hourly-csv", type=Path)
    parser.add_argument("--archive-dir", type=Path, help="Local immutable forecast snapshots")
    parser.add_argument("--now", help="Aware audit timestamp for reproducible local replay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    issue_detail_path = getattr(args, "issue_detail_output", DEFAULT_ISSUE_DETAIL_OUTPUT)
    issue_summary_path = getattr(args, "issue_summary_output", DEFAULT_ISSUE_SUMMARY_OUTPUT)
    comparison_path = getattr(args, "refresh_comparison_output", DEFAULT_REFRESH_COMPARISON_OUTPUT)
    local = any([args.daily_csv, args.hourly_csv, args.archive_dir, args.now])
    if local:
        if not all([args.daily_csv, args.hourly_csv, args.archive_dir, args.no_dropbox_output]):
            raise ValueError("Local replay requires daily-csv, hourly-csv, archive-dir and no-dropbox-output")
        inputs = {args.daily_csv.resolve(), args.hourly_csv.resolve()}
        snapshots = sorted(args.archive_dir.glob("daily_visits_forecast_*.csv"))
        inputs.update(path.resolve() for path in snapshots)
        outputs = [path.resolve() for path in
                   [args.detail_output, args.summary_output, args.quality_summary_output,
                    issue_detail_path, issue_summary_path, comparison_path]]
        if inputs.intersection(outputs) or len(set(outputs)) != len(outputs):
            raise ValueError("Replay outputs must be distinct and must not overwrite inputs or snapshots")
        daily, _ = verified_daily_targets(
            pd.read_csv(args.daily_csv), pd.read_csv(args.hourly_csv),
            now=pd.Timestamp(args.now) if args.now else None,
        )
        actuals = daily.rename(columns={"Daily_Inflow_Total": "actual"})
        frames = [normalize_snapshot(pd.read_csv(path), snapshot_name=path.name)
                  for path in snapshots]
        if not frames:
            raise ValueError("No local immutable forecast snapshots")
        all_issues = pd.concat(frames, ignore_index=True)
        archive = select_earliest_issues(all_issues)
        dbx = None
    else:
        dbx = _dropbox_client()
        actuals = load_actual_daily(dbx)
        all_issues = load_forecast_archive(dbx, all_issues=True)
        archive = select_earliest_issues(all_issues)
    detail = score_archive(archive, actuals)
    summary = summarize_by_horizon(detail)
    quality_summary = summarize_by_quality_version(detail)
    issue_detail = score_archive(select_scheduled_issues(all_issues), actuals)
    issue_summary = summarize_by_issue_slot(issue_detail)
    comparison = compare_refreshes(issue_detail)

    outputs = [(detail, args.detail_output), (summary, args.summary_output),
               (quality_summary, args.quality_summary_output), (issue_detail, issue_detail_path),
               (issue_summary, issue_summary_path), (comparison, comparison_path)]
    for frame, path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    matured_origins = detail["data_cutoff"].nunique() if not detail.empty else 0
    print(
        f"Scored {len(detail)} matured forecast rows from {matured_origins} issue date(s); "
        f"archive contains {archive['data_cutoff'].nunique()} issue date(s)."
    )
    print(summary.to_string(index=False))

    if not args.no_dropbox_output:
        for _frame, path in outputs:
            _upload_csv(dbx, path, path.name)
        print("Uploaded prospective daily-arrivals score outputs to Dropbox")


if __name__ == "__main__":
    main()
