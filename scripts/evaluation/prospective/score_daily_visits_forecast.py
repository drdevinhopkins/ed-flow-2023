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

This script is evaluation only; it never changes the operational forecast itself.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import dropbox
import numpy as np
import pandas as pd
import requests

SNAPSHOT_FOLDER = "/daily_visits_forecast_snapshots"
DAILY_INFLOW_PATH = "/daily_inflow.csv"
DEFAULT_DETAIL_OUTPUT = Path("daily_visits_prospective_detail.csv")
DEFAULT_SUMMARY_OUTPUT = Path("daily_visits_prospective_horizon_summary.csv")
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
    raw = _download_csv(dbx, DAILY_INFLOW_PATH)
    required = {"ds", "Daily_Inflow_Total"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"{DAILY_INFLOW_PATH} missing columns: {sorted(missing)}")

    out = raw[["ds", "Daily_Inflow_Total"]].copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce").dt.normalize()
    out["actual"] = pd.to_numeric(out["Daily_Inflow_Total"], errors="coerce")
    out = (
        out.drop(columns=["Daily_Inflow_Total"])
        .dropna(subset=["ds"])
        .sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )
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

    out["snapshot_name"] = snapshot_name
    return out


def load_forecast_archive(dbx: dropbox.Dropbox) -> pd.DataFrame:
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
    issue_times = archive.groupby("data_cutoff")["forecast_generated_at_utc"].transform("min")
    archive = archive.loc[archive["forecast_generated_at_utc"].eq(issue_times)].copy()
    # Defend against duplicate files containing the exact same earliest issue.
    archive = (
        archive.sort_values(["data_cutoff", "forecast_generated_at_utc", "snapshot_name", "horizon_day"])
        .drop_duplicates(["data_cutoff", "horizon_day"], keep="first")
        .reset_index(drop=True)
    )
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
        "evidence_ready": (
            n_issue_dates >= MIN_ISSUE_DATES and span_days >= MIN_PROSPECTIVE_SPAN_DAYS
        ),
    }


def summarize_by_horizon(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon_day in range(1, 8):
        group = detail.loc[detail["horizon_day"].astype(int).eq(horizon_day)]
        rows.append(_summary_row(group, f"D+{horizon_day}"))
    rows.append(_summary_row(detail, "all"))
    return pd.DataFrame(rows)


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
    parser.add_argument("--no-dropbox-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dbx = _dropbox_client()
    actuals = load_actual_daily(dbx)
    archive = load_forecast_archive(dbx)
    detail = score_archive(archive, actuals)
    summary = summarize_by_horizon(detail)

    args.detail_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.detail_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    matured_origins = detail["data_cutoff"].nunique() if not detail.empty else 0
    print(
        f"Scored {len(detail)} matured forecast rows from {matured_origins} issue date(s); "
        f"archive contains {archive['data_cutoff'].nunique()} issue date(s)."
    )
    print(summary.to_string(index=False))

    if not args.no_dropbox_output:
        _upload_csv(dbx, args.detail_output, args.detail_output.name)
        _upload_csv(dbx, args.summary_output, args.summary_output.name)
        print("Uploaded prospective daily-arrivals score outputs to Dropbox")


if __name__ == "__main__":
    main()
