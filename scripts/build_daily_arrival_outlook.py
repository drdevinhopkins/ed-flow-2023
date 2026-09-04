#!/usr/bin/env python3
"""Build the Power BI presentation-layer daily ED arrival outlook.

The underlying forecast contracts remain unchanged. This script maps the explained
D+1..D+7 daily forecast into a friendly schema and, when a valid same-day intraday
completion forecast is available, replaces only today's estimate with that newer
forecast. Future dates continue to come from the daily Chronos-2 model.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

LOCAL_TZ = "America/Montreal"
DAILY_MODEL_VERSION = "amazon/chronos-2"
DEFAULT_DAILY_EXPLAINED = Path("daily_visits_forecast_explained.csv")
DEFAULT_INTRADAY = Path("intraday-daily-inflow-forecast.csv")
DEFAULT_OUTPUT = Path("daily_arrival_outlook.csv")
DEFAULT_DROPBOX_DAILY_PATH = "/daily_visits_forecast_explained.csv"
DEFAULT_DROPBOX_INTRADAY_PATH = "/intraday-daily-inflow-forecast.csv"
DEFAULT_DROPBOX_OUTPUT_PATH = "/daily_arrival_outlook.csv"

DAILY_REQUIRED = {
    "ds",
    "daily_visits_prediction",
    "0.1",
    "0.9",
    "data_cutoff",
    "forecast_generated_at_utc",
    "seasonal_weekday_baseline",
    "explainability_method",
    "top_driver_1",
    "top_driver_1_effect",
    "top_driver_2",
    "top_driver_2_effect",
    "top_driver_3",
    "top_driver_3_effect",
    "explanation_text",
}

INTRADAY_REQUIRED = {
    "generated_at_utc",
    "generated_at_local",
    "forecast_day",
    "cutoff_ds_local",
    "observed_arrivals",
    "predicted_total",
    "p10_total",
    "p90_total",
    "expected_additional_arrivals",
    "model_version",
    "method",
    "forecast_text",
}

OUTLOOK_COLUMNS = [
    "target_date",
    "generated_at_local",
    "forecast_stage",
    "horizon_day",
    "predicted_arrivals",
    "lower_80",
    "upper_80",
    "observed_arrivals",
    "expected_remaining",
    "seasonal_weekday_baseline",
    "delta_vs_baseline",
    "top_driver_1",
    "top_driver_1_effect",
    "top_driver_2",
    "top_driver_2_effect",
    "top_driver_3",
    "top_driver_3_effect",
    "explainability_method",
    "explanation_text",
    "source_model",
    "data_cutoff",
    "model_version",
]


def _normalize_today(value: str | pd.Timestamp | None = None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).normalize()
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert(LOCAL_TZ).tz_localize(None)
    return stamp.normalize()


def _to_local_iso(value: object) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(LOCAL_TZ).isoformat()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_daily_outlook(
    daily_explained: pd.DataFrame,
    *,
    intraday: pd.DataFrame | None = None,
    today: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return one best-current-estimate row per target date.

    Daily rows are retained for today through seven days ahead. A same-day intraday
    forecast supersedes the daily estimate for today only. Stale intraday rows are
    ignored rather than allowed to overwrite the current date.
    """

    _require_columns(daily_explained, DAILY_REQUIRED, "daily explained forecast")
    current_day = _normalize_today(today)
    max_day = current_day + pd.Timedelta(days=7)

    daily = daily_explained.copy()
    daily["_target_date"] = pd.to_datetime(daily["ds"], errors="raise").dt.normalize()
    daily = daily.loc[
        daily["_target_date"].between(current_day, max_day, inclusive="both")
    ].copy()
    if daily["_target_date"].duplicated().any():
        duplicated = daily.loc[daily["_target_date"].duplicated(), "ds"].tolist()
        raise ValueError(f"daily explained forecast contains duplicate dates: {duplicated}")

    rows: list[dict[str, object]] = []
    for record in daily.sort_values("_target_date").to_dict("records"):
        target = pd.Timestamp(record["_target_date"])
        horizon = int((target - current_day).days)
        prediction = float(record["daily_visits_prediction"])
        baseline = float(record["seasonal_weekday_baseline"])
        rows.append(
            {
                "target_date": target.date().isoformat(),
                "generated_at_local": _to_local_iso(record["forecast_generated_at_utc"]),
                "forecast_stage": "day_ahead" if horizon == 0 else "daily",
                "horizon_day": horizon,
                "predicted_arrivals": prediction,
                "lower_80": float(record["0.1"]),
                "upper_80": float(record["0.9"]),
                "observed_arrivals": pd.NA,
                "expected_remaining": pd.NA,
                "seasonal_weekday_baseline": baseline,
                "delta_vs_baseline": prediction - baseline,
                "top_driver_1": record.get("top_driver_1"),
                "top_driver_1_effect": record.get("top_driver_1_effect"),
                "top_driver_2": record.get("top_driver_2"),
                "top_driver_2_effect": record.get("top_driver_2_effect"),
                "top_driver_3": record.get("top_driver_3"),
                "top_driver_3_effect": record.get("top_driver_3_effect"),
                "explainability_method": record.get("explainability_method"),
                "explanation_text": record.get("explanation_text"),
                "source_model": "daily_chronos2",
                "data_cutoff": str(record["data_cutoff"]),
                "model_version": DAILY_MODEL_VERSION,
            }
        )

    outlook = pd.DataFrame(rows, columns=OUTLOOK_COLUMNS)

    if intraday is not None and not intraday.empty:
        _require_columns(intraday, INTRADAY_REQUIRED, "intraday forecast")
        if len(intraday) != 1:
            raise ValueError(f"expected one intraday forecast row, got {len(intraday)}")
        live = intraday.iloc[0]
        forecast_day = pd.Timestamp(live["forecast_day"]).normalize()
        if forecast_day == current_day:
            today_key = current_day.date().isoformat()
            baseline = pd.NA
            if not outlook.empty and outlook["target_date"].eq(today_key).any():
                baseline = outlook.loc[
                    outlook["target_date"].eq(today_key), "seasonal_weekday_baseline"
                ].iloc[0]
                outlook = outlook.loc[~outlook["target_date"].eq(today_key)].copy()

            prediction = float(live["predicted_total"])
            delta_vs_baseline = (
                prediction - float(baseline) if not pd.isna(baseline) else pd.NA
            )
            intraday_row = pd.DataFrame(
                [
                    {
                        "target_date": today_key,
                        "generated_at_local": str(live["generated_at_local"]),
                        "forecast_stage": "intraday",
                        "horizon_day": 0,
                        "predicted_arrivals": prediction,
                        "lower_80": float(live["p10_total"]),
                        "upper_80": float(live["p90_total"]),
                        "observed_arrivals": float(live["observed_arrivals"]),
                        "expected_remaining": float(live["expected_additional_arrivals"]),
                        "seasonal_weekday_baseline": baseline,
                        "delta_vs_baseline": delta_vs_baseline,
                        "top_driver_1": pd.NA,
                        "top_driver_1_effect": pd.NA,
                        "top_driver_2": pd.NA,
                        "top_driver_2_effect": pd.NA,
                        "top_driver_3": pd.NA,
                        "top_driver_3_effect": pd.NA,
                        "explainability_method": pd.NA,
                        "explanation_text": str(live["forecast_text"]),
                        "source_model": "intraday_day_completion",
                        "data_cutoff": str(live["cutoff_ds_local"]),
                        "model_version": str(live["model_version"]),
                    }
                ],
                columns=OUTLOOK_COLUMNS,
            )
            outlook = pd.concat([outlook, intraday_row], ignore_index=True)

    if outlook.empty:
        raise ValueError("no current or future outlook rows were produced")

    outlook["_sort_date"] = pd.to_datetime(outlook["target_date"])
    outlook = outlook.sort_values("_sort_date").drop(columns="_sort_date").reset_index(drop=True)

    numeric_triplet = outlook[["lower_80", "predicted_arrivals", "upper_80"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not (
        numeric_triplet["lower_80"].le(numeric_triplet["predicted_arrivals"]).all()
        and numeric_triplet["predicted_arrivals"].le(numeric_triplet["upper_80"]).all()
    ):
        raise ValueError("outlook interval invariant failed")
    if outlook["target_date"].duplicated().any():
        raise ValueError("outlook contains duplicate target dates")
    return outlook[OUTLOOK_COLUMNS]


def _dropbox_client():
    import dropbox
    import requests

    app_key = os.environ.get("DROPBOX_APP_KEY")
    app_secret = os.environ.get("DROPBOX_APP_SECRET")
    refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")
    if not all([app_key, app_secret, refresh_token]):
        raise RuntimeError("Dropbox credentials are required for Dropbox I/O")
    response = requests.post(
        "https://api.dropboxapi.com/oauth2/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": app_key,
            "client_secret": app_secret,
        },
        timeout=30,
    )
    response.raise_for_status()
    return dropbox.Dropbox(response.json()["access_token"])


def _download_dropbox(remote_path: str, local_path: Path, *, optional: bool) -> bool:
    try:
        client = _dropbox_client()
        _metadata, response = client.files_download(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(response.content)
        return True
    except Exception as exc:
        if optional:
            print(
                f"Optional Dropbox download skipped for {remote_path}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return False
        raise


def _upload_dropbox(local_path: Path, remote_path: str) -> None:
    import dropbox

    client = _dropbox_client()
    client.files_upload(
        local_path.read_bytes(),
        remote_path,
        mode=dropbox.files.WriteMode.overwrite,
        mute=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-explained", type=Path, default=DEFAULT_DAILY_EXPLAINED)
    parser.add_argument("--intraday", type=Path, default=DEFAULT_INTRADAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--today", help="Override Montreal-local current date for testing")
    parser.add_argument("--download-daily-dropbox", action="store_true")
    parser.add_argument("--download-intraday-dropbox", action="store_true")
    parser.add_argument("--upload-dropbox", action="store_true")
    parser.add_argument("--dropbox-daily-path", default=DEFAULT_DROPBOX_DAILY_PATH)
    parser.add_argument("--dropbox-intraday-path", default=DEFAULT_DROPBOX_INTRADAY_PATH)
    parser.add_argument("--dropbox-output-path", default=DEFAULT_DROPBOX_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.download_daily_dropbox:
        _download_dropbox(args.dropbox_daily_path, args.daily_explained, optional=False)
    if args.download_intraday_dropbox:
        _download_dropbox(args.dropbox_intraday_path, args.intraday, optional=True)

    daily = pd.read_csv(args.daily_explained)
    intraday = pd.read_csv(args.intraday) if args.intraday.exists() else None
    outlook = build_daily_outlook(daily, intraday=intraday, today=args.today)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    outlook.to_csv(args.output, index=False)
    print(f"Wrote {len(outlook)} outlook rows to {args.output}")
    print(
        outlook[
            [
                "target_date",
                "forecast_stage",
                "predicted_arrivals",
                "lower_80",
                "upper_80",
                "observed_arrivals",
                "expected_remaining",
            ]
        ].to_string(index=False)
    )
    if args.upload_dropbox:
        _upload_dropbox(args.output, args.dropbox_output_path)
        print(f"Uploaded {args.output} to Dropbox {args.dropbox_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
