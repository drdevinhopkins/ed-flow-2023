#!/usr/bin/env python3
"""Run the operational Chronos-2 daily ED forecast from ``daily_inflow.csv``.

``daily_inflow.csv`` is the authoritative daily target produced by ``get_current.py``
and uploaded to the app-folder root in Dropbox.  This wrapper intentionally bypasses
the hourly reconstruction used by the research/backtest loader, so an isolated missing
hour in ``allDataWithCalculatedColumns.csv`` cannot collapse the production Chronos
context window.

The forecasting, calendar, weather, uncertainty, archival, and output logic remains in
``forecast_daily_visits.py``.
"""

from __future__ import annotations

import io

import pandas as pd

import forecast_daily_visits as forecast

DAILY_INFLOW_DROPBOX_PATH = "/daily_inflow.csv"
DAILY_INFLOW_SOURCE = f"dropbox:{DAILY_INFLOW_DROPBOX_PATH}"


def load_daily_visits_from_dropbox(_source: str = DAILY_INFLOW_SOURCE) -> pd.DataFrame:
    """Load the already-aggregated daily ED arrival target from Dropbox."""
    dbx = forecast._dropbox_client()
    if dbx is None:
        raise RuntimeError(
            "daily_inflow.csv requires DROPBOX_APP_KEY, DROPBOX_APP_SECRET, and "
            "DROPBOX_REFRESH_TOKEN"
        )

    _metadata, response = dbx.files_download(DAILY_INFLOW_DROPBOX_PATH)
    raw = pd.read_csv(io.BytesIO(response.content))

    required = {"ds", "Daily_Inflow_Total"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(
            f"{DAILY_INFLOW_DROPBOX_PATH} is missing required columns: {sorted(missing)}"
        )

    daily = raw[["ds", "Daily_Inflow_Total"]].copy()
    daily["ds"] = pd.to_datetime(daily["ds"], errors="coerce").dt.normalize()
    daily[forecast.TARGET] = pd.to_numeric(
        daily["Daily_Inflow_Total"], errors="coerce"
    ).astype("float64")
    daily = (
        daily.drop(columns=["Daily_Inflow_Total"])
        .dropna(subset=["ds"])
        .sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )

    if daily.empty or daily[forecast.TARGET].notna().sum() == 0:
        raise ValueError(f"{DAILY_INFLOW_DROPBOX_PATH} contains no usable daily arrivals")

    # Preserve any genuine missing calendar dates as NaN so the existing production
    # continuity guard can still detect a real gap in the daily source itself.
    full_index = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D", name="ds")
    daily = daily.set_index("ds").reindex(full_index).reset_index()

    latest = daily.loc[daily[forecast.TARGET].notna(), "ds"].max()
    history_start = daily.loc[daily[forecast.TARGET].notna(), "ds"].min()
    missing_days = int(daily[forecast.TARGET].isna().sum())
    print(
        f"Loaded daily target from {DAILY_INFLOW_DROPBOX_PATH}: "
        f"{history_start.date()}..{latest.date()} ({missing_days} missing calendar days)",
        flush=True,
    )
    return daily[["ds", forecast.TARGET]]


def main() -> None:
    # parse_args() reads FLOW_URL at call time.  Keep --flow-url compatibility while
    # making the operational default explicitly describe the actual source.
    forecast.FLOW_URL = DAILY_INFLOW_SOURCE
    forecast.load_daily_visits = load_daily_visits_from_dropbox
    forecast.main()


if __name__ == "__main__":
    main()
