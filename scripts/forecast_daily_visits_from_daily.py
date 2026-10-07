#!/usr/bin/env python3
"""Run the daily ED forecast from coverage-verified daily_inflow.csv targets."""

from __future__ import annotations

from daily_arrival_quality import DAILY_PATH, load_verified_daily, require_latest_completed_day

DAILY_INFLOW_DROPBOX_PATH = DAILY_PATH
DAILY_INFLOW_SOURCE = f"dropbox:{DAILY_PATH}"


def load_daily_visits_from_dropbox(_source=DAILY_INFLOW_SOURCE):
    import forecast_daily_visits as forecast

    dbx = forecast._dropbox_client()
    if dbx is None:
        raise RuntimeError("Daily target verification requires Dropbox credentials")
    daily, _quality = load_verified_daily(dbx)
    require_latest_completed_day(daily)
    daily = daily.rename(columns={"Daily_Inflow_Total": forecast.TARGET})
    return daily


def main():
    import forecast_daily_visits as forecast

    forecast.FLOW_URL = DAILY_INFLOW_SOURCE
    forecast.load_daily_visits = load_daily_visits_from_dropbox
    forecast.main()


if __name__ == "__main__":
    main()
