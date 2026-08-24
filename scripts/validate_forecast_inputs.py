#!/usr/bin/env python3
"""Validate that production forecast inputs are fresh and cover the forecast horizon.

The hourly workflow intentionally allows some upstream refresh jobs to continue on error.
This gate prevents Chronos from silently running on stale or incomplete ED, staffing,
weather, or anomaly-range inputs after one of those refreshes fails.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ED_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
SHIFT_URL = (
    "https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/"
    "all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1"
)
WEATHER_URL = (
    "https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/"
    "weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1"
)
ANOMALY_RANGES_URL = (
    "https://www.dropbox.com/scl/fi/fjz0am427gw35sz7l994m/"
    "anomaly_detection_ranges.csv?rlkey=lib9w0jz2zei5n566jv76o7ol&raw=1"
)
LOCAL_TZ = ZoneInfo("America/Montreal")

ANOMALY_TARGET_ALIASES = {
    "Total_TBS": "total_tbs",
    "POD_TBS": "pod_tbs",
    "Vertical_TBS": "vert_tbs",
    "TTStr": "TTStr",
    "Overflow": "overflow",
    "WAITINGADM": "WAITINGADM",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _parse_local_naive(series: pd.Series) -> pd.Series:
    """Parse timestamps as Montreal wall-clock timestamps used by this repository."""
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return parsed


def _hour_set(series: pd.Series) -> set[pd.Timestamp]:
    parsed = _parse_local_naive(series).dropna().dt.floor("h")
    return set(parsed.tolist())


def check_ed_data(
    df: pd.DataFrame,
    *,
    now_local: pd.Timestamp,
    max_age_hours: float,
    continuity_hours: int,
) -> tuple[list[CheckResult], pd.Timestamp]:
    results: list[CheckResult] = []
    if "ds" not in df.columns:
        raise ValueError("ED dataset is missing required timestamp column 'ds'.")

    ds = _parse_local_naive(df["ds"]).dropna().dt.floor("h")
    if ds.empty:
        raise ValueError("ED dataset contains no valid timestamps.")

    latest = ds.max()
    age_hours = (now_local - latest).total_seconds() / 3600
    results.append(
        CheckResult(
            "ED freshness",
            -1.0 <= age_hours <= max_age_hours,
            f"latest={latest}, age={age_hours:.2f}h, allowed<= {max_age_hours:.1f}h",
        )
    )

    unique_hours = set(ds.tolist())
    continuity_start = latest - pd.Timedelta(hours=continuity_hours - 1)
    expected = pd.date_range(continuity_start, latest, freq="h")
    missing = [ts for ts in expected if ts not in unique_hours]
    results.append(
        CheckResult(
            "ED hourly continuity",
            not missing,
            (
                f"last {continuity_hours}h complete"
                if not missing
                else f"missing {len(missing)} hour(s), first={missing[:5]}"
            ),
        )
    )
    return results, latest


def check_staffing(
    df: pd.DataFrame,
    *,
    forecast_start: pd.Timestamp,
    horizon: int,
) -> list[CheckResult]:
    required = {"shift_start", "shift_end"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Staffing dataset missing columns: {sorted(missing_cols)}")

    starts = _parse_local_naive(df["shift_start"])
    ends = _parse_local_naive(df["shift_end"])
    valid = pd.DataFrame({"start": starts, "end": ends}).dropna()

    expected = pd.date_range(forecast_start, periods=horizon, freq="h")
    uncovered: list[pd.Timestamp] = []
    for hour in expected:
        covered = ((valid["start"] <= hour) & (valid["end"] > hour)).any()
        if not covered:
            uncovered.append(hour)

    latest_end = valid["end"].max() if not valid.empty else pd.NaT
    return [
        CheckResult(
            "Staffing horizon coverage",
            not uncovered,
            (
                f"all {horizon} forecast hours covered; latest shift end={latest_end}"
                if not uncovered
                else f"uncovered {len(uncovered)} hour(s), first={uncovered[:5]}"
            ),
        )
    ]


def check_weather(
    df: pd.DataFrame,
    *,
    forecast_start: pd.Timestamp,
    horizon: int,
) -> list[CheckResult]:
    if "ds" not in df.columns:
        raise ValueError("Weather dataset is missing required timestamp column 'ds'.")

    hours = _hour_set(df["ds"])
    expected = pd.date_range(forecast_start, periods=horizon, freq="h")
    missing = [ts for ts in expected if ts not in hours]
    latest = max(hours) if hours else pd.NaT
    return [
        CheckResult(
            "Weather horizon coverage",
            not missing,
            (
                f"all {horizon} forecast hours covered; latest weather={latest}"
                if not missing
                else f"missing {len(missing)} hour(s), first={missing[:5]}, latest={latest}"
            ),
        )
    ]


def check_anomaly_ranges(
    df: pd.DataFrame,
    *,
    forecast_start: pd.Timestamp,
    horizon: int,
) -> list[CheckResult]:
    """Ensure v2 anomaly references exist and are usable for every forecast hour."""
    if "ds" not in df.columns:
        raise ValueError("Anomaly range dataset is missing required timestamp column 'ds'.")

    frame = df.copy()
    frame["ds"] = _parse_local_naive(frame["ds"]).dt.floor("h")
    frame = frame.dropna(subset=["ds"]).drop_duplicates("ds", keep="last")
    frame = frame.set_index("ds")

    expected = pd.date_range(forecast_start, periods=horizon, freq="h")
    available_hours = set(frame.index.tolist())
    missing_hours = [ts for ts in expected if ts not in available_hours]
    latest = max(available_hours) if available_hours else pd.NaT

    results = [
        CheckResult(
            "Anomaly range horizon coverage",
            not missing_hours,
            (
                f"all {horizon} forecast hours covered; latest anomaly range={latest}"
                if not missing_hours
                else (
                    f"missing {len(missing_hours)} hour(s), first={missing_hours[:5]}, "
                    f"latest={latest}"
                )
            ),
        )
    ]

    target_problems: list[str] = []
    present_hours = [ts for ts in expected if ts in available_hours]
    for target, alias in ANOMALY_TARGET_ALIASES.items():
        columns = None
        for prefix in (alias, target):
            candidate = (
                f"{prefix}_yhat",
                f"{prefix}_yhat_lower",
                f"{prefix}_yhat_upper",
            )
            if all(column in frame.columns for column in candidate):
                columns = candidate
                break

        if columns is None:
            target_problems.append(f"{target}: missing yhat columns")
            continue

        if present_hours:
            values = frame.loc[present_hours, list(columns)].apply(
                pd.to_numeric, errors="coerce"
            )
            null_rows = values.isna().any(axis=1)
            if null_rows.any():
                first_bad = values.index[null_rows][:3].tolist()
                target_problems.append(f"{target}: null interval(s) at {first_bad}")

    results.append(
        CheckResult(
            "Anomaly range target coverage",
            not target_problems,
            (
                f"all {len(ANOMALY_TARGET_ALIASES)} v2 targets have non-null intervals"
                if not target_problems
                else "; ".join(target_problems[:6])
            ),
        )
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ed-url", default=ED_URL)
    parser.add_argument("--shift-url", default=SHIFT_URL)
    parser.add_argument("--weather-url", default=WEATHER_URL)
    parser.add_argument("--anomaly-url", default=ANOMALY_RANGES_URL)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--max-ed-age-hours", type=float, default=3.0)
    parser.add_argument("--continuity-hours", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now_local = pd.Timestamp(datetime.now(LOCAL_TZ).replace(tzinfo=None)).floor("min")
    print(f"Forecast input validation time: {now_local} America/Montreal")

    ed_df = pd.read_csv(args.ed_url)
    shift_df = pd.read_csv(args.shift_url)
    weather_df = pd.read_csv(args.weather_url)
    anomaly_df = pd.read_csv(args.anomaly_url)

    results, latest_ed = check_ed_data(
        ed_df,
        now_local=now_local,
        max_age_hours=args.max_ed_age_hours,
        continuity_hours=args.continuity_hours,
    )
    forecast_start = latest_ed + pd.Timedelta(hours=1)
    results.extend(
        check_staffing(shift_df, forecast_start=forecast_start, horizon=args.horizon)
    )
    results.extend(
        check_weather(weather_df, forecast_start=forecast_start, horizon=args.horizon)
    )
    results.extend(
        check_anomaly_ranges(
            anomaly_df, forecast_start=forecast_start, horizon=args.horizon
        )
    )

    failed = [result for result in results if not result.ok]
    for result in results:
        marker = "PASS" if result.ok else "FAIL"
        print(f"[{marker}] {result.name}: {result.detail}")

    if failed:
        names = ", ".join(result.name for result in failed)
        raise SystemExit(f"Forecast input validation failed: {names}")

    print(
        f"All forecast inputs validated for {args.horizon}h beginning {forecast_start}."
    )


if __name__ == "__main__":
    main()
