#!/usr/bin/env python3
"""Regression tests for forecast input validation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_forecast_inputs import ANOMALY_TARGET_ALIASES, check_anomaly_ranges


def _anomaly_frame(start: pd.Timestamp, periods: int = 24) -> pd.DataFrame:
    frame = pd.DataFrame({"ds": pd.date_range(start, periods=periods, freq="h")})
    for alias in ANOMALY_TARGET_ALIASES.values():
        frame[f"{alias}_yhat"] = 10.0
        frame[f"{alias}_yhat_lower"] = 5.0
        frame[f"{alias}_yhat_upper"] = 20.0
    return frame


def test_complete_anomaly_horizon_passes() -> None:
    start = pd.Timestamp("2026-08-24 07:00:00")
    results = check_anomaly_ranges(_anomaly_frame(start), forecast_start=start, horizon=24)
    assert all(result.ok for result in results), results


def test_missing_final_anomaly_hour_fails_horizon_gate() -> None:
    start = pd.Timestamp("2026-08-24 07:00:00")
    frame = _anomaly_frame(start, periods=23)
    results = check_anomaly_ranges(frame, forecast_start=start, horizon=24)
    by_name = {result.name: result for result in results}

    coverage = by_name["Anomaly range horizon coverage"]
    assert not coverage.ok
    assert "2026-08-25 06:00:00" in coverage.detail


def test_null_target_interval_fails_target_gate() -> None:
    start = pd.Timestamp("2026-08-24 07:00:00")
    frame = _anomaly_frame(start)
    frame.loc[23, "total_tbs_yhat_upper"] = float("nan")
    results = check_anomaly_ranges(frame, forecast_start=start, horizon=24)
    by_name = {result.name: result for result in results}

    target_coverage = by_name["Anomaly range target coverage"]
    assert not target_coverage.ok
    assert "Total_TBS" in target_coverage.detail


def main() -> None:
    test_complete_anomaly_horizon_passes()
    test_missing_final_anomaly_hour_fails_horizon_gate()
    test_null_target_interval_fails_target_gate()
    print("forecast input anomaly-range regression tests passed")


if __name__ == "__main__":
    main()
