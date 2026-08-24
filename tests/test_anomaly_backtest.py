#!/usr/bin/env python3
"""Unit tests for leakage-safe anomaly threshold calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from backtest_anomaly_detection import (
    add_empirical_thresholds,
    parse_training_windows,
    summarize_detectors,
)


def _prediction_frame(periods: int = 48) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01 00:00:00", periods=periods, freq="h")
    residual = np.linspace(-2.0, 2.0, periods)
    yhat = np.full(periods, 10.0)
    return pd.DataFrame(
        {
            "ds": ds,
            "training_window": "365d",
            "target": "Total_TBS",
            "actual": yhat + residual,
            "yhat": yhat,
            "yhat_lower": 5.0,
            "yhat_upper": 15.0,
            "residual": residual,
            "hour_of_week": ds.dayofweek * 24 + ds.hour,
        }
    )


def test_training_window_parser_supports_full_history() -> None:
    assert parse_training_windows("180,365,all,365") == (180, 365, None)


def test_empirical_threshold_does_not_use_current_residual() -> None:
    baseline = _prediction_frame()
    shocked = baseline.copy()
    shocked.loc[shocked.index[-1], "actual"] = 1000.0
    shocked.loc[shocked.index[-1], "residual"] = 990.0

    kwargs = dict(calibration_days=30, min_global_samples=8, min_context_samples=2)
    calibrated_baseline = add_empirical_thresholds(baseline, **kwargs)
    calibrated_shocked = add_empirical_thresholds(shocked, **kwargs)

    last = baseline.index[-1]
    assert calibrated_baseline.loc[last, "residual_q95_global"] == calibrated_shocked.loc[
        last, "residual_q95_global"
    ]
    assert calibrated_baseline.loc[last, "residual_q99_global"] == calibrated_shocked.loc[
        last, "residual_q99_global"
    ]


def test_sparse_context_falls_back_to_global_threshold() -> None:
    frame = _prediction_frame()
    calibrated = add_empirical_thresholds(
        frame,
        calibration_days=30,
        min_global_samples=8,
        min_context_samples=100,
    )
    eligible = calibrated.dropna(subset=["empirical_upper_global_95"])
    assert not eligible.empty
    assert eligible["residual_q95_how"].isna().all()
    assert np.allclose(
        eligible["empirical_upper_how_95"],
        eligible["empirical_upper_global_95"],
    )


def test_summary_counts_persistent_alert_episodes() -> None:
    ds = pd.date_range("2026-02-01 00:00:00", periods=8, freq="h")
    actual = [9, 12, 13, 9, 14, 15, 16, 9]
    scored = pd.DataFrame(
        {
            "ds": ds,
            "training_window": "365d",
            "target": "Total_TBS",
            "actual": actual,
            "yhat": 10.0,
            "yhat_upper": 11.0,
            "empirical_upper_global_95": 11.0,
            "empirical_upper_how_95": 11.0,
            "empirical_upper_global_99": 20.0,
        }
    )
    summary = summarize_detectors(scored)
    nominal = summary.loc[summary["detector"].eq("prophet_nominal_95")].iloc[0]

    assert nominal["n_alerts"] == 5
    assert nominal["episode_count"] == 2
    assert nominal["max_episode_hours"] == 3
    assert nominal["mean_episode_hours"] == 2.5


def main() -> None:
    test_training_window_parser_supports_full_history()
    test_empirical_threshold_does_not_use_current_residual()
    test_sparse_context_falls_back_to_global_threshold()
    test_summary_counts_persistent_alert_episodes()
    print("anomaly backtest tests passed")


if __name__ == "__main__":
    main()
