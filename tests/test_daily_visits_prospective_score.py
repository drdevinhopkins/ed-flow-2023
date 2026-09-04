#!/usr/bin/env python3
"""Regression checks for prospective D+1..D+7 daily-arrival scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROSPECTIVE = ROOT / "scripts" / "evaluation" / "prospective"
if str(PROSPECTIVE) not in sys.path:
    sys.path.insert(0, str(PROSPECTIVE))

import score_daily_visits_forecast as score  # noqa: E402


def synthetic_actuals(days: int = 160) -> pd.DataFrame:
    ds = pd.date_range("2026-01-01", periods=days, freq="D")
    actual = 250 + 14 * np.sin(np.arange(days) * 2 * np.pi / 7)
    return pd.DataFrame({"ds": ds, "actual": actual})


def synthetic_snapshot(cutoff: pd.Timestamp, generated: str, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=7, freq="D")
    prediction = np.arange(7, dtype=float) + 245.0 + offset
    return pd.DataFrame(
        {
            "ds": dates,
            "daily_visits_prediction": prediction,
            "0.1": prediction - 20.0,
            "0.9": prediction + 20.0,
            "data_cutoff": cutoff,
            "horizon_day": np.arange(1, 8),
            "forecast_generated_at_utc": generated,
        }
    )


def test_normalize_snapshot_contract() -> None:
    cutoff = pd.Timestamp("2026-04-30")
    frame = synthetic_snapshot(cutoff, "2026-05-01T10:15:00Z")
    normalized = score.normalize_snapshot(frame, snapshot_name="test.csv")
    assert len(normalized) == 7
    assert normalized["horizon_day"].astype(int).tolist() == list(range(1, 8))
    assert normalized["ds"].iloc[0] == cutoff + pd.Timedelta(days=1)
    assert normalized["snapshot_name"].eq("test.csv").all()


def test_same_weekday_baseline_is_cutoff_safe() -> None:
    actuals = synthetic_actuals()
    cutoff = pd.Timestamp("2026-04-30")
    target = cutoff + pd.Timedelta(days=3)
    baseline = score.same_weekday_baseline(actuals, cutoff=cutoff, target_date=target, weeks=8)

    eligible = actuals.loc[
        (actuals["ds"] <= cutoff) & (actuals["ds"].dt.weekday == target.weekday()), "actual"
    ].tail(8)
    assert np.isclose(baseline, eligible.mean())

    # Changing post-cutoff actuals must not alter the baseline.
    modified = actuals.copy()
    modified.loc[modified["ds"] > cutoff, "actual"] = 9999.0
    baseline_modified = score.same_weekday_baseline(
        modified, cutoff=cutoff, target_date=target, weeks=8
    )
    assert np.isclose(baseline, baseline_modified)


def test_score_and_summary_by_horizon() -> None:
    actuals = synthetic_actuals()
    cutoff1 = pd.Timestamp("2026-04-20")
    cutoff2 = pd.Timestamp("2026-04-27")
    frames = [
        score.normalize_snapshot(
            synthetic_snapshot(cutoff1, "2026-04-21T10:15:00Z", offset=0.0),
            snapshot_name="one.csv",
        ),
        score.normalize_snapshot(
            synthetic_snapshot(cutoff2, "2026-04-28T10:15:00Z", offset=2.0),
            snapshot_name="two.csv",
        ),
    ]
    archive = pd.concat(frames, ignore_index=True)
    detail = score.score_archive(archive, actuals)
    summary = score.summarize_by_horizon(detail)

    assert len(detail) == 14
    assert set(detail["horizon_day"].astype(int)) == set(range(1, 8))
    assert detail["baseline_prediction"].notna().all()
    assert detail["interval_80_covered"].dtype == bool
    assert len(summary) == 8
    assert summary["horizon"].tolist() == [f"D+{day}" for day in range(1, 8)] + ["all"]
    assert summary.loc[summary["horizon"] == "D+1", "n"].iloc[0] == 2
    assert summary.loc[summary["horizon"] == "all", "n"].iloc[0] == 14
    assert not summary["evidence_ready"].any()


def test_earliest_issue_selection_logic() -> None:
    cutoff = pd.Timestamp("2026-04-30")
    early = score.normalize_snapshot(
        synthetic_snapshot(cutoff, "2026-05-01T10:15:00Z", offset=0.0),
        snapshot_name="early.csv",
    )
    late = score.normalize_snapshot(
        synthetic_snapshot(cutoff, "2026-05-01T14:00:00Z", offset=100.0),
        snapshot_name="late.csv",
    )
    archive = pd.concat([late, early], ignore_index=True)
    issue_times = archive.groupby("data_cutoff")["forecast_generated_at_utc"].transform("min")
    chosen = archive.loc[archive["forecast_generated_at_utc"].eq(issue_times)]
    assert chosen["snapshot_name"].eq("early.csv").all()
    assert len(chosen) == 7


if __name__ == "__main__":
    test_normalize_snapshot_contract()
    test_same_weekday_baseline_is_cutoff_safe()
    test_score_and_summary_by_horizon()
    test_earliest_issue_selection_logic()
    print("daily visits prospective scoring tests passed")
