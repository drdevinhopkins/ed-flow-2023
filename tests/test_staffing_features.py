#!/usr/bin/env python3
"""Lightweight tests for staffing feature engineering; runnable without pytest."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from staffing_features import (  # noqa: E402
    build_effect_score_features,
    build_schedule_feature_frames,
    fit_physician_effect_profiles,
    sanitize_identity_for_cutoff,
)


def _shift(first: str, start: pd.Timestamp, end: pd.Timestamp, code: str = "A1") -> dict[str, object]:
    return {
        "first_name": first,
        "last_name": "Doctor",
        "shift_start": start,
        "shift_end": end,
        "shift_short_name": code,
    }


def test_structural_handoff_features() -> None:
    day = pd.Timestamp("2026-01-05")
    shifts = pd.DataFrame(
        [
            _shift("Alice", day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=16), "A1"),
            _shift("Bob", day + pd.Timedelta(hours=16), day + pd.Timedelta(hours=23), "B1"),
            _shift("Flow", day + pd.Timedelta(hours=12), day + pd.Timedelta(hours=20), "V1"),
        ]
    )
    frames = build_schedule_feature_frames(shifts)
    structure = frames.structure.set_index("ds")
    identity = frames.identity.set_index("ds")

    assert structure.loc[day + pd.Timedelta(hours=15), "n_pod"] == 1
    assert structure.loc[day + pd.Timedelta(hours=16), "n_shift_ends_pod"] == 1
    assert structure.loc[day + pd.Timedelta(hours=16), "n_shift_starts_vertical"] == 1
    assert structure.loc[day + pd.Timedelta(hours=15), "n_last_1h"] >= 1
    assert structure.loc[day + pd.Timedelta(hours=16), "n_team_changes_prev_1h"] >= 2
    assert identity.loc[day + pd.Timedelta(hours=15), "physician__AliceDoctor"] == "pod"
    assert identity.loc[day + pd.Timedelta(hours=16), "physician__AliceDoctor"] == "NotWorking"
    assert identity.loc[day + pd.Timedelta(hours=16), "physician__BobDoctor"] == "vertical"


def synthetic_physician_signal(days: int = 56) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp("2025-01-06")
    hours = pd.date_range(start, periods=days * 24, freq="h")
    shifts: list[dict[str, object]] = []
    delta = np.zeros(len(hours), dtype=float)

    for day_idx in range(days):
        day = start + pd.Timedelta(days=day_idx)
        physician = "Alice" if day_idx % 2 == 0 else "Bob"
        shifts.append(_shift(physician, day + pd.Timedelta(hours=8), day + pd.Timedelta(hours=16), "A1"))
        mask = (hours >= day + pd.Timedelta(hours=8)) & (hours < day + pd.Timedelta(hours=16))
        delta[mask] += -1.75 if physician == "Alice" else 1.75

    delta += 0.15 * np.sin(2 * np.pi * hours.hour.to_numpy() / 24)
    target = 40.0 + np.cumsum(delta)
    flow = pd.DataFrame({"ds": hours, "Total_TBS": target})
    return flow, pd.DataFrame(shifts)


def test_physician_effect_direction_and_leakage_boundary() -> None:
    flow, shifts = synthetic_physician_signal()
    cutoff = flow.loc[len(flow) - 24 * 10, "ds"]

    profile = fit_physician_effect_profiles(
        flow,
        shifts,
        ["Total_TBS"],
        profile_end=cutoff,
        min_active_hours=12,
        shrinkage_hours=12,
    )
    effects = profile.set_index("physician_id")["effect__Total_TBS"]
    assert effects["AliceDoctor"] < 0
    assert effects["BobDoctor"] > 0
    assert effects["AliceDoctor"] < effects["BobDoctor"]

    changed = flow.copy()
    changed.loc[changed["ds"] > cutoff, "Total_TBS"] += np.linspace(
        0, 10000, (changed["ds"] > cutoff).sum()
    )
    profile_changed = fit_physician_effect_profiles(
        changed,
        shifts,
        ["Total_TBS"],
        profile_end=cutoff,
        min_active_hours=12,
        shrinkage_hours=12,
    )
    left = profile.sort_values(["physician_id", "shift_type"]).reset_index(drop=True)
    right = profile_changed.sort_values(["physician_id", "shift_type"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)

    score_features = build_effect_score_features(shifts, profile, ["Total_TBS"])
    assert "staff_effect__Total_TBS_sum" in score_features
    alice_hour = pd.Timestamp("2025-01-06 10:00")
    bob_hour = pd.Timestamp("2025-01-07 10:00")
    indexed = score_features.set_index("ds")
    assert indexed.loc[alice_hour, "staff_effect__Total_TBS_sum"] < 0
    assert indexed.loc[bob_hour, "staff_effect__Total_TBS_sum"] > 0


def test_unseen_identity_category_is_sanitized() -> None:
    history = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01", periods=3, freq="h"),
            "physician__NewDoctor": ["NotWorking"] * 3,
        }
    )
    future = pd.DataFrame(
        {
            "ds": pd.date_range("2026-01-01 03:00", periods=2, freq="h"),
            "physician__NewDoctor": ["pod", "NotWorking"],
        }
    )
    _, safe_future = sanitize_identity_for_cutoff(history, future)
    assert safe_future["physician__NewDoctor"].tolist() == ["NotWorking", "NotWorking"]


def main() -> None:
    test_structural_handoff_features()
    test_physician_effect_direction_and_leakage_boundary()
    test_unseen_identity_category_is_sanitized()
    print("staffing feature tests passed")


if __name__ == "__main__":
    main()
