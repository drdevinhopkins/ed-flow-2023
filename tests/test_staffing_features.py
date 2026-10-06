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
    prepare_shifts,
    expand_shift_hours,
    build_current_staffing_features,
    build_legacy_staffing_identity,
)
from staffing_roles import l1_l2_schedule_context, resolve_hourly_roles


def test_l1_l2_effective_date_and_concurrent_hours() -> None:
    old = pd.Timestamp("2026-09-30")
    new = pd.Timestamp("2026-10-01")
    friday = pd.Timestamp("2026-10-02")
    shifts = pd.DataFrame([
        _shift("OldL1", old + pd.Timedelta(hours=12), old + pd.Timedelta(hours=21), "L1"),
        _shift("OldL2", old + pd.Timedelta(hours=13), old + pd.Timedelta(hours=17), "L2"),
        _shift("NewL1", new + pd.Timedelta(hours=12), new + pd.Timedelta(hours=21), "L1"),
        _shift("NewL2", new + pd.Timedelta(hours=13), new + pd.Timedelta(hours=18), "L2"),
        _shift("FridayL1", friday + pd.Timedelta(hours=12), friday + pd.Timedelta(hours=21), "L1"),
    ])
    frames = build_schedule_feature_frames(shifts)
    s = frames.structure.set_index("ds")
    identity = frames.identity.set_index("ds")
    assert identity.loc[old + pd.Timedelta(hours=14), "physician__OldL1Doctor"] == "overlap"
    assert identity.loc[old + pd.Timedelta(hours=14), "physician__OldL2Doctor"] == "overlap"
    assert identity.loc[new + pd.Timedelta(hours=12), "physician__NewL1Doctor"] == "flexible"
    assert identity.loc[new + pd.Timedelta(hours=13), "physician__NewL1Doctor"] == "vertical"
    assert identity.loc[new + pd.Timedelta(hours=13), "physician__NewL2Doctor"] == "pod"
    assert identity.loc[new + pd.Timedelta(hours=18), "physician__NewL1Doctor"] == "flexible"
    assert identity.loc[friday + pd.Timedelta(hours=14), "physician__FridayL1Doctor"] == "flexible"
    assert s.loc[new + pd.Timedelta(hours=13), "n_total_scheduled"] == 2
    assert s.loc[new + pd.Timedelta(hours=13), "n_l1_vertical"] == 1
    assert s.loc[new + pd.Timedelta(hours=13), "n_l2_pod"] == 1
    assert s.loc[old + pd.Timedelta(hours=14), "n_flexible"] == 0
    assert s.loc[new + pd.Timedelta(hours=12), "delta_n_vertical_next_1h"] == 1
    assert s.loc[new + pd.Timedelta(hours=13), "n_shift_starts_pod"] == 1
    assert s.loc[new + pd.Timedelta(hours=13), "n_shift_starts_vertical"] == 0
    assert s.loc[new + pd.Timedelta(hours=18), "n_shift_ends_pod"] == 1
    expanded = expand_shift_hours(prepare_shifts(shifts))
    assert set(expanded.shift_short_name) == {"L1", "L2"}
    assert expanded.loc[expanded.physician_id.eq("OldL2Doctor"), "role_assignment_rule"].eq("legacy").all()
    pd.testing.assert_frame_equal(build_current_staffing_features(shifts), frames.current)
    legacy = build_legacy_staffing_identity(shifts).add_prefix("physician__").reset_index()
    legacy.columns.name = frames.identity.columns.name
    pd.testing.assert_frame_equal(legacy, frames.identity)


def test_l2_alone_and_unexpected_friday_do_not_impute_l1() -> None:
    shifts = pd.DataFrame([
        _shift("Alone", pd.Timestamp("2026-10-05 13:00"), pd.Timestamp("2026-10-05 21:00"), "L2"),
        _shift("Friday", pd.Timestamp("2026-10-09 13:00"), pd.Timestamp("2026-10-09 21:00"), "L2"),
    ])
    current = build_current_staffing_features(shifts).set_index("ds")
    assert current.loc["2026-10-05 15:00", "n_pod"] == 1
    assert current.loc["2026-10-05 15:00", "n_l1_vertical"] == 0
    assert current.loc["2026-10-09 15:00", "n_overlap"] == 1


def test_role_rules_use_montreal_weekday_and_exclude_end_hour() -> None:
    shifts = pd.DataFrame([
        _shift("L1", pd.Timestamp("2026-10-01 16:00Z"), pd.Timestamp("2026-10-02 01:00Z"), "L1"),
        _shift("L2", pd.Timestamp("2026-10-01 17:00Z"), pd.Timestamp("2026-10-02 01:00Z"), "L2"),
    ])
    current = build_current_staffing_features(shifts).set_index("ds")
    assert current.loc["2026-10-01 20:00", "n_vertical"] == 1
    context = l1_l2_schedule_context(shifts, pd.Timestamp("2026-10-02 00:00Z"))
    assert context["l1_l2_split"]
    context = l1_l2_schedule_context(shifts, pd.Timestamp("2026-10-02 01:00Z"))
    assert context["l1_role"] == "absent"
    assert context["l2_role"] == "absent"


def test_effect_scores_follow_resolved_physician_roles() -> None:
    day = pd.Timestamp("2026-10-05")
    shifts = pd.DataFrame([
        _shift("One", day + pd.Timedelta(hours=12), day + pd.Timedelta(hours=21), "L1"),
        _shift("Two", day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=21), "L2"),
    ])
    profiles = pd.DataFrame({"physician_id": ["OneDoctor", "OneDoctor", "TwoDoctor"],
                             "shift_type": ["flexible", "vertical", "pod"],
                             "effect__Total_TBS": [1., 2., 3.]})
    scores = build_effect_score_features(shifts, profiles, ["Total_TBS"]).set_index("ds")
    assert scores.loc[day + pd.Timedelta(hours=12), "staff_effect__Total_TBS_sum"] == 1
    assert scores.loc[day + pd.Timedelta(hours=13), "staff_effect__Total_TBS_sum"] == 5


def test_oncall_builders_share_role_semantics_without_loading_models() -> None:
    # Run the real lightweight entry-point functions without importing GPU/model dependencies.
    import ast
    day = pd.Timestamp("2026-10-05")
    shifts = pd.DataFrame([
        _shift("One", day + pd.Timedelta(hours=12), day + pd.Timedelta(hours=21), "L1"),
        _shift("Two", day + pd.Timedelta(hours=13), day + pd.Timedelta(hours=21), "L2"),
    ])
    for name in ("forecast_oncall_impact.py", "forecast_oncall_probability.py"):
        tree = ast.parse((Path(__file__).parents[1] / "scripts" / name).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_staffing_features")
        namespace = {"pd": pd, "build_current_staffing_features": build_current_staffing_features}
        exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), namespace)
        pd.testing.assert_frame_equal(namespace["build_staffing_features"](shifts), build_current_staffing_features(shifts))


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
    test_l1_l2_effective_date_and_concurrent_hours()
    test_l2_alone_and_unexpected_friday_do_not_impute_l1()
    test_role_rules_use_montreal_weekday_and_exclude_end_hour()
    test_effect_scores_follow_resolved_physician_roles()
    test_oncall_builders_share_role_semantics_without_loading_models()
    test_structural_handoff_features()
    test_physician_effect_direction_and_leakage_boundary()
    test_unseen_identity_category_is_sanitized()
    print("staffing feature tests passed")


if __name__ == "__main__":
    main()
