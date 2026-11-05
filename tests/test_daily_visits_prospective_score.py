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
    chosen = score.select_earliest_issues(archive)
    assert chosen["snapshot_name"].eq("early.csv").all()
    assert len(chosen) == 7


def test_unverified_outcomes_and_baselines_cannot_enter_comparisons() -> None:
    actuals = synthetic_actuals()
    cutoff = pd.Timestamp("2026-04-30")
    target = cutoff + pd.Timedelta(days=1)
    baseline_day = cutoff - pd.Timedelta(days=6)  # same weekday as D+1
    actuals.loc[actuals.ds.isin([target, baseline_day]), "actual"] = np.nan
    actuals.attrs["target_quality_version"] = score.QUALITY_VERSION
    archive = score.normalize_snapshot(synthetic_snapshot(cutoff, "2026-05-01T10:15:00Z"),
                                       snapshot_name="original.csv")
    detail = score.score_archive(archive, actuals)
    assert len(detail) == 6 and not detail.ds.eq(target).any()
    expected = actuals.loc[(actuals.ds <= cutoff) & actuals.actual.notna()
                          & actuals.ds.dt.weekday.eq(target.weekday()), "actual"].tail(8).mean()
    assert np.isclose(score.same_weekday_baseline(actuals, cutoff=cutoff, target_date=target), expected)
    retained = archive.set_index("ds").loc[detail.ds]
    assert np.array_equal(detail.daily_visits_prediction, retained.daily_visits_prediction)
    assert detail.actual_quality_version.eq(score.QUALITY_VERSION).all()


def test_corrected_actuals_do_not_relabel_legacy_forecasts_as_validated() -> None:
    actuals = synthetic_actuals()
    actuals.attrs["target_quality_version"] = score.QUALITY_VERSION
    frames = [score.normalize_snapshot(synthetic_snapshot(cutoff, str(cutoff + pd.Timedelta(days=1)) + "Z"),
                                       snapshot_name=f"{cutoff.date()}.csv")
              for cutoff in pd.date_range("2026-03-01", periods=29)]
    archive = pd.concat(frames, ignore_index=True)
    legacy_detail = score.score_archive(archive, actuals)
    legacy = score.summarize_by_horizon(legacy_detail)
    assert legacy.collection_ready.all() and legacy.actual_quality_verified.all()
    assert not legacy.evidence_ready.any()
    archive["target_quality_version"] = score.QUALITY_VERSION
    verified_detail = score.score_archive(archive, actuals)
    assert score.summarize_by_horizon(verified_detail).evidence_ready.all()
    # A short new collection cannot borrow the legacy collection's size/span.
    mixed = pd.concat([legacy_detail, verified_detail.loc[verified_detail.data_cutoff >= "2026-03-25"]])
    grouped = score.summarize_by_quality_version(mixed)
    assert not grouped.evidence_ready.any()
    corrected = grouped.loc[grouped.forecast_quality_version.eq(score.QUALITY_VERSION)]
    assert corrected.n_issue_dates.eq(5).all()


def test_local_replay_cannot_publish_or_overwrite_source_files(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    import pytest
    daily = tmp_path / "daily.csv"
    hourly = tmp_path / "hourly.csv"
    daily.write_text("source sentinel")
    args = SimpleNamespace(daily_csv=daily, hourly_csv=hourly, archive_dir=tmp_path,
                           now=None, no_dropbox_output=False, detail_output=tmp_path / "detail.csv",
                           summary_output=tmp_path / "summary.csv", quality_summary_output=tmp_path / "quality.csv")
    monkeypatch.setattr(score, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="no-dropbox-output"):
        score.main()
    args.no_dropbox_output = True
    args.detail_output = daily
    with pytest.raises(ValueError, match="overwrite"):
        score.main()
    assert daily.read_text() == "source sentinel"
    args.detail_output = tmp_path / "detail.csv"
    args.summary_output = args.detail_output
    with pytest.raises(ValueError, match="distinct"):
        score.main()


def test_local_replay_cannot_overwrite_immutable_snapshot(monkeypatch, tmp_path) -> None:
    from types import SimpleNamespace
    import pytest
    snapshot = tmp_path / "daily_visits_forecast_original.csv"
    snapshot.write_text("immutable sentinel")
    args = SimpleNamespace(daily_csv=tmp_path / "daily.csv", hourly_csv=tmp_path / "hourly.csv",
                           archive_dir=tmp_path, now=None, no_dropbox_output=True,
                           detail_output=snapshot, summary_output=tmp_path / "summary.csv",
                           quality_summary_output=tmp_path / "quality.csv")
    monkeypatch.setattr(score, "parse_args", lambda: args)
    with pytest.raises(ValueError, match="snapshots"):
        score.main()
    assert snapshot.read_text() == "immutable sentinel"


def scheduled_snapshot(cutoff, time, slot, name, offset=0):
    frame = synthetic_snapshot(cutoff, time, offset=offset)
    frame["forecast_issue_slot"] = slot
    frame["forecast_issue_day_local"] = (cutoff + pd.Timedelta(days=1)).date().isoformat()
    frame["target_quality_version"] = score.QUALITY_VERSION
    return score.normalize_snapshot(frame, snapshot_name=name)


def test_scheduled_issues_keep_morning_evening_and_drop_later_retries():
    cutoff = pd.Timestamp("2026-04-30")
    morning = scheduled_snapshot(cutoff, "2026-05-01T10:15Z", "morning", "am.csv")
    evening = scheduled_snapshot(cutoff, "2026-05-01T22:15Z", "evening", "pm.csv", 5)
    retry = scheduled_snapshot(cutoff, "2026-05-01T23:00Z", "evening", "retry.csv", 999)
    legacy = score.normalize_snapshot(synthetic_snapshot(cutoff, "2026-05-01T09:00Z"), snapshot_name="legacy.csv")
    archive = pd.concat([retry, legacy, evening, morning], ignore_index=True)
    chosen = score.select_scheduled_issues(archive)
    assert len(chosen) == 14
    assert set(chosen.snapshot_name) == {"am.csv", "pm.csv"}
    assert score.select_earliest_issues(archive).snapshot_name.eq("legacy.csv").all()
    actuals = synthetic_actuals()
    actuals.attrs["target_quality_version"] = score.QUALITY_VERSION
    detail = score.score_archive(chosen, actuals)
    comparison = score.compare_refreshes(detail)
    assert len(comparison) == 7 and comparison.n_pairs.eq(1).all()
    assert np.allclose(comparison.mean_absolute_revision, 5)
    assert np.allclose(comparison.morning_mean_lead_hours - comparison.evening_mean_lead_hours, 12)
    assert comparison.calendar_days_ahead.tolist() == list(range(7))
    by_slot = score.summarize_by_issue_slot(detail)
    assert len(by_slot) == 16
    assert by_slot.n_issue_dates.eq(1).all()
    assert not by_slot.evidence_ready.any()


def test_issue_metadata_rejects_backdating_or_wrong_window():
    import pytest
    cutoff = pd.Timestamp("2026-04-30")
    frame = synthetic_snapshot(cutoff, "2026-05-01T22:15Z")
    frame["forecast_issue_slot"] = "morning"
    frame["forecast_issue_day_local"] = "2026-05-01"
    with pytest.raises(ValueError, match="outside"):
        score.normalize_snapshot(frame, snapshot_name="backdated.csv")
    frame["forecast_issue_slot"] = "evening"
    frame["forecast_issue_day_local"] = "2026-04-30"
    with pytest.raises(ValueError, match="issue date"):
        score.normalize_snapshot(frame, snapshot_name="wrong_day.csv")


def test_refresh_comparison_uses_only_matched_targets_and_same_quality_version():
    cutoff = pd.Timestamp("2026-04-30")
    morning = scheduled_snapshot(cutoff, "2026-05-01T10:15Z", "morning", "am.csv")
    evening = scheduled_snapshot(cutoff, "2026-05-01T22:15Z", "evening", "pm.csv")
    evening.loc[evening.horizon_day.eq(3), "target_quality_version"] = "different_policy"
    actuals = synthetic_actuals()
    detail = score.score_archive(pd.concat([morning, evening]), actuals)
    comparison = score.compare_refreshes(detail)
    assert 3 not in comparison.horizon_day.tolist()
    assert comparison.n_pairs.eq(1).all()
    assert score.compare_refreshes(detail.loc[detail.forecast_issue_slot.eq("morning")]).empty


def test_legacy_only_archive_has_empty_scheduled_scores():
    frame = score.normalize_snapshot(synthetic_snapshot(pd.Timestamp("2026-04-30"),
                                     "2026-05-01T10:15Z"), snapshot_name="legacy.csv")
    detail = score.score_archive(score.select_scheduled_issues(frame), synthetic_actuals())
    assert detail.empty
    assert score.compare_refreshes(detail).empty
    assert score.summarize_by_issue_slot(detail).n.eq(0).all()


if __name__ == "__main__":
    test_normalize_snapshot_contract()
    test_same_weekday_baseline_is_cutoff_safe()
    test_score_and_summary_by_horizon()
    test_earliest_issue_selection_logic()
    print("daily visits prospective scoring tests passed")


def test_local_replay_writes_both_slot_scores_and_keeps_immutable_inputs(monkeypatch, tmp_path):
    from types import SimpleNamespace
    cutoff = pd.Timestamp('2026-04-30')
    archive = tmp_path / 'archive'
    archive.mkdir()
    for slot, stamp in [('morning', '2026-05-01T10:15Z'), ('evening', '2026-05-01T22:15Z')]:
        frame = scheduled_snapshot(cutoff, stamp, slot, slot + '.csv')
        frame.to_csv(archive / f'daily_visits_forecast_{slot}.csv', index=False)
    originals = {path: path.read_bytes() for path in archive.iterdir()}
    hourly = pd.DataFrame({'ds': pd.date_range('2026-04-01', '2026-05-07T23:00', freq='h'),
                           'Inflow_Total': 2.0})
    daily = hourly.groupby(hourly.ds.dt.normalize()).Inflow_Total.sum().rename('Daily_Inflow_Total').reset_index()
    hourly.to_csv(tmp_path / 'hourly.csv', index=False)
    daily.to_csv(tmp_path / 'daily.csv', index=False)
    outputs = {key: tmp_path / (key + '.csv') for key in [
        'detail_output', 'summary_output', 'quality_summary_output', 'issue_detail_output',
        'issue_summary_output', 'refresh_comparison_output']}
    args = SimpleNamespace(daily_csv=tmp_path / 'daily.csv', hourly_csv=tmp_path / 'hourly.csv',
                           archive_dir=archive, now='2026-05-10T18:00Z', no_dropbox_output=True,
                           **outputs)
    monkeypatch.setattr(score, 'parse_args', lambda: args)
    score.main()
    assert len(pd.read_csv(outputs['detail_output'])) == 7
    assert len(pd.read_csv(outputs['issue_detail_output'])) == 14
    assert pd.read_csv(outputs['refresh_comparison_output']).n_pairs.eq(1).all()
    assert all(path.read_bytes() == data for path, data in originals.items())
