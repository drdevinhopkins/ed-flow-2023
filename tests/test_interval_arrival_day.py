"""Report-day migration safeguards, including unchanged legacy defaults."""
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from arrival_day_policy import (
    INTERVAL_QUALITY_VERSION, INTERVAL_TARGET_VERSION,
    build_interval_daily_outputs, interval_day_flow, interval_view,
)
from daily_arrival_quality import build_daily_inflow_outputs, target_history_fingerprint, verify_explanation_context
from forecast_intraday_daily_inflow import DataQualityError, validate_live_flow
from intraday_day_completion_model import build_snapshots, fit_completion_curve, TOTAL_TBS_COMPONENTS

spec = importlib.util.spec_from_file_location("interval_runner", Path(__file__).resolve().parents[1]/
                                             "scripts/experiments/arrival_day/run_interval_day.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
NOW = pd.Timestamp("2026-02-04T19:00:00Z")


def hourly():
    frame = pd.DataFrame({"ds": pd.date_range("2026-02-01", "2026-02-04T14:00", freq="h"),
                          "Inflow_Total": 2})
    frame.loc[frame.ds.eq(pd.Timestamp("2026-02-02")), "Inflow_Total"] = 9
    for column in TOTAL_TBS_COMPONENTS:
        frame[column] = 1
    return frame


def test_closing_hour_changes_arrival_day_not_endpoint_or_raw_state():
    source = hourly()
    before = source.copy(deep=True)
    daily, quality = build_interval_daily_outputs(source, now=NOW)
    legacy, _ = build_daily_inflow_outputs(source, now=NOW)
    assert daily.set_index("ds").loc["2026-02-01", "Daily_Inflow_Total"] == 55
    assert legacy.set_index("ds").loc["2026-02-01", "Daily_Inflow_Total"] == 48
    flow = interval_day_flow(source, now=NOW)
    closing = flow.loc[flow.ds.eq(pd.Timestamp("2026-02-02"))].iloc[0]
    assert closing.day == pd.Timestamp("2026-02-01")
    assert closing._arrival_report_hour == 24
    assert quality.source_hourly_cutoff.eq("2026-02-04T14:00:00").all()
    assert daily.attrs["target_quality_version"] == INTERVAL_QUALITY_VERSION
    pd.testing.assert_frame_equal(source, before)
    pd.testing.assert_series_equal(flow.ds, source.ds, check_names=False)
    for column in TOTAL_TBS_COMPONENTS:
        assert flow[column].equals(source[column])


@pytest.mark.parametrize("defect", ["missing_closing", "duplicate", "nan", "negative"])
def test_unverified_intervals_never_become_numeric_targets_or_training_days(defect):
    source = hourly()
    closing = source.ds.eq(pd.Timestamp("2026-02-02"))
    if defect == "missing_closing":
        source = source.loc[~closing]
    elif defect == "duplicate":
        source = pd.concat([source, source.loc[closing]], ignore_index=True)
    else:
        source.loc[closing, "Inflow_Total"] = np.nan if defect == "nan" else -1
    daily, _ = build_interval_daily_outputs(source, now=NOW)
    assert pd.isna(daily.set_index("ds").loc["2026-02-01", "Daily_Inflow_Total"])
    flow = interval_day_flow(source, now=NOW)
    assert not flow.loc[flow.day.eq(pd.Timestamp("2026-02-01")), "is_complete_day"].any()


@pytest.mark.parametrize("day", ["2026-03-08", "2026-11-01"])
def test_dst_stays_unverified_even_with_24_naive_intervals(day):
    source = pd.DataFrame({"ds": pd.date_range(pd.Timestamp(day)+pd.Timedelta(hours=1), periods=24, freq="h"),
                          "Inflow_Total": 2})
    _, quality = build_interval_daily_outputs(source, now=pd.Timestamp(day, tz="UTC")+pd.Timedelta(days=2))
    assert quality.quality_status.eq("dst_requires_verification").all()
    assert not quality.audit_eligible.any()


def test_future_endpoint_not_hidden_by_interval_start_shift():
    with pytest.raises(ValueError, match="future endpoints"):
        interval_view(hourly(), now=NOW-pd.Timedelta(minutes=30))


def test_current_report_day_needs_hours_one_through_cutoff_not_prior_midnight():
    source = hourly().loc[lambda f: f.ds.le(pd.Timestamp("2026-02-02T11:00"))]
    flow = interval_day_flow(source, now="2026-02-02T16:30:00Z")
    _, current = validate_live_flow(flow, now="2026-02-02T16:30:00Z", interval_day=True)
    assert current.Inflow_Total.sum() == 22
    assert current.ds.dt.hour.tolist() == list(range(1, 12))
    broken = flow.loc[~flow.ds.eq(pd.Timestamp("2026-02-02T03:00"))]
    with pytest.raises(DataQualityError, match="missing or out-of-order"):
        validate_live_flow(broken, now="2026-02-02T16:30:00Z", interval_day=True)


def test_closing_midnight_is_suppressed_not_assigned_to_new_live_day():
    flow = interval_day_flow(hourly().iloc[:25], now="2026-02-02T05:10:00Z")
    with pytest.raises(DataQualityError, match="outside"):
        validate_live_flow(flow, now="2026-02-02T05:10:00Z", interval_day=True)


def test_snapshot_progress_and_curve_close_at_hour_24_without_future_features():
    flow = interval_day_flow(hourly(), now=NOW)
    snapshots = build_snapshots(flow, calendar_mode="basic")
    day = snapshots.loc[snapshots.day.eq(pd.Timestamp("2026-02-01"))]
    assert day.cutoff_hour.tolist() == list(range(1, 25))
    assert day.cumulative_arrivals.iloc[0] == 2
    assert day.cumulative_arrivals.iloc[-1] == 55
    assert day.remaining_arrivals.iloc[-1] == 0
    assert day.day_progress_fraction.iloc[-1] == 1
    assert day.reports_remaining.iloc[-1] == 0
    curve = fit_completion_curve(snapshots)
    assert curve.index.tolist() == list(range(1, 25))
    assert 0 < curve.loc[1, "expected_fraction"] < 1
    assert curve.loc[24, "expected_fraction"] == 1
    changed = hourly()
    changed.loc[changed.ds.eq(pd.Timestamp("2026-02-02")), "Inflow_Total"] = 100
    other = build_snapshots(interval_day_flow(changed, now=NOW), calendar_mode="basic")
    cutoff = pd.Timestamp("2026-02-01T11:00")
    # Labels change; the observed prefix and cutoff features must not see closing arrivals.
    assert other.loc[other.ds.eq(cutoff), "cumulative_arrivals"].iat[0] == day.loc[day.ds.eq(cutoff), "cumulative_arrivals"].iat[0]


def test_explanation_version_and_fingerprint_must_match():
    history = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=3), "daily_visits": [55, 48, 48]})
    formatted = pd.DataFrame({"history_days": [3], "target_quality_version": [INTERVAL_QUALITY_VERSION],
                              "target_history_sha256": [target_history_fingerprint(history)]})
    verify_explanation_context(formatted, history, quality_version=INTERVAL_QUALITY_VERSION)
    with pytest.raises(ValueError, match="predates"):
        verify_explanation_context(formatted, history)  # legacy default must not accept v2
    history.loc[0, "daily_visits"] = 48
    with pytest.raises(ValueError, match="differ"):
        verify_explanation_context(formatted, history, quality_version=INTERVAL_QUALITY_VERSION)


def test_scoring_excludes_legacy_before_earliest_issue_selection():
    daily, _ = build_interval_daily_outputs(hourly(), now=NOW)
    actuals = daily.rename(columns={"Daily_Inflow_Total": "actual"})
    rows = []
    for version, time, prediction in [("legacy_unverified", "2026-02-01T12:00:00Z", 999),
                                      (INTERVAL_QUALITY_VERSION, "2026-02-01T13:00:00Z", 48)]:
        rows.append({"ds": pd.Timestamp("2026-02-02"), "data_cutoff": pd.Timestamp("2026-02-01"),
                     "forecast_generated_at_utc": pd.Timestamp(time), "snapshot_name": version,
                     "horizon_day": 1, "daily_visits_prediction": prediction, "0.1": 40, "0.9": 60,
                     "target_quality_version": version,
                     "target_definition_version": INTERVAL_TARGET_VERSION if prediction == 48 else None})
    detail = runner.score_interval_daily(pd.DataFrame(rows), actuals)
    assert detail.daily_visits_prediction.tolist() == [48]
    assert detail.actual_quality_version.eq(INTERVAL_QUALITY_VERSION).all()
    assert detail.actual.tolist() == [48]
    with pytest.raises(ValueError, match="v2 actuals"):
        runner.score_interval_daily(pd.DataFrame(rows), pd.DataFrame({"ds": actuals.ds, "actual": actuals.actual}))


def test_interval_zip_runner_scores_original_values_against_corrected_actuals(tmp_path):
    from test_intraday_prospective_audit import forecast
    artifacts, runs = [], []
    for hour in range(11, 19):
        number = hour
        row = forecast(forecast_day="2026-02-01", cutoff_hour=hour,
                       cutoff_ds_local=f"2026-02-01T{hour:02d}:00:00",
                       generated_at_utc=f"2026-02-01T{hour+5:02d}:20:00Z",
                       target_definition_version=INTERVAL_TARGET_VERSION,
                       target_quality_version=INTERVAL_QUALITY_VERSION, model_version="interval-v2")
        payload = {"status": "forecast_written", "forecast": row,
                   "generated_at_utc": row["generated_at_utc"]}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("intraday_status_interval_v2.json", json.dumps(payload))
            archive.writestr("intraday_forecast_interval_v2.csv", pd.DataFrame([row]).to_csv(index=False))
        data = buffer.getvalue()
        (tmp_path/f"{number}.zip").write_bytes(data)
        artifacts.append({"id": number, "workflow_run": {"id": number, "head_branch": "main", "head_sha": "abc"},
                          "created_at": row["generated_at_utc"], "expired": False,
                          "digest": "sha256:"+hashlib.sha256(data).hexdigest()})
        runs.append({"id": number})
    (tmp_path/"artifacts.json").write_text(json.dumps(artifacts))
    (tmp_path/"runs.json").write_text(json.dumps(runs))
    output = tmp_path/"scores"
    output.mkdir()
    result = runner.run_intraday_score(SimpleNamespace(
        artifacts=tmp_path/"artifacts.json", runs=tmp_path/"runs.json", zip_dir=tmp_path,
        now=NOW.isoformat(), output_dir=output), hourly())
    scores = pd.read_csv(output/"intraday_scores_interval_v2.csv")
    assert scores.actual.tolist() == [55]*8  # legacy stored-calendar total is 48
    assert scores.predicted_total.tolist() == [48]*8  # never refit or relabel issues
    assert scores.target_quality_version.eq(INTERVAL_QUALITY_VERSION).all()
    assert result["scored_rows"] == 8
    assert result["gates"][0]["complete_operational_days"] == 1
    assert result["operational_promotion"] is False


def test_no_overwrites_and_no_remote_sources():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source = root/"source.csv"
        source.write_text("ds,Inflow_Total\n")
        args = SimpleNamespace(hourly=source, weather_csv=None, artifacts=None, runs=None,
                               archive_dir=None, zip_dir=None, output_dir=root/"new", now=NOW.isoformat())
        runner.validate_paths(args)
        args.max_iter = 0
        with pytest.raises(ValueError, match="positive"):
            runner.validate_paths(args)
        args.max_iter = 200
        args.output_dir.mkdir()
        (args.output_dir/"daily_forecast_interval_v2.csv").write_text("original")
        with pytest.raises(ValueError, match="fresh output"):
            runner.validate_paths(args)
        assert (args.output_dir/"daily_forecast_interval_v2.csv").read_text() == "original"
        args.hourly = Path("https://example.com/source.csv")
        with pytest.raises(ValueError, match="local files"):
            runner.validate_paths(args)


def test_fresh_interval_model_fit_cpu_smoke_has_new_version_and_correct_observed_count():
    from forecast_intraday_daily_inflow import build_intraday_forecast
    source = pd.DataFrame({"ds": pd.date_range("2026-01-01", "2026-03-12T11:00", freq="h"),
                           "Inflow_Total": 2})
    source.loc[source.ds.eq(pd.Timestamp("2026-03-12")), "Inflow_Total"] = 9
    for column in TOTAL_TBS_COMPONENTS:
        source[column] = 1
    weather = pd.DataFrame({"ds": source.ds, "temperature_2m": 10+np.sin(source.ds.dt.hour)})
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        source.to_csv(root/"flow.csv", index=False)
        weather.to_csv(root/"weather.csv", index=False)
        row = build_intraday_forecast(flow_source=root/"flow.csv", weather_source=root/"weather.csv",
            generated_at=pd.Timestamp("2026-03-12T15:30:00Z"), interval_day=True,
            min_train_days=60, max_iter=2)
    assert row["target_definition_version"] == INTERVAL_TARGET_VERSION
    assert row["target_quality_version"] == INTERVAL_QUALITY_VERSION
    assert row["model_version"].endswith("research-iter2")
    assert row["observed_arrivals"] == 22
    assert 22 <= row["p10_total"] <= row["predicted_total"] <= row["p90_total"]
    assert row["expected_additional_arrivals"] == row["predicted_total"]-22


def test_daily_runner_explains_the_same_corrected_history(monkeypatch, tmp_path):
    from chronos import BaseChronosPipeline
    from test_daily_visit_forecast import synthetic_hourly_weather

    calls = []

    class Pipeline:
        def predict_df(self, history, **kwargs):
            calls.append(history.copy())
            future = kwargs["future_df"]
            return pd.DataFrame({"ds": future.ds, "target_name": "daily_visits",
                                 "predictions": 48., "0.1": 40., "0.5": 48., "0.9": 60.})

    monkeypatch.setattr(BaseChronosPipeline, "from_pretrained", lambda *a, **k: Pipeline())
    source = pd.DataFrame({"ds": pd.date_range("2025-01-01", "2025-02-14T11:00", freq="h"),
                           "Inflow_Total": 2})
    source.loc[source.ds.eq(pd.Timestamp("2025-02-01")), "Inflow_Total"] = 9
    daily, _ = build_interval_daily_outputs(source, now="2025-02-14T16:30:00Z")
    weather_path = tmp_path/"weather.csv"
    synthetic_hourly_weather(start="2024-12-15", days=80).to_csv(weather_path, index=False)
    args = SimpleNamespace(weather_csv=weather_path, now="2025-02-14T16:30:00Z",
                           context_days=1095, min_history_days=28, output_dir=tmp_path)
    result = runner.run_daily(args, daily)
    assert result["forecast_rows"] == 7
    assert len(calls) > 1  # inference and actual scenario explanation calls
    for history in calls:
        assert history.loc[history.ds.eq(pd.Timestamp("2025-01-31")), "daily_visits"].iat[0] == 55
        assert target_history_fingerprint(history) == result["target_history_sha256"]
    for name in ["daily_forecast_interval_v2.csv", "daily_explained_interval_v2.csv"]:
        frame = pd.read_csv(tmp_path/name)
        assert frame.target_quality_version.eq(INTERVAL_QUALITY_VERSION).all()
        assert frame.target_definition_version.eq(INTERVAL_TARGET_VERSION).all()
        assert frame.target_history_sha256.eq(result["target_history_sha256"]).all()
