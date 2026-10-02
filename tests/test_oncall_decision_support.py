import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from oncall_labels import add_activation_targets, merge_activation_labels


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


facts_module = load("review_facts", "scripts/automation/compute_blurb_facts.py")
wrapper = load("review_wrapper", "scripts/automation/blurb_automation_wrapper.py")


def test_unknown_labels_are_not_negative_targets():
    hours = pd.date_range("2026-05-15", periods=12, freq="h")
    hourly = pd.DataFrame({"ds": hours})
    labels = pd.DataFrame({"ds": hours[:7], "oncall-used-for-busy": [0, 0, 1, 0, 0, 0, 0]})
    merged = merge_activation_labels(hourly, labels)
    assert merged.loc[7:, "oncall_active"].isna().all()
    targets = add_activation_targets(merged, (4, 6, 8))
    assert targets.loc[0, "oncall_within_4h"] == 1
    assert pd.isna(targets.loc[3, "oncall_within_4h"])
    assert targets["oncall_within_8h"].isna().all()


def test_targets_use_clock_hours_and_reject_conflicting_labels():
    hours = pd.to_datetime(["2026-05-15 00:00", "2026-05-15 01:00", "2026-05-15 03:00"])
    df = pd.DataFrame({"ds": hours, "oncall_active": [0, 0, 1]})
    assert pd.isna(add_activation_targets(df, (2,)).loc[0, "oncall_within_2h"])
    with pytest.raises(ValueError, match="Conflicting"):
        merge_activation_labels(df.drop(columns="oncall_active"),
                                pd.DataFrame({"ds": [hours[0], hours[0]], "oncall_active": [0, 1]}))


def input_bundle(tmp_path, tbs=(39, 42, 56), hour="2026-09-29 16:00", models=True):
    origin = pd.Timestamp(hour, tz="America/Montreal")
    pd.DataFrame({"ds": [origin]}).to_csv(tmp_path / "current.csv", index=False)
    rows = []
    values = {"Total_TBS": tbs[-1], "POD_TBS": 15, "Vertical_TBS": 41,
              "TTStr": 137, "Overflow": 34, "WAITINGADM": 55,
              "TRG_HALLWAY1": 20, "TRG_HALLWAY_TBS": 4}
    for target, value in values.items():
        for step in range(-2, 25):
            ds = origin + pd.Timedelta(hours=step)
            rows.append({"forecast_origin": origin, "ds": ds, "target_name": target,
                         "horizon_hour": step, "row_type": "observed" if step <= 0 else "forecast",
                         "actual": (tbs[step + 2] if target == "Total_TBS" else value) if step <= 0 else None,
                         "forecast": (30 if step <= 8 else 60) if step > 0 else None,
                         "actual_anomaly": "no", "forecast_anomaly": "no"})
    pd.DataFrame(rows).to_csv(tmp_path / "forecast-v2.1.csv", index=False)
    (tmp_path / "blurb_reference_stats.json").write_text("{}")
    if models:
        pd.DataFrame({"ds": [origin] * 3, "horizon_hours": [4, 6, 8],
                      "calibrated_probability": [.07, .13, .11]}).to_csv(tmp_path / "oncall_need_probability.csv", index=False)
        pd.DataFrame({"forecast_origin": [origin], "target_name": ["stretcher_occupancy"],
                      "estimated_improvement": [-2]}).to_csv(tmp_path / "oncall_impact_summary.csv", index=False)
    return origin


def test_september_29_review_cannot_be_vetoed_by_reassuring_models(tmp_path):
    input_bundle(tmp_path)
    facts = facts_module.compute(tmp_path)
    assert facts["ready"]
    assert facts["oncall_recommendation"] == "STAFFING REVIEW REQUIRED"
    assert facts["staffing_review"]["forecast_hours_ge_45_next_6h"] == 0
    assert facts["oncall_impact_summary"]["direction"] == "worsens"
    assert facts["oncall_impact_summary"]["max_adverse_stretcher"] == pytest.approx(1.06)
    blurb = wrapper.build_blurb(facts)
    assert "Staffing review is needed now" in blurb
    assert "peak appears to have passed" not in blurb
    assert "No staffing change" not in blurb


def test_review_survives_missing_model_files(tmp_path):
    input_bundle(tmp_path, models=False)
    facts = facts_module.compute(tmp_path)
    assert facts["ready"]
    assert facts["oncall_recommendation"] == "STAFFING REVIEW REQUIRED"
    assert facts["model_warnings"]


def test_corrupt_optional_models_and_context_do_not_hide_review(tmp_path):
    input_bundle(tmp_path)
    (tmp_path / "oncall_need_probability.csv").write_text("invalid,column\n1,2\n")
    (tmp_path / "oncall_impact_summary.csv").write_text("forecast_origin,estimated_improvement\ninvalid,-2\n")
    (tmp_path / "oncall_operational_context.json").write_text("broken json")
    facts = facts_module.compute(tmp_path)
    assert facts["ready"]
    assert facts["oncall_recommendation"] == "STAFFING REVIEW REQUIRED"
    assert facts["staffing_review"]["availability"] == "unknown"


def test_missing_current_canonical_value_prevents_publication(tmp_path):
    input_bundle(tmp_path)
    fc = pd.read_csv(tmp_path / "forecast-v2.1.csv")
    fc.loc[(fc.target_name == "Total_TBS") & (fc.horizon_hour == 0), "actual"] = None
    fc.to_csv(tmp_path / "forecast-v2.1.csv", index=False)
    assert not facts_module.compute(tmp_path)["ready"]


def test_low_workload_does_not_assert_oncall_is_unnecessary(tmp_path):
    input_bundle(tmp_path, tbs=(20, 21, 22))
    facts = facts_module.compute(tmp_path)
    assert facts["oncall_recommendation"] == "NO ESCALATION DETECTED"
    assert "not currently needed" not in wrapper.build_blurb(facts)


@pytest.mark.parametrize("values,expected", [([44, 45, 45], False), ([45, 45, 45], True),
                                            ([29, 35, 40], True), ([39, 40, 49], True)])
def test_sustained_and_worsening_backlog(values, expected):
    hour = pd.Timestamp("2026-09-29 15:00", tz="America/Montreal")
    history = pd.DataFrame({"actual": values}, index=pd.date_range(hour-pd.Timedelta(hours=2), hour, freq="h"))
    future = pd.DataFrame({"forecast": []}, index=pd.DatetimeIndex([], tz="America/Montreal"))
    assert facts_module.staffing_review(history, future, hour)["required"] == expected


def test_gaps_and_future_observations_do_not_create_sustained_trigger():
    hour = pd.Timestamp("2026-09-29 15:00", tz="America/Montreal")
    history = pd.DataFrame({"actual": [45, 45, 100]},
                           index=[hour-pd.Timedelta(hours=2), hour, hour+pd.Timedelta(hours=1)])
    future = pd.DataFrame({"forecast": []}, index=pd.DatetimeIndex([], tz="America/Montreal"))
    assert not facts_module.staffing_review(history, future, hour)["required"]


def test_availability_and_late_hour_preserve_workload_review(tmp_path):
    origin = input_bundle(tmp_path, hour="2026-09-29 22:00")
    import json
    (tmp_path / "oncall_operational_context.json").write_text(json.dumps(
        {"data_hour": origin.isoformat(), "availability": "unavailable"}))
    facts = facts_module.compute(tmp_path)
    blurb = wrapper.build_blurb(facts)
    assert "on-call is unavailable" in blurb
    assert "next-morning duties" in blurb
    assert facts["staffing_review"]["required"]


def test_next_day_peak_does_not_hide_remaining_day_rise(tmp_path):
    input_bundle(tmp_path, tbs=(20, 21, 22))
    f = pd.read_csv(tmp_path / "forecast-v2.1.csv")
    f.loc[(f.target_name == "Total_TBS") & (f.horizon_hour == 3), "forecast"] = 40
    f.to_csv(tmp_path / "forecast-v2.1.csv", index=False)
    blurb = wrapper.build_blurb(facts_module.compute(tmp_path))
    assert "about 40 TBS in roughly 3 hours" in blurb
    assert "peak appears to have passed" not in blurb


def test_forecast_duration_triggers_review_only_with_complete_six_hours(tmp_path):
    input_bundle(tmp_path, tbs=(20, 21, 22))
    f = pd.read_csv(tmp_path / "forecast-v2.1.csv")
    mask = (f.target_name == "Total_TBS") & f.horizon_hour.isin([1, 2, 3])
    f.loc[mask, "forecast"] = 45
    f.to_csv(tmp_path / "forecast-v2.1.csv", index=False)
    assert facts_module.compute(tmp_path)["staffing_review"]["required"]
    f = f[~((f.target_name == "Total_TBS") & (f.horizon_hour == 6))]
    f.to_csv(tmp_path / "forecast-v2.1.csv", index=False)
    facts = facts_module.compute(tmp_path)
    assert not facts["staffing_review"]["required"]
    assert facts["staffing_review"]["forecast_hours_available_next_6h"] == 5


def test_schedule_presence_does_not_confirm_availability():
    hour = pd.Timestamp("2026-09-29 18:00", tz="America/Montreal")
    shifts = pd.DataFrame({"user_id": [1, 1, 2, 1], "shift_short_name": ["OC1", "E1", "E2", "D1"],
                           "shift_start": ["2026-09-29 08:00", "2026-09-29 16:00", "2026-09-29 16:00", "2026-09-30 08:00"],
                           "shift_end": ["2026-09-30 01:00", "2026-09-30 00:00", "2026-09-30 00:00", "2026-09-30 17:00"]})
    context = facts_module.schedule_context(shifts, hour)
    assert context["availability"] == "unknown"
    assert context["scheduled_working_physicians_now"] == 2
    assert context["oncall_also_scheduled_regular_shift"]
    assert context["oncall_has_next_morning_shift"]


def test_llm_cannot_omit_staffing_review_or_late_qualifications(tmp_path):
    llm = load("review_llm", "scripts/automation/llm_blurb_automation_wrapper.py")
    input_bundle(tmp_path, hour="2026-09-29 22:00")
    facts = facts_module.compute(tmp_path)
    with pytest.raises(ValueError, match="review"):
        llm.validate_blurb("On-call is not currently needed.", facts, wrapper.build_blurb(facts))
    with pytest.raises(ValueError, match="qualifications"):
        llm.validate_blurb("Staffing review is needed now.", facts, wrapper.build_blurb(facts))
    llm.validate_blurb(wrapper.staffing_review_sentence(facts), facts, wrapper.build_blurb(facts))
