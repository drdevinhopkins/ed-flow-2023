import importlib.util
from pathlib import Path

WRAPPER = Path(__file__).parents[1] / "scripts" / "automation" / "blurb_automation_wrapper.py"
spec = importlib.util.spec_from_file_location("blurb_wrapper", WRAPPER)
blurb_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blurb_wrapper)

def test_oncall_sentence_gives_actionable_reason_when_need_is_mixed():
    facts = {
        "now": {"Total_TBS": 46, "Overflow": 9, "POD_TBS": 4, "Vertical_TBS": 14},
        "ttstr_occupancy": 213, "peak_tbs": 52, "peak_horizon": 3,
        "midnight": 31, "midnight_band": "typical", "oncall_all_low": False,
        "oncall_recommendation": "NOT INDICATED",
        "oncall_probabilities": {4: 0.2286, 6: 0.1519, 8: 0.4},
        "oncall_impact_summary": {"direction": "worsens", "max_adverse_stretcher": 2.5},
        "reassign_trigger": False, "pod_pressure": False,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert "please review the on-call impact summary" not in blurb
    assert "On-call is not currently needed" in blurb
    assert "calibrated" not in blurb
    assert "modeled" not in blurb

def test_oncall_metadata_uses_actionable_recommendation_and_rationale():
    facts = {
        "oncall_recommendation": "NOT INDICATED",
        "oncall_probabilities": {4: 0.2286, 6: 0.1519, 8: 0.4},
        "oncall_impact_summary": {"direction": "worsens", "max_adverse_stretcher": 2.5},
    }
    recommendation, rationale = blurb_wrapper.oncall_metadata(facts)
    assert recommendation == "NOT INDICATED"
    assert "23% at 4h" in rationale and "40% at 8h" in rationale
    assert "worsens flow" in rationale


def test_reassignment_recommends_new_vertical_patients_and_l1_flexibility():
    facts = {
        "now": {"Total_TBS": 46, "Overflow": 9, "POD_TBS": 4, "Vertical_TBS": 14},
        "ttstr_occupancy": 213, "peak_tbs": 52, "peak_horizon": 3,
        "midnight": 31, "midnight_band": "typical", "oncall_all_low": True,
        "oncall_recommendation": "NOT INDICATED", "oncall_probabilities": {},
        "oncall_impact_summary": {}, "reassign_trigger": True, "pod_pressure": False,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert "new patients in Vertical" in blurb
    assert "L1" in blurb
    assert "area under greatest pressure" in blurb
    assert "orange shift" in blurb
    assert "orange overlap shift" not in blurb


def test_pod_pressure_keeps_l1_as_flexible_resource():
    facts = {
        "now": {"Total_TBS": 46, "Overflow": 9, "POD_TBS": 4, "Vertical_TBS": 14},
        "ttstr_occupancy": 213, "peak_tbs": 52, "peak_horizon": 3,
        "midnight": 31, "midnight_band": "typical", "oncall_all_low": True,
        "oncall_recommendation": "NOT INDICATED", "oncall_probabilities": {},
        "oncall_impact_summary": {}, "reassign_trigger": True, "pod_pressure": True,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert "L1" in blurb
    assert "area under greatest pressure" in blurb


def test_weekend_reassignment_does_not_mention_unavailable_l1_shift():
    facts = {
        "data_hour": __import__("pandas").Timestamp("2026-08-29 15:00", tz="America/Montreal"),
        "now": {"Total_TBS": 46, "Overflow": 9},
        "ttstr_occupancy": 87, "peak_tbs": 52, "peak_horizon": 3,
        "midnight": 31, "midnight_band": "typical",
        "oncall_recommendation": "NOT INDICATED", "oncall_probabilities": {},
        "oncall_impact_summary": {}, "reassign_trigger": True, "pod_pressure": False,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert "L1" not in blurb


def test_blurb_is_short_and_avoids_model_language():
    facts = {
        "now": {"Total_TBS": 46, "Overflow": 9},
        "ttstr_occupancy": 87, "peak_tbs": 52, "peak_horizon": 3,
        "midnight": 31, "midnight_band": "typical",
        "oncall_recommendation": "NOT INDICATED",
        "oncall_probabilities": {4: 0.1, 6: 0.1, 8: 0.1},
        "oncall_impact_summary": {"direction": "worsens", "max_adverse_stretcher": 2.5},
        "reassign_trigger": False, "pod_pressure": False,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert len(blurb.split()) <= 90
    assert "calibrated" not in blurb
    assert "modeled" not in blurb
    assert "at 4h" not in blurb


def test_stretcher_occupancy_uses_canonical_capacity():
    facts = {
        "now": {"Total_TBS": 40, "Overflow": 0, "TTStr": 114},
        "ttstr_occupancy": 114 / 53 * 100, "peak_tbs": None, "peak_horizon": None,
        "midnight": None, "midnight_band": None,
        "oncall_recommendation": "NOT INDICATED", "oncall_probabilities": {},
        "oncall_impact_summary": {}, "reassign_trigger": False, "pod_pressure": False,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert "215%" in blurb
