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
    assert "On-call is not recommended" in blurb
    assert "23% at 4h" in blurb and "15% at 6h" in blurb and "40% at 8h" in blurb
    assert "modeled activation worsens flow" in blurb

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
    assert "L1 (1–9 PM)" in blurb
    assert "prepod, POD, or Vertical" in blurb
    assert "orange evening overlap shift (4 PM–midnight)" in blurb


def test_pod_pressure_keeps_l1_as_flexible_resource():
    facts = {
        "now": {"Total_TBS": 46, "Overflow": 9, "POD_TBS": 4, "Vertical_TBS": 14},
        "ttstr_occupancy": 213, "peak_tbs": 52, "peak_horizon": 3,
        "midnight": 31, "midnight_band": "typical", "oncall_all_low": True,
        "oncall_recommendation": "NOT INDICATED", "oncall_probabilities": {},
        "oncall_impact_summary": {}, "reassign_trigger": True, "pod_pressure": True,
    }
    blurb = blurb_wrapper.build_blurb(facts)
    assert "L1 (1–9 PM)" in blurb
    assert "prepod, POD, or Vertical" in blurb
