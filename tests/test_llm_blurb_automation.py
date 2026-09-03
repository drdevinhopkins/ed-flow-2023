from pathlib import Path
import importlib.util
import pandas as pd

WRAPPER = Path(__file__).parents[1] / "scripts" / "automation" / "llm_blurb_automation_wrapper.py"
spec = importlib.util.spec_from_file_location("llm_blurb_wrapper", WRAPPER)
llm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm)


def facts():
    return {
        "data_hour": pd.Timestamp("2026-09-03 20:00", tz="America/Montreal"),
        "now": {"Total_TBS": 40, "Overflow": 25},
        "ttstr_occupancy": 150,
        "peak_tbs": 45,
        "peak_horizon": 2,
        "peak_time": pd.Timestamp("2026-09-03 22:00", tz="America/Montreal"),
        "midnight": 31,
        "midnight_band": "typical",
        "anomalies": [{"target": "POD_TBS", "status": "current", "value": 17}],
        "oncall_recommendation": "NOT INDICATED",
        "reassign_trigger": False,
        "pod_pressure": False,
    }


def test_llm_output_is_separate_file():
    assert llm.LLM_OUTPUT_PATH.endswith("hourly_forecast_blurbs_llm.csv")


def test_validation_preserves_required_anomaly():
    llm.validate_blurb(
        "POD TBS is currently above its anomaly threshold (17).",
        facts(),
        "POD TBS is currently above its anomaly threshold (17).",
    )


def test_validation_rejects_unsupported_number():
    try:
        llm.validate_blurb("POD TBS is currently above its anomaly threshold (19).", facts(), "17")
    except ValueError as exc:
        assert "unsupported numeric" in str(exc)
    else:
        raise AssertionError("unsupported numeric claim was accepted")
