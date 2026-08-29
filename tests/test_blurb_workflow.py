from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "scripts" / "run_ed_flow_update.sh"


def test_active_workflow_generates_blurb_forecast():
    text = WORKFLOW.read_text()
    assert "run_step python scripts/hourly_forecast_v2_1.py" in text
