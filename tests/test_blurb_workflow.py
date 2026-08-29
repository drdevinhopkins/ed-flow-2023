from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "scripts" / "run_ed_flow_update.sh"


def test_active_workflow_generates_blurb_forecast():
    text = WORKFLOW.read_text()
    assert "run_step python scripts/hourly_forecast_v2_1.py" in text


def test_blurb_append_worker_runs_in_writable_scratch_directory():
    text = (Path(__file__).parents[1] / "scripts" / "automation" / "blurb_automation_wrapper.py").read_text()
    assert 'cwd=str(SCRATCH)' in text
