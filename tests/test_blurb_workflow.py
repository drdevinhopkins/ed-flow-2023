from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / "scripts" / "run_ed_flow_update.sh"


def test_active_workflow_generates_blurb_forecast():
    text = WORKFLOW.read_text()
    assert "run_step python scripts/hourly_forecast_v2_1.py" in text


def test_active_workflow_generates_intraday_arrival_forecast_before_blurb():
    text = WORKFLOW.read_text()
    intraday = text.index("scripts/forecast_intraday_daily_inflow.py")
    blurb = text.index("scripts/automation/blurb_automation_wrapper.py")
    assert intraday < blurb
    assert "--weather-csv weather.csv" in text
    assert "--upload-dropbox" in text[intraday:blurb]


def test_daily_arrival_forecast_is_guarded_and_published_from_host():
    text = WORKFLOW.read_text()
    assert "run_daily_arrival_forecast()" in text
    assert "TZ=America/Montreal date +%H" in text
    assert "local_hour" in text and "06" in text
    assert "forecast_daily_visits_from_daily.py" in text
    assert "explain_daily_visits_forecast.py" in text
    assert "build_daily_arrival_outlook.py" in text


def test_blurb_append_worker_runs_in_writable_scratch_directory():
    text = (Path(__file__).parents[1] / "scripts" / "automation" / "blurb_automation_wrapper.py").read_text()
    assert 'cwd=str(SCRATCH)' in text


def test_publisher_has_durable_per_hour_outbox():
    text = (Path(__file__).parents[1] / "scripts" / "automation" / "blurb_automation_wrapper.py").read_text()
    assert "blurb_outbox" in text
    assert "request_path_for" in text
