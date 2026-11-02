import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("facts_dst", ROOT / "scripts/automation/compute_blurb_facts.py")
facts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(facts)
spec = importlib.util.spec_from_file_location("wrapper_dst", ROOT / "scripts/automation/blurb_automation_wrapper.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


@pytest.mark.parametrize("origin", ["2026-10-31 01:00", "2026-11-01 01:00", "2026-11-01 20:00"])
def test_blurb_uses_unique_local_clock_hours_through_fallback(tmp_path, origin):
    origin = pd.Timestamp(origin)
    times = pd.date_range(origin - pd.Timedelta(hours=2), periods=27, freq="h")
    pd.DataFrame({"ds": times[:3]}).to_csv(tmp_path / "current.csv", index=False)
    rows = []
    for target in facts.TARGETS:
        for index, time in enumerate(times):
            horizon = index - 2
            rows.append({"ds": time, "forecast_origin": origin, "target_name": target,
                         "row_type": "forecast" if horizon > 0 else "observed",
                         "horizon_hour": horizon, "actual": 20, "forecast": 21,
                         "actual_anomaly": "no", "forecast_anomaly": "no"})
    pd.DataFrame(rows).to_csv(tmp_path / "forecast-v2.1.csv", index=False)
    (tmp_path / "blurb_reference_stats.json").write_text(json.dumps({}))
    result = facts.compute(tmp_path, data_hour=origin)
    assert result["ready"], result["failures"]
    assert result["staffing_review"]["forecast_hours_available_next_6h"] == 6
    assert not facts.compute(tmp_path, data_hour=origin + pd.Timedelta(hours=1))["ready"]


def test_ambiguous_forecast_origin_is_a_clock_label(monkeypatch):
    monkeypatch.setattr(wrapper, "download", lambda *_: b"forecast_origin\n2026-11-01 01:00:00\n")
    assert wrapper.forecast_origin(None) == pd.Timestamp("2026-11-01 01:00")
