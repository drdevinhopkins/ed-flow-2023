import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hourly_feature_routing import (  # noqa: E402
    FLOW_TARGETS,
    horizon_band,
    scenario_for,
    scenarios_needed,
    validate_routes,
)


def test_horizon_bands():
    assert horizon_band(1) == "h01_04"
    assert horizon_band(4) == "h01_04"
    assert horizon_band(5) == "h05_08"
    assert horizon_band(8) == "h05_08"
    assert horizon_band(9) == "h09_12"
    assert horizon_band(12) == "h09_12"
    assert horizon_band(13) == "h13_24"
    assert horizon_band(24) == "h13_24"


def test_safe_routes_match_validated_production_policy():
    assert scenario_for("Total_TBS", 2) == "calendar_demand"
    assert scenario_for("Total_TBS", 7) == "staffing_current"
    assert scenario_for("POD_TBS", 2) == "staffing_structure_effects"
    assert scenario_for("POD_TBS", 6) == "baseline"
    assert scenario_for("TTStr", 3) == "staffing_current"
    assert scenario_for("TTStr", 18) == "calendar_demand"
    assert scenario_for("Vertical_TBS", 3) == "calendar_demand"
    assert scenario_for("Vertical_TBS", 18) == "staffing_current"
    assert scenario_for("WAITINGADM", 10) == "staffing_structure_effects"
    assert scenario_for("WAITINGADM", 18) == "baseline"
    assert scenario_for("Overflow", 18) == "baseline"


def test_weather_routes_are_opt_in():
    assert scenario_for("Overflow", 10) == "baseline"
    assert scenario_for("Overflow", 10, allow_weather=True) == "weather_raw"
    assert scenario_for("Overflow", 20, allow_weather=True) == "weather_raw"
    assert scenario_for("Total_TBS", 2) == "calendar_demand"
    assert scenario_for("Total_TBS", 2, allow_weather=True) == "weather_raw_plus_snow"


def test_all_targets_have_all_hours_and_needed_scenarios():
    validate_routes()
    for target in FLOW_TARGETS:
        for hour in range(1, 25):
            assert scenario_for(target, hour)
    assert scenarios_needed() == {
        "baseline",
        "calendar_demand",
        "staffing_current",
        "staffing_structure_effects",
    }
    assert scenarios_needed(allow_weather=True) == {
        "baseline",
        "calendar_demand",
        "staffing_current",
        "staffing_structure_effects",
        "weather_raw",
        "weather_raw_plus_snow",
    }


if __name__ == "__main__":
    test_horizon_bands()
    test_safe_routes_match_validated_production_policy()
    test_weather_routes_are_opt_in()
    test_all_targets_have_all_hours_and_needed_scenarios()
    print("hourly feature routing tests passed")
