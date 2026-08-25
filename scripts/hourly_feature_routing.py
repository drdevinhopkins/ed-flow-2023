"""Production routing table for validated hourly Chronos-2 feature families.

Routes come from the common-cutoff 24-hour ablation recorded under
``validation/hourly-final-ablation``. Weather-winning routes are guarded because the
historical weather ablation used revised/realized weather rather than archived
forecast-time snapshots.

Triage hallway occupancy and triage hallway TBS were added after the original six-target
ablation. They use the history-only baseline route until target-specific feature-family
validation is completed.
"""

from __future__ import annotations

FLOW_TARGETS = (
    "Total_TBS",
    "POD_TBS",
    "Vertical_TBS",
    "TTStr",
    "Overflow",
    "WAITINGADM",
    "TRG_HALLWAY1",
    "TRG_HALLWAY_TBS",
)

ROUTING_VERSION = "hourly-final-ablation-plus-triage-baseline-2026-08-25"

# Conservative production routes. Every non-baseline route below beat baseline in the
# common-cutoff ablation for the corresponding horizon band. Weather is excluded here
# until archived forecast-time weather snapshots confirm the retrospective result.
SAFE_ROUTES: dict[str, dict[str, str]] = {
    "Overflow": {
        "h01_04": "baseline",
        "h05_08": "baseline",
        "h09_12": "baseline",
        "h13_24": "baseline",
    },
    "POD_TBS": {
        "h01_04": "staffing_structure_effects",
        "h05_08": "baseline",
        "h09_12": "staffing_current",
        "h13_24": "staffing_current",
    },
    "TTStr": {
        "h01_04": "staffing_current",
        "h05_08": "staffing_current",
        "h09_12": "staffing_current",
        "h13_24": "calendar_demand",
    },
    "Total_TBS": {
        # raw+snow won, but calendar was the best non-weather option (+4.87% MAE).
        "h01_04": "calendar_demand",
        "h05_08": "staffing_current",
        "h09_12": "staffing_current",
        "h13_24": "staffing_current",
    },
    "Vertical_TBS": {
        "h01_04": "calendar_demand",
        "h05_08": "staffing_current",
        "h09_12": "staffing_current",
        "h13_24": "staffing_current",
    },
    "WAITINGADM": {
        "h01_04": "staffing_structure_effects",
        "h05_08": "staffing_structure_effects",
        "h09_12": "staffing_structure_effects",
        "h13_24": "baseline",
    },
    "TRG_HALLWAY1": {
        "h01_04": "baseline",
        "h05_08": "baseline",
        "h09_12": "baseline",
        "h13_24": "baseline",
    },
    "TRG_HALLWAY_TBS": {
        "h01_04": "baseline",
        "h05_08": "baseline",
        "h09_12": "baseline",
        "h13_24": "baseline",
    },
}

# Opt-in overrides for the two weather winners. These should stay disabled in production
# until archived forecast-time weather snapshots confirm the retrospective result.
WEATHER_OVERRIDES: dict[tuple[str, str], str] = {
    ("Overflow", "h09_12"): "weather_raw",
    ("Overflow", "h13_24"): "weather_raw",
    ("Total_TBS", "h01_04"): "weather_raw_plus_snow",
}


def horizon_band(horizon_hour: int) -> str:
    hour = int(horizon_hour)
    if 1 <= hour <= 4:
        return "h01_04"
    if 5 <= hour <= 8:
        return "h05_08"
    if 9 <= hour <= 12:
        return "h09_12"
    if 13 <= hour <= 24:
        return "h13_24"
    raise ValueError(f"horizon_hour must be in 1..24, got {horizon_hour!r}")


def scenario_for(target: str, horizon_hour: int, *, allow_weather: bool = False) -> str:
    if target not in SAFE_ROUTES:
        raise KeyError(f"No hourly feature route for target {target!r}")
    band = horizon_band(horizon_hour)
    if allow_weather:
        override = WEATHER_OVERRIDES.get((target, band))
        if override is not None:
            return override
    return SAFE_ROUTES[target][band]


def scenarios_needed(*, allow_weather: bool = False) -> set[str]:
    scenarios = {
        scenario
        for target_routes in SAFE_ROUTES.values()
        for scenario in target_routes.values()
    }
    if allow_weather:
        scenarios.update(WEATHER_OVERRIDES.values())
    return scenarios


def validate_routes() -> None:
    expected_bands = {"h01_04", "h05_08", "h09_12", "h13_24"}
    if set(SAFE_ROUTES) != set(FLOW_TARGETS):
        raise ValueError("Routing table does not cover exactly the eight forecast targets")
    for target, routes in SAFE_ROUTES.items():
        if set(routes) != expected_bands:
            raise ValueError(f"Incomplete routing bands for {target}: {sorted(routes)}")
    for target in FLOW_TARGETS:
        for hour in range(1, 25):
            scenario_for(target, hour)
            scenario_for(target, hour, allow_weather=True)


validate_routes()
