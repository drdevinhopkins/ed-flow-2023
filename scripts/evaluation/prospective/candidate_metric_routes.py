"""Prospective-only routing policy for the five candidate ED flow targets.

This module encodes the robustness-aware decisions in
``validation/candidate-metrics-ablation/DECISION.md``.  It is intentionally separate
from ``hourly_feature_routing.py`` so importing it cannot change production targets or
production routing.

Every candidate/horizon has an explicit route.  Horizon bands that did not meet the
8-common-cutoff robustness rule remain history-only ``baseline`` challengers.
"""

from __future__ import annotations

from candidate_flow_metrics import CANDIDATE_TARGETS

HORIZON_BANDS: tuple[str, ...] = ("h01_04", "h05_08", "h09_12", "h13_24")

# Robustness-aware prospective routes from DECISION.md, not raw aggregate winners.
# In particular:
# - AdmissionRequests_New h05_08 uses calendar_demand (7/8 cutoff wins) rather than
#   the slightly better aggregate staffing_structure_effects route;
# - INFLOW_AMBULANCES is routed only at h01_04;
# - Inflow_Total is routed only at h01_04;
# - Workup_Delay_Burden h13_24 remains baseline because only 3/8 cutoffs improved.
PROSPECTIVE_CANDIDATE_ROUTES: dict[str, dict[str, str]] = {
    "Inflow_Total": {
        "h01_04": "calendar_demand",
        "h05_08": "baseline",
        "h09_12": "baseline",
        "h13_24": "baseline",
    },
    "INFLOW_STRETCHER": {
        "h01_04": "staffing_current",
        "h05_08": "calendar_demand",
        "h09_12": "staffing_current",
        "h13_24": "baseline",
    },
    "INFLOW_AMBULANCES": {
        "h01_04": "staffing_structure_effects",
        "h05_08": "baseline",
        "h09_12": "baseline",
        "h13_24": "baseline",
    },
    "AdmissionRequests_New": {
        "h01_04": "baseline",
        "h05_08": "calendar_demand",
        "h09_12": "baseline",
        "h13_24": "staffing_structure_effects",
    },
    "Workup_Delay_Burden": {
        "h01_04": "staffing_current",
        "h05_08": "calendar_demand",
        "h09_12": "staffing_current",
        "h13_24": "baseline",
    },
}


def horizon_band(horizon_hour: int) -> str:
    """Return the standard 24-hour horizon band for an integer lead hour."""

    if 1 <= horizon_hour <= 4:
        return "h01_04"
    if 5 <= horizon_hour <= 8:
        return "h05_08"
    if 9 <= horizon_hour <= 12:
        return "h09_12"
    if 13 <= horizon_hour <= 24:
        return "h13_24"
    raise ValueError(f"horizon_hour must be in 1..24; got {horizon_hour}")


def scenario_for(target: str, horizon_hour: int) -> str:
    """Return the pre-registered prospective scenario for one candidate forecast row."""

    if target not in PROSPECTIVE_CANDIDATE_ROUTES:
        raise KeyError(f"Unknown prospective candidate target: {target}")
    return PROSPECTIVE_CANDIDATE_ROUTES[target][horizon_band(horizon_hour)]


def scenarios_needed() -> tuple[str, ...]:
    """Return the minimal scenario set needed to generate all candidate challengers."""

    scenarios = {
        scenario
        for target_routes in PROSPECTIVE_CANDIDATE_ROUTES.values()
        for scenario in target_routes.values()
    }
    return tuple(sorted(scenarios))


def validate_policy() -> None:
    """Fail fast if the prospective policy is incomplete or contains unsupported routes."""

    expected_targets = set(CANDIDATE_TARGETS)
    actual_targets = set(PROSPECTIVE_CANDIDATE_ROUTES)
    if actual_targets != expected_targets:
        raise RuntimeError(
            "Candidate route target mismatch: "
            f"missing={sorted(expected_targets - actual_targets)} "
            f"extra={sorted(actual_targets - expected_targets)}"
        )

    allowed = {"baseline", "calendar_demand", "staffing_current", "staffing_structure_effects"}
    for target, routes in PROSPECTIVE_CANDIDATE_ROUTES.items():
        if set(routes) != set(HORIZON_BANDS):
            raise RuntimeError(f"Incomplete horizon routing for {target}: {sorted(routes)}")
        unsupported = set(routes.values()) - allowed
        if unsupported:
            raise RuntimeError(f"Unsupported prospective route(s) for {target}: {sorted(unsupported)}")


validate_policy()
