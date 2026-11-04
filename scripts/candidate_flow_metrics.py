"""Candidate hourly ED flow targets used for pre-production validation only.

Nothing in this module changes the production forecast target set. It provides
reproducible derivations for candidate targets that can be evaluated with the same
Chronos-2 common-cutoff framework as the existing operational targets.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from hourly_feature_routing import FLOW_TARGETS as PRODUCTION_FLOW_TARGETS

CANDIDATE_TARGETS: tuple[str, ...] = (
    "Inflow_Total",
    "INFLOW_STRETCHER",
    "INFLOW_AMBULANCES",
    "AdmissionRequests_New",
    "Workup_Delay_Burden",
)

ALL_EXPERIMENT_TARGETS: tuple[str, ...] = (*PRODUCTION_FLOW_TARGETS, *CANDIDATE_TARGETS)

TOTAL_TBS_COMPONENTS: tuple[str, ...] = (
    "TRG_HALLWAY_TBS",
    "POD_GREEN_TBS",
    "POD_YELLOW_TBS",
    "POD_ORANGE_TBS",
    "RAZ_TBS",
    "AMBVERTTBS",
    "QTrack_TBS",
    "Garage_TBS",
)
POD_TBS_COMPONENTS: tuple[str, ...] = (
    "TRG_HALLWAY_TBS",
    "POD_GREEN_TBS",
    "POD_YELLOW_TBS",
    "POD_ORANGE_TBS",
)
VERTICAL_TBS_COMPONENTS: tuple[str, ...] = (
    "RAZ_TBS",
    "AMBVERTTBS",
    "QTrack_TBS",
    "Garage_TBS",
)
WORKUP_DELAY_COMPONENTS: tuple[str, ...] = (
    "POD_CONS_MORE2H",
    "POD_IMCONS_MORE4H",
    "POD_XRAY_MORE2H",
    "POD_CT_MORE2H",
    "RAZ_CONS_MORE2H",
    "RAZ_IMCONS_MORE4H",
    "RAZ_XRAY_MORE2H",
    "RAZ_CT_MORE2H1",
)

COUNT_LIKE_TARGETS: frozenset[str] = frozenset(CANDIDATE_TARGETS)


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.loc[:, columns].copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _derive_sum(frame: pd.DataFrame, target: str, components: Sequence[str]) -> None:
    missing = [column for column in components if column not in frame.columns]
    if missing:
        raise ValueError(f"Cannot derive {target}; missing: {', '.join(missing)}")
    values = _numeric(frame, components)
    frame[target] = values.sum(axis=1, min_count=len(components))


def add_production_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Recreate the eight current production targets without importing model code."""

    out = raw.copy()
    _derive_sum(out, "Total_TBS", TOTAL_TBS_COMPONENTS)
    _derive_sum(out, "POD_TBS", POD_TBS_COMPONENTS)
    _derive_sum(out, "Vertical_TBS", VERTICAL_TBS_COMPONENTS)
    _derive_sum(out, "Overflow", ("POST_POD1", "TRG_HALLWAY1"))

    direct = ("TTStr", "WAITINGADM", "TRG_HALLWAY1", "TRG_HALLWAY_TBS")
    missing = [column for column in direct if column not in out.columns]
    if missing:
        raise ValueError(f"Missing direct production target(s): {', '.join(missing)}")
    for column in direct:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def reset_aware_increment(counter: pd.Series, ds: pd.Series) -> pd.Series:
    """Convert a cumulative counter into hourly increments without creating reset spikes.

    A negative difference is interpreted as a counter reset and the post-reset counter
    value becomes that hour's increment. Differences across gaps longer than one hour are
    left missing so they are not falsely attributed to a single hour.
    """

    values = pd.to_numeric(counter, errors="coerce")
    stamps = pd.to_datetime(ds, errors="coerce")
    delta = values.diff()
    elapsed = stamps.diff()

    increment = delta.where(delta >= 0, values)
    increment = increment.where(delta.notna())
    increment = increment.where(elapsed.eq(pd.Timedelta(hours=1)))
    return increment.clip(lower=0)


def add_candidate_metrics(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the five candidate targets derived and numeric."""

    out = raw.copy()
    required_direct = ["Inflow_Total", "INFLOW_STRETCHER", "INFLOW_AMBULANCES", "CUM_ADMREQ"]
    missing_direct = [column for column in required_direct if column not in out.columns]
    missing_workup = [column for column in WORKUP_DELAY_COMPONENTS if column not in out.columns]
    missing = [*missing_direct, *missing_workup]
    if missing:
        raise ValueError(f"Missing candidate-metric source columns: {', '.join(missing)}")

    for column in ["Inflow_Total", "INFLOW_STRETCHER", "INFLOW_AMBULANCES"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["AdmissionRequests_New"] = reset_aware_increment(out["CUM_ADMREQ"], out["ds"])

    workup = _numeric(out, WORKUP_DELAY_COMPONENTS)
    # This is deliberately a burden score, not a unique-patient count: one patient can
    # contribute to more than one delayed process bucket.
    out["Workup_Delay_Burden"] = workup.sum(axis=1, min_count=len(WORKUP_DELAY_COMPONENTS))
    return out


def build_experiment_flow(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a regular hourly 13-target frame for common-cutoff candidate validation."""

    frame = raw.copy()
    frame["ds"] = pd.to_datetime(frame["ds"], format="mixed", errors="coerce")
    if getattr(frame["ds"].dt, "tz", None) is not None:
        frame["ds"] = frame["ds"].dt.tz_convert("America/Montreal").dt.tz_localize(None)
    frame["ds"] = frame["ds"].dt.floor("h")
    frame = frame.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    derived = add_production_metrics(frame)
    derived = add_candidate_metrics(derived)
    missing = [target for target in ALL_EXPERIMENT_TARGETS if target not in derived.columns]
    if missing:
        raise ValueError(f"Missing experiment target(s): {', '.join(missing)}")

    flow = derived[["ds", *ALL_EXPERIMENT_TARGETS]].copy()
    for target in ALL_EXPERIMENT_TARGETS:
        flow[target] = pd.to_numeric(flow[target], errors="coerce")

    index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    flow = flow.set_index("ds").reindex(index).reset_index()

    # Existing operational targets are state-like measures and keep production's forward
    # fill convention. Candidate hourly counts/burdens are not forward-filled across data
    # gaps; short isolated gaps are interpolated only to preserve regular model frequency.
    for target in PRODUCTION_FLOW_TARGETS:
        flow[target] = flow[target].ffill()
    for target in CANDIDATE_TARGETS:
        flow[target] = flow[target].interpolate(limit=1, limit_direction="both")

    return flow


def trailing_complete_history_window(
    flow: pd.DataFrame,
    cutoff: pd.Timestamp,
    *,
    targets: Sequence[str] = ALL_EXPERIMENT_TARGETS,
    max_history_days: int = 365,
    min_history_days: int = 28,
) -> tuple[int, pd.Timestamp, int]:
    """Choose the longest trailing complete target window for prospective forecasting.

    Historical ablations deliberately selected cutoffs with a complete 365-day target
    window. A live prospective stream can contain an older isolated gap, which should not
    invalidate today's forecast if a sufficiently long complete trailing segment remains.
    The selected window is capped at ``max_history_days`` and must contain at least
    ``min_history_days`` complete days. A recent gap therefore still fails closed.
    """

    cutoff = pd.Timestamp(cutoff).floor("h")
    eligible = flow.loc[flow["ds"].le(cutoff), ["ds", *targets]].copy()
    if eligible.empty:
        raise ValueError(f"No candidate history available at cutoff {cutoff}")

    incomplete = eligible[list(targets)].isna().any(axis=1)
    last_bad = eligible.loc[incomplete, "ds"].max() if incomplete.any() else pd.NaT
    if pd.isna(last_bad):
        complete_start = pd.Timestamp(eligible["ds"].min())
    else:
        complete_start = pd.Timestamp(last_bad) + pd.Timedelta(hours=1)

    complete_hours = int(eligible["ds"].between(complete_start, cutoff).sum())
    complete_days = complete_hours // 24
    history_days = min(int(max_history_days), complete_days)
    if history_days < min_history_days:
        raise ValueError(
            "Insufficient trailing complete candidate history: "
            f"{complete_hours}h ({complete_days} full days), minimum={min_history_days}d; "
            f"last_incomplete={last_bad}, incomplete_rows_before_cutoff={int(incomplete.sum())}"
        )

    history_start = cutoff - pd.Timedelta(days=history_days) + pd.Timedelta(hours=1)
    selected = eligible.loc[eligible["ds"].between(history_start, cutoff)]
    if len(selected) != history_days * 24 or selected[list(targets)].isna().any().any():
        raise RuntimeError("Selected prospective candidate history window is not complete")
    return history_days, history_start, int(incomplete.sum())


def candidate_quality_summary(flow: pd.DataFrame) -> pd.DataFrame:
    """Compact completeness/range diagnostics used before launching expensive backtests."""

    rows: list[dict[str, object]] = []
    for target in CANDIDATE_TARGETS:
        series = pd.to_numeric(flow[target], errors="coerce")
        rows.append(
            {
                "target_name": target,
                "n": int(series.size),
                "n_nonnull": int(series.notna().sum()),
                "missing_pct": float(series.isna().mean() * 100.0),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "p95": float(series.quantile(0.95)),
                "max": float(series.max()),
                "zero_pct": float(series.eq(0).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)
