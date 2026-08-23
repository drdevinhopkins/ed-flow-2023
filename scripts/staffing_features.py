#!/usr/bin/env python3
"""Leakage-safe staffing feature engineering for hourly ED Chronos-2 forecasts.

The module deliberately separates three ideas:

1. *Structure*: how many physicians are working, in which roles, where handoffs occur,
   how deep into their shifts the active team is, and how coverage changes next hour.
2. *Identity*: the specific physician scheduled in each hour, represented as stable
   per-physician categorical role/``NotWorking`` columns.
3. *Historical flow fingerprints*: shrunk, target-specific associations between a
   physician-role pair and the next-hour change in a flow target. These are fitted only
   on observations available at a supplied cutoff and then frozen for the forecast.

The fingerprint is an association, not a causal productivity estimate. It is residualized
for hour-of-week and broad staffing intensity before physician-level averaging so it is
less likely to simply rediscover that particular doctors tend to work nights or busier
coverage patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TS_COL = "ds"
LOCAL_TZ = "America/Montreal"

SHIFT_TYPES = {
    "W1": "flow", "X1": "pod", "X3": "pod", "X4": "vertical", "X2": "vertical",
    "WOC1": "oncall", "WOC2": "oncall", "WOC3": "oncall", "X5": "pod",
    "W3": "overlap", "Y1": "pod", "Y3": "pod", "Y4": "vertical",
    "Y2": "vertical", "Y5": "pod", "Z1": "night", "Z2": "night", "D1": "pod",
    "R1": "pod", "P1": "vertical", "D2": "vertical", "OC1": "oncall",
    "OC2": "oncall", "V1": "flow", "A1": "pod", "G1": "vertical", "E1": "pod",
    "R2": "pod", "A2": "pod", "P2": "vertical", "E2": "vertical",
    "N1": "night", "N2": "night", "L2": "overlap", "L4": "overlap",
    "H1": "teaching", "B1": "vertical", "L1": "overlap", "W5": "overlap",
    "L6": "overlap", "B2": "vertical",
}
ROLE_TYPES = ("flow", "pod", "vertical", "overlap", "teaching", "night", "oncall")


@dataclass(frozen=True)
class StaffingFeatureFrames:
    """Reusable schedule-only feature families."""

    current: pd.DataFrame
    structure: pd.DataFrame
    identity: pd.DataFrame


def parse_hour(series: pd.Series) -> pd.Series:
    """Parse timestamps and normalize timezone-aware values to Montreal wall clock."""
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return parsed.dt.floor("h")


def prepare_shifts(all_shifts_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw shift exports into one row per valid physician shift."""
    required = {"shift_start", "shift_end", "shift_short_name", "first_name", "last_name"}
    missing = required - set(all_shifts_df.columns)
    if missing:
        raise ValueError(f"Shift data missing required columns: {sorted(missing)}")

    shifts = all_shifts_df.copy()
    shifts["shift_start"] = parse_hour(shifts["shift_start"])
    shifts["shift_end"] = parse_hour(shifts["shift_end"])
    shifts["shift_type"] = shifts["shift_short_name"].map(SHIFT_TYPES)
    shifts["physician_id"] = (
        shifts["first_name"].fillna("").astype(str).str.strip()
        + shifts["last_name"].fillna("").astype(str).str.strip()
    )
    shifts = shifts.dropna(subset=["shift_start", "shift_end", "shift_type"])
    shifts = shifts.loc[shifts["physician_id"].ne("") & shifts["shift_end"].gt(shifts["shift_start"])].copy()
    shifts["shift_hours"] = (
        (shifts["shift_end"] - shifts["shift_start"]) / pd.Timedelta(hours=1)
    ).astype(float)
    return shifts.sort_values(["shift_start", "physician_id", "shift_short_name"]).reset_index(drop=True)


def expand_shift_hours(shifts: pd.DataFrame) -> pd.DataFrame:
    """Expand normalized shifts to [start, end) hourly physician-role rows."""
    rows: list[dict[str, object]] = []
    for row in shifts.itertuples(index=False):
        for hour in pd.date_range(row.shift_start, row.shift_end, freq="h", inclusive="left"):
            since_start = float((hour - row.shift_start) / pd.Timedelta(hours=1))
            until_end = float((row.shift_end - hour) / pd.Timedelta(hours=1))
            rows.append(
                {
                    TS_COL: hour,
                    "physician_id": row.physician_id,
                    "shift_type": row.shift_type,
                    "shift_short_name": row.shift_short_name,
                    "shift_start": row.shift_start,
                    "shift_end": row.shift_end,
                    "hours_since_start": since_start,
                    "hours_until_end": until_end,
                }
            )
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("No valid physician shift-hours could be built.")
    return expanded.sort_values([TS_COL, "physician_id", "shift_type"]).reset_index(drop=True)


def _identity_matrix(expanded: pd.DataFrame) -> pd.DataFrame:
    identity = (
        expanded.pivot_table(
            index=TS_COL,
            columns="physician_id",
            values="shift_type",
            aggfunc="first",
        )
        .fillna("NotWorking")
        .add_prefix("physician__")
    )
    return identity


def _role_counts(expanded: pd.DataFrame) -> pd.DataFrame:
    counts = (
        expanded.groupby([TS_COL, "shift_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ROLE_TYPES, fill_value=0)
        .add_prefix("n_")
        .astype(float)
    )
    counts["n_total_scheduled"] = counts.sum(axis=1)
    counts["scheduled_oncall"] = (counts["n_oncall"] > 0).astype(float)
    return counts


def _transition_counts(shifts: pd.DataFrame, prefix: str, timestamp_col: str) -> pd.DataFrame:
    base = (
        shifts.groupby([timestamp_col, "shift_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ROLE_TYPES, fill_value=0)
        .astype(float)
    )
    base.columns = [f"n_{prefix}_{role}" for role in base.columns]
    base[f"n_{prefix}"] = base.sum(axis=1)
    base.index.name = TS_COL
    return base


def _team_continuity(expanded: pd.DataFrame) -> pd.DataFrame:
    active_sets = expanded.groupby(TS_COL)["physician_id"].agg(lambda x: frozenset(x))
    hours = pd.date_range(active_sets.index.min(), active_sets.index.max(), freq="h")
    active_sets = active_sets.reindex(hours, fill_value=frozenset())

    rows: list[dict[str, float | pd.Timestamp]] = []
    for idx, hour in enumerate(hours):
        current = active_sets.iloc[idx]
        previous = active_sets.iloc[idx - 1] if idx else frozenset()
        following = active_sets.iloc[idx + 1] if idx + 1 < len(hours) else frozenset()

        def retention(a: frozenset[str], b: frozenset[str]) -> float:
            union = a | b
            return float(len(a & b) / len(union)) if union else 1.0

        rows.append(
            {
                TS_COL: hour,
                "team_retention_prev_1h": retention(current, previous),
                "team_retention_next_1h": retention(current, following),
                "n_team_changes_prev_1h": float(len(current ^ previous)),
                "n_team_changes_next_1h": float(len(current ^ following)),
            }
        )
    return pd.DataFrame(rows).set_index(TS_COL)


def build_schedule_feature_frames(all_shifts_df: pd.DataFrame) -> StaffingFeatureFrames:
    """Build current, structural, and identity schedule feature families.

    ``current`` reproduces the pre-existing staffing representation: per-physician
    categorical role plus role counts and on-call identity.

    ``structure`` adds handoff, shift-phase, composition, continuity, and next-hour
    coverage features without using physician names.
    """
    shifts = prepare_shifts(all_shifts_df)
    expanded = expand_shift_hours(shifts)
    identity = _identity_matrix(expanded)
    counts = _role_counts(expanded)

    oncall_ids = (
        expanded.loc[expanded["shift_type"].eq("oncall")]
        .groupby(TS_COL)["physician_id"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("oncall_physician_id")
    )

    current = identity.join(counts, how="outer").join(oncall_ids, how="left")
    current["oncall_physician_id"] = current["oncall_physician_id"].fillna("None")

    starts = _transition_counts(shifts, "shift_starts", "shift_start")
    ends = _transition_counts(shifts, "shift_ends", "shift_end")

    phase = expanded.groupby(TS_COL).agg(
        team_mean_hours_since_start=("hours_since_start", "mean"),
        team_mean_hours_remaining=("hours_until_end", "mean"),
        team_min_hours_remaining=("hours_until_end", "min"),
        n_first_2h=("hours_since_start", lambda s: float((s < 2).sum())),
        n_last_2h=("hours_until_end", lambda s: float((s <= 2).sum())),
        n_last_1h=("hours_until_end", lambda s: float((s <= 1).sum())),
    )
    continuity = _team_continuity(expanded)

    structure = counts.join(starts, how="outer").join(ends, how="outer").join(phase, how="outer").join(continuity, how="outer")
    structure = structure.sort_index()
    full_hours = pd.date_range(structure.index.min(), structure.index.max(), freq="h", name=TS_COL)
    structure = structure.reindex(full_hours)
    numeric_cols = structure.columns.tolist()
    structure[numeric_cols] = structure[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    denom = structure["n_total_scheduled"].replace(0, np.nan)
    for role in ROLE_TYPES:
        structure[f"share_{role}"] = (structure[f"n_{role}"] / denom).fillna(0.0)
        structure[f"delta_n_{role}_next_1h"] = structure[f"n_{role}"].shift(-1) - structure[f"n_{role}"]
    structure["delta_n_total_next_1h"] = structure["n_total_scheduled"].shift(-1) - structure["n_total_scheduled"]
    lead_cols = [c for c in structure.columns if c.startswith("delta_n_")]
    structure[lead_cols] = structure[lead_cols].fillna(0.0)

    current = current.reset_index().sort_values(TS_COL).reset_index(drop=True)
    structure = structure.reset_index().sort_values(TS_COL).reset_index(drop=True)
    identity = identity.reset_index().sort_values(TS_COL).reset_index(drop=True)
    return StaffingFeatureFrames(current=current, structure=structure, identity=identity)


def _flow_training_frame(
    flow_df: pd.DataFrame,
    targets: Sequence[str],
    profile_end: pd.Timestamp | None,
) -> pd.DataFrame:
    flow = flow_df.copy()
    if TS_COL not in flow:
        raise ValueError(f"Flow data must contain {TS_COL!r}")
    flow[TS_COL] = parse_hour(flow[TS_COL])
    flow = flow.dropna(subset=[TS_COL]).sort_values(TS_COL).drop_duplicates(TS_COL, keep="last")
    if profile_end is not None:
        profile_end = pd.Timestamp(profile_end).floor("h")
        flow = flow.loc[flow[TS_COL].le(profile_end)].copy()
    missing = [target for target in targets if target not in flow.columns]
    if missing:
        raise ValueError(f"Flow data missing effect target(s): {missing}")
    for target in targets:
        flow[target] = pd.to_numeric(flow[target], errors="coerce")
        flow[f"__delta__{target}"] = flow[target].shift(-1) - flow[target]
    delta_cols = [f"__delta__{target}" for target in targets]
    return flow.dropna(subset=delta_cols, how="all")


def _ridge_residualize(y: pd.Series, controls: pd.DataFrame, alpha: float) -> pd.Series:
    valid = y.notna()
    if controls.empty:
        return y.copy()
    x = controls.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    valid &= np.isfinite(yv)
    if valid.sum() < max(12, x.shape[1] + 2):
        return y.copy()

    xv = x[valid.to_numpy()]
    yy = yv[valid.to_numpy()]
    means = xv.mean(axis=0)
    scales = xv.std(axis=0)
    scales[scales < 1e-8] = 1.0
    xs = (xv - means) / scales
    design = np.column_stack([np.ones(len(xs)), xs])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ yy)

    all_xs = (x - means) / scales
    pred = np.column_stack([np.ones(len(all_xs)), all_xs]) @ beta
    out = y.copy()
    out.loc[valid] = y.loc[valid] - pred[valid.to_numpy()]
    return out


def fit_physician_effect_profiles(
    flow_df: pd.DataFrame,
    all_shifts_df: pd.DataFrame,
    targets: Sequence[str],
    *,
    profile_end: pd.Timestamp | None = None,
    min_active_hours: int = 12,
    shrinkage_hours: float = 48.0,
    control_ridge_alpha: float = 8.0,
) -> pd.DataFrame:
    """Estimate leakage-safe physician-role flow fingerprints.

    For each target the outcome is the next-hour change (``target[t+1]-target[t]``).
    We remove an hour-of-week baseline and a ridge fit of structural role counts plus
    current target level (and ``Inflow_Total`` when available). The remaining mean while
    each physician-role pair is active is shrunk toward zero by ``shrinkage_hours``.

    The returned scores are intentionally *associative*. They can be useful predictive
    covariates, but should not be interpreted as causal physician performance rankings.
    """
    if min_active_hours < 1 or shrinkage_hours < 0:
        raise ValueError("min_active_hours must be >=1 and shrinkage_hours must be >=0")

    targets = list(targets)
    flow = _flow_training_frame(flow_df, targets, profile_end)
    shifts = prepare_shifts(all_shifts_df)
    expanded = expand_shift_hours(shifts)

    active = expanded[[TS_COL, "physician_id", "shift_type"]].drop_duplicates()
    role_counts = _role_counts(expanded).reset_index()
    training = flow.merge(role_counts, on=TS_COL, how="left")
    count_cols = [f"n_{role}" for role in ROLE_TYPES] + ["n_total_scheduled"]
    training[count_cols] = training[count_cols].fillna(0.0)

    if "Inflow_Total" in flow_df.columns and "Inflow_Total" not in training.columns:
        inflow = flow_df[[TS_COL, "Inflow_Total"]].copy()
        inflow[TS_COL] = parse_hour(inflow[TS_COL])
        inflow["Inflow_Total"] = pd.to_numeric(inflow["Inflow_Total"], errors="coerce")
        training = training.merge(inflow.drop_duplicates(TS_COL, keep="last"), on=TS_COL, how="left")

    training["hour_of_week"] = training[TS_COL].dt.dayofweek * 24 + training[TS_COL].dt.hour
    profiles: pd.DataFrame | None = None

    for target in targets:
        delta_col = f"__delta__{target}"
        y = training[delta_col].copy()
        how_baseline = training.groupby("hour_of_week")[delta_col].transform("mean")
        y = y - how_baseline.fillna(y.mean())

        control_cols = count_cols + [target]
        if "Inflow_Total" in training.columns:
            control_cols.append("Inflow_Total")
        residual = _ridge_residualize(y, training[control_cols], control_ridge_alpha)
        residual_frame = training[[TS_COL]].copy()
        residual_frame["residual"] = residual

        joined = active.merge(residual_frame, on=TS_COL, how="inner").dropna(subset=["residual"])
        stats = joined.groupby(["physician_id", "shift_type"], as_index=False).agg(
            n_hours=("residual", "size"),
            raw_effect=("residual", "mean"),
            effect_sd=("residual", "std"),
        )
        stats["profiled"] = (stats["n_hours"] >= int(min_active_hours)).astype(int)
        weight = stats["n_hours"] / (stats["n_hours"] + float(shrinkage_hours))
        stats[f"effect__{target}"] = np.where(
            stats["profiled"].eq(1), stats["raw_effect"] * weight, 0.0
        )
        stats = stats.drop(columns=["raw_effect", "effect_sd"])
        stats = stats.rename(columns={"n_hours": f"n_hours__{target}", "profiled": f"profiled__{target}"})
        if profiles is None:
            profiles = stats
        else:
            profiles = profiles.merge(stats, on=["physician_id", "shift_type"], how="outer")

    if profiles is None:
        return pd.DataFrame(columns=["physician_id", "shift_type"])
    value_cols = [c for c in profiles.columns if c not in {"physician_id", "shift_type"}]
    profiles[value_cols] = profiles[value_cols].fillna(0.0)
    return profiles.sort_values(["physician_id", "shift_type"]).reset_index(drop=True)


def build_effect_score_features(
    all_shifts_df: pd.DataFrame,
    profiles: pd.DataFrame,
    targets: Sequence[str],
) -> pd.DataFrame:
    """Convert frozen physician profiles + schedule into hourly known covariates."""
    shifts = prepare_shifts(all_shifts_df)
    active = expand_shift_hours(shifts)[[TS_COL, "physician_id", "shift_type"]].drop_duplicates()
    merged = active.merge(profiles, on=["physician_id", "shift_type"], how="left")

    rows: list[pd.DataFrame] = []
    for target in targets:
        effect_col = f"effect__{target}"
        profiled_col = f"profiled__{target}"
        hours_col = f"n_hours__{target}"
        if effect_col not in merged:
            merged[effect_col] = 0.0
        if profiled_col not in merged:
            merged[profiled_col] = 0.0
        if hours_col not in merged:
            merged[hours_col] = 0.0
        merged[[effect_col, profiled_col, hours_col]] = merged[[effect_col, profiled_col, hours_col]].fillna(0.0)

        grouped = merged.groupby(TS_COL).agg(
            **{
                f"staff_effect__{target}_sum": (effect_col, "sum"),
                f"staff_effect__{target}_mean": (effect_col, "mean"),
                f"staff_effect__{target}_profiled_n": (profiled_col, "sum"),
                f"staff_effect__{target}_profile_hours_mean": (hours_col, "mean"),
            }
        )
        rows.append(grouped)

    if not rows:
        return pd.DataFrame({TS_COL: sorted(active[TS_COL].unique())})
    result = rows[0]
    for frame in rows[1:]:
        result = result.join(frame, how="outer")
    result = result.fillna(0.0).reset_index().sort_values(TS_COL)
    return result.reset_index(drop=True)


def sanitize_identity_for_cutoff(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    prefix: str = "physician__",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace unseen future physician-role categories with ``NotWorking``.

    Chronos categorical encoders should never be asked to transform a role level that was
    absent from the historical context for that physician column (for example, a newly
    hired physician whose first shift is in the forecast horizon).
    """
    history = history.copy()
    future = future.copy()
    for column in [c for c in future.columns if c.startswith(prefix) and c in history.columns]:
        history[column] = history[column].fillna("NotWorking").astype(str)
        future[column] = future[column].fillna("NotWorking").astype(str)
        seen = set(history[column].dropna().astype(str))
        future.loc[~future[column].isin(seen), column] = "NotWorking"
    return history, future


def select_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Small helper used by backtests to keep deterministic feature order."""
    wanted = [TS_COL, *[c for c in columns if c != TS_COL]]
    return frame[[c for c in wanted if c in frame.columns]].copy()
