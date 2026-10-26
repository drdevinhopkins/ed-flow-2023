from __future__ import annotations

"""Matched observational estimate of flow change after on-call activation.

This experiment uses only the trusted exact-hour activation-label interval. It is
intended to answer a narrower question than the activation-probability model:

    After the on-call physician actually starts working, how does subsequent ED
    congestion change versus comparable callable hours when on-call is not activated?

The estimate is observational, not causal. Matching reduces measured state/time
imbalance but cannot remove confounding by indication, clinician behaviour, or
unmeasured operational context.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecast_oncall_probability import TS_COL, load_dataset  # noqa: E402
from retrospective_oncall_availability_adjusted import add_callability_features  # noqa: E402
from retrospective_oncall_label_coverage import (  # noqa: E402
    trusted_hourly_label_bounds,
    truncate_to_explicit_label_coverage,
)

FLOW_METRICS = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "stretcher_occupancy",
    "overflow",
    "WAITINGADM",
)
MATCH_FEATURE_CANDIDATES = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "stretcher_occupancy",
    "overflow",
    "WAITINGADM",
    "Inflow_Total",
    "n_pod",
    "n_vertical",
    "n_overlap",
    "n_flow",
    "n_total_working_excl_oncall",
)
DEFAULT_HORIZONS = (2, 4, 6)
DEFAULT_MATCHES = 5
DEFAULT_MAX_HOUR_DISTANCE = 2
DEFAULT_MAX_DATE_GAP_DAYS = 120
DEFAULT_LATE_CALL_HOUR = 21
DEFAULT_MIN_REST_BUFFER_HOURS = 10.0
DEFAULT_BOOTSTRAP_REPS = 2000
RANDOM_SEED = 42


def circular_hour_distance(a: int, b: pd.Series) -> pd.Series:
    diff = (pd.to_numeric(b, errors="coerce") - int(a)).abs()
    return np.minimum(diff, 24 - diff)


def add_analysis_columns(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    out = df.copy().sort_values(TS_COL).reset_index(drop=True)
    out[TS_COL] = pd.to_datetime(out[TS_COL], errors="coerce")
    out["oncall_active"] = pd.to_numeric(out["oncall_active"], errors="coerce").fillna(0).clip(0, 1)
    total_scheduled = pd.to_numeric(out["n_total_scheduled"], errors="coerce")
    n_oncall = pd.to_numeric(out["n_oncall"], errors="coerce").fillna(0)
    out["n_total_working_excl_oncall"] = total_scheduled - n_oncall
    out["hour"] = out[TS_COL].dt.hour
    out["is_weekend"] = (out[TS_COL].dt.dayofweek >= 5).astype(int)

    prev = out["oncall_active"].shift(1).fillna(0)
    out["activation_start"] = out["oncall_active"].eq(1) & prev.eq(0)

    for horizon in horizons:
        future_active = [
            out["oncall_active"].shift(-step) for step in range(1, horizon + 1)
        ]
        out[f"activation_next_{horizon}h"] = pd.concat(future_active, axis=1).max(axis=1)
        for metric in FLOW_METRICS:
            if metric not in out.columns:
                continue
            values = pd.to_numeric(out[metric], errors="coerce")
            future = values.shift(-horizon)
            out[f"{metric}_delta_{horizon}h"] = future - values
            forward = pd.concat(
                [values.shift(-step) for step in range(1, horizon + 1)], axis=1
            )
            out[f"{metric}_mean_next_{horizon}h"] = forward.mean(axis=1)
    return out


def candidate_frames(
    df: pd.DataFrame,
    max_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scheduled = df["oncall_scheduled"].fillna(False).astype(bool)
    treated = df[df["activation_start"] & scheduled].copy()

    prior_active = pd.concat(
        [df["oncall_active"].shift(step) for step in (1, 2)], axis=1
    ).max(axis=1)
    no_recent_activation = prior_active.fillna(0).eq(0)
    no_future_activation = df[f"activation_next_{max_horizon}h"].fillna(1).eq(0)
    callable_now = df["callability_status"].eq("callable_proxy")
    controls = df[
        df["oncall_active"].eq(0)
        & scheduled
        & callable_now
        & no_recent_activation
        & no_future_activation
    ].copy()
    return treated.reset_index(drop=True), controls.reset_index(drop=True)


def standardize_features(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
) -> tuple[list[str], pd.Series, pd.Series]:
    features = [
        c
        for c in MATCH_FEATURE_CANDIDATES
        if c in treated.columns
        and c in controls.columns
        and pd.to_numeric(pd.concat([treated[c], controls[c]]), errors="coerce").notna().sum() > 10
    ]
    if not features:
        raise ValueError("No usable numeric match features are available.")

    combined = pd.concat([treated[features], controls[features]], ignore_index=True)
    numeric = combined.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scales = numeric.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return features, medians, scales


def match_controls(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
    medians: pd.Series,
    scales: pd.Series,
    matches: int,
    max_hour_distance: int,
    max_date_gap_days: int,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    control_matrix = controls[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    control_z = (control_matrix - medians) / scales

    rows: list[dict[str, object]] = []
    for treated_row in treated.itertuples(index=False):
        t = pd.Series(treated_row._asdict())
        t_time = pd.Timestamp(t[TS_COL])
        hour_dist = circular_hour_distance(int(t["hour"]), controls["hour"])
        date_gap = (pd.to_datetime(controls[TS_COL]) - t_time).abs().dt.total_seconds() / 86400.0
        eligible = (
            controls["is_weekend"].eq(int(t["is_weekend"]))
            & hour_dist.le(max_hour_distance)
            & date_gap.le(max_date_gap_days)
        )

        pool = controls.loc[eligible].copy()
        if pool.empty:
            continue
        pool_z = control_z.loc[pool.index]
        t_values = pd.to_numeric(t[features], errors="coerce").fillna(medians)
        t_z = (t_values - medians) / scales
        state_distance = np.sqrt(((pool_z - t_z) ** 2).mean(axis=1))
        time_penalty = hour_dist.loc[pool.index].astype(float) / max(max_hour_distance, 1)
        date_penalty = date_gap.loc[pool.index].astype(float) / max(max_date_gap_days, 1)
        distance = state_distance + 0.20 * time_penalty + 0.10 * date_penalty
        chosen = distance.nsmallest(min(matches, len(distance)))

        treated_id = t_time.isoformat()
        for rank, (control_idx, dist) in enumerate(chosen.items(), start=1):
            control = controls.loc[control_idx]
            row: dict[str, object] = {
                "treated_id": treated_id,
                "treated_ds": t_time,
                "control_ds": control[TS_COL],
                "match_rank": rank,
                "match_distance": float(dist),
                "hour_distance": float(hour_dist.loc[control_idx]),
                "date_gap_days": float(date_gap.loc[control_idx]),
            }
            for feature in features:
                row[f"treated_{feature}"] = t.get(feature, np.nan)
                row[f"control_{feature}"] = control.get(feature, np.nan)
            for horizon in horizons:
                for metric in FLOW_METRICS:
                    delta = f"{metric}_delta_{horizon}h"
                    if delta in treated.columns and delta in controls.columns:
                        row[f"treated_{delta}"] = t.get(delta, np.nan)
                        row[f"control_{delta}"] = control.get(delta, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def standardized_mean_difference(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if x.empty or y.empty:
        return np.nan
    pooled = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled == 0:
        return 0.0 if np.isclose(x.mean(), y.mean()) else np.nan
    return float((x.mean() - y.mean()) / pooled)


def balance_table(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    pairs: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    rows = []
    matched_treated = pairs[["treated_id"] + [f"treated_{f}" for f in features]].drop_duplicates("treated_id")
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "smd_before": standardized_mean_difference(treated[feature], controls[feature]),
                "smd_after": standardized_mean_difference(
                    matched_treated[f"treated_{feature}"],
                    pairs[f"control_{feature}"],
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["abs_smd_before"] = out["smd_before"].abs()
    out["abs_smd_after"] = out["smd_after"].abs()
    return out.sort_values("abs_smd_after", ascending=False).reset_index(drop=True)


def bootstrap_ci(values: pd.Series, reps: int, seed: int = RANDOM_SEED) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 2 or reps <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def effect_summary(
    pairs: pd.DataFrame,
    horizons: tuple[int, ...],
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, object]] = []
    for treated_id, group in pairs.groupby("treated_id", sort=True):
        for horizon in horizons:
            for metric in FLOW_METRICS:
                tcol = f"treated_{metric}_delta_{horizon}h"
                ccol = f"control_{metric}_delta_{horizon}h"
                if tcol not in group or ccol not in group:
                    continue
                treated_change = pd.to_numeric(group[tcol], errors="coerce").iloc[0]
                control_change = pd.to_numeric(group[ccol], errors="coerce").mean()
                if pd.isna(treated_change) or pd.isna(control_change):
                    continue
                episode_rows.append(
                    {
                        "treated_id": treated_id,
                        "horizon_hours": horizon,
                        "metric": metric,
                        "treated_change": float(treated_change),
                        "matched_control_change": float(control_change),
                        "observational_effect": float(treated_change - control_change),
                    }
                )
    episodes = pd.DataFrame(episode_rows)

    summary_rows: list[dict[str, object]] = []
    if not episodes.empty:
        for (horizon, metric), group in episodes.groupby(["horizon_hours", "metric"]):
            lo, hi = bootstrap_ci(group["observational_effect"], bootstrap_reps)
            att = float(group["observational_effect"].mean())
            summary_rows.append(
                {
                    "horizon_hours": int(horizon),
                    "metric": metric,
                    "matched_activation_episodes": int(group["treated_id"].nunique()),
                    "treated_mean_change": float(group["treated_change"].mean()),
                    "matched_control_mean_change": float(group["matched_control_change"].mean()),
                    "matched_observational_effect": att,
                    "bootstrap_ci_low": lo,
                    "bootstrap_ci_high": hi,
                    "lower_is_better": True,
                    "direction": (
                        "associated_with_improvement"
                        if att < 0
                        else "associated_with_worsening"
                        if att > 0
                        else "no_difference"
                    ),
                    "causal_interpretation_allowed": False,
                }
            )
    return pd.DataFrame(summary_rows), episodes


def run_analysis(
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    matches: int = DEFAULT_MATCHES,
    max_hour_distance: int = DEFAULT_MAX_HOUR_DISTANCE,
    max_date_gap_days: int = DEFAULT_MAX_DATE_GAP_DAYS,
    late_call_hour: int = DEFAULT_LATE_CALL_HOUR,
    min_rest_buffer_hours: float = DEFAULT_MIN_REST_BUFFER_HOURS,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = truncate_to_explicit_label_coverage(load_dataset())
    raw = add_callability_features(raw, late_call_hour, min_rest_buffer_hours)
    analysis = add_analysis_columns(raw, horizons)
    max_horizon = max(horizons)
    treated, controls = candidate_frames(analysis, max_horizon)

    features, medians, scales = standardize_features(treated, controls)
    pairs = match_controls(
        treated,
        controls,
        features,
        medians,
        scales,
        matches,
        max_hour_distance,
        max_date_gap_days,
        horizons,
    )
    if pairs.empty:
        raise ValueError("No treated activation episodes could be matched to eligible controls.")

    matched_ids = set(pairs["treated_id"])
    treated_matched = treated[
        treated[TS_COL].map(lambda x: pd.Timestamp(x).isoformat()).isin(matched_ids)
    ]
    balance = balance_table(treated_matched, controls, pairs, features)
    summary, episodes = effect_summary(pairs, horizons, bootstrap_reps)
    return summary, episodes, balance, pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched observational estimate of ED flow after on-call activation."
    )
    parser.add_argument("--horizons", default="2,4,6")
    parser.add_argument("--matches", type=int, default=DEFAULT_MATCHES)
    parser.add_argument("--max-hour-distance", type=int, default=DEFAULT_MAX_HOUR_DISTANCE)
    parser.add_argument("--max-date-gap-days", type=int, default=DEFAULT_MAX_DATE_GAP_DAYS)
    parser.add_argument("--late-call-hour", type=int, default=DEFAULT_LATE_CALL_HOUR)
    parser.add_argument("--min-rest-buffer-hours", type=float, default=DEFAULT_MIN_REST_BUFFER_HOURS)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    horizons = tuple(sorted({int(x) for x in args.horizons.split(",") if x.strip()}))
    if not horizons:
        raise ValueError("At least one horizon is required.")
    if any(h not in DEFAULT_HORIZONS for h in horizons):
        raise ValueError(f"Supported horizons are {DEFAULT_HORIZONS}; got {horizons}")

    summary, episodes, balance, pairs = run_analysis(
        horizons=horizons,
        matches=args.matches,
        max_hour_distance=args.max_hour_distance,
        max_date_gap_days=args.max_date_gap_days,
        late_call_hour=args.late_call_hour,
        min_rest_buffer_hours=args.min_rest_buffer_hours,
        bootstrap_reps=args.bootstrap_reps,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "oncall_matched_effect_summary.csv", index=False)
    episodes.to_csv(out / "oncall_matched_effect_episodes.csv", index=False)
    balance.to_csv(out / "oncall_matched_effect_balance.csv", index=False)
    pairs.to_csv(out / "oncall_matched_effect_pairs.csv", index=False)

    start, end = trusted_hourly_label_bounds()
    print(f"Trusted exact-hour labels: {start} through {end}")
    print(f"Matched activation episodes: {episodes['treated_id'].nunique() if not episodes.empty else 0}")
    print("\nMatched observational effects (negative = less congestion versus controls):")
    print(summary.to_string(index=False))
    print("\nWorst post-match balance diagnostics:")
    print(balance.head(12).to_string(index=False))
    print(
        "\nCAVEAT: matched observational estimates are not causal; residual confounding by "
        "indication, clinician behaviour, and unmeasured operational context may remain."
    )


if __name__ == "__main__":
    main()
