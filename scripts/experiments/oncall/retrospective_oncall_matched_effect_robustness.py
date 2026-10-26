from __future__ import annotations

"""Robustness analysis for retrospective on-call activation effects.

The broad nearest-neighbour analysis is useful for exploration but activation occurs in
systematically busier states, so regression to the mean/confounding by indication can
inflate apparent benefit. This runner therefore estimates the same 2/4/6-hour flow
changes using common-support calipers plus pre-activation trajectory features.

These remain observational associations, not causal treatment effects.
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
from retrospective_oncall_matched_effect import (  # noqa: E402
    FLOW_METRICS,
    MATCH_FEATURE_CANDIDATES,
    add_analysis_columns,
    bootstrap_ci,
    candidate_frames,
    circular_hour_distance,
    standardized_mean_difference,
)

HORIZONS = (2, 4, 6)
TREND_METRICS = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "stretcher_occupancy",
    "overflow",
    "WAITINGADM",
    "Inflow_Total",
)
TREND_FEATURES = (
    "total_tbs_change_1h",
    "total_tbs_change_2h",
    "pod_tbs_change_1h",
    "vertical_tbs_change_1h",
    "overflow_change_1h",
    "WAITINGADM_change_1h",
    "Inflow_Total_change_1h",
)

SPECS = {
    # Primary: prioritizes balance/common support over retaining every activation.
    "primary_common_support_pretrend_1nn": {
        "matches": 1,
        "max_hour_distance": 1,
        "max_date_gap_days": 365,
        "calipers": {
            "total_tbs": 5,
            "vertical_tbs": 4,
            "pod_tbs": 4,
            "n_overlap": 0,
            "n_vertical": 1,
            "n_pod": 1,
        },
    },
    # Tighter congestion-state calipers as a sensitivity check.
    "strict_common_support_pretrend_1nn": {
        "matches": 1,
        "max_hour_distance": 1,
        "max_date_gap_days": 365,
        "calipers": {
            "total_tbs": 4,
            "vertical_tbs": 3,
            "pod_tbs": 3,
            "n_overlap": 0,
            "n_vertical": 1,
            "n_pod": 1,
        },
    },
}


def add_pretrends(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(TS_COL).reset_index(drop=True)
    for metric in TREND_METRICS:
        if metric not in out.columns:
            continue
        values = pd.to_numeric(out[metric], errors="coerce")
        out[f"{metric}_change_1h"] = values - values.shift(1)
        out[f"{metric}_change_2h"] = values - values.shift(2)
    return out


def numeric_features(treated: pd.DataFrame, controls: pd.DataFrame) -> list[str]:
    candidates = [*MATCH_FEATURE_CANDIDATES, *TREND_FEATURES]
    features: list[str] = []
    for col in candidates:
        if col not in treated.columns or col not in controls.columns:
            continue
        values = pd.to_numeric(pd.concat([treated[col], controls[col]]), errors="coerce")
        if values.notna().sum() > 10:
            features.append(col)
    if not features:
        raise ValueError("No usable matching features are available.")
    return features


def match_spec(
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
    spec: dict[str, object],
) -> pd.DataFrame:
    combined = pd.concat([treated[features], controls[features]], ignore_index=True)
    numeric = combined.apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scales = numeric.std(ddof=0).replace(0, 1.0).fillna(1.0)
    control_matrix = controls[features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    control_z = (control_matrix - medians) / scales

    max_hour_distance = int(spec["max_hour_distance"])
    max_date_gap_days = int(spec["max_date_gap_days"])
    matches = int(spec["matches"])
    calipers = dict(spec["calipers"])
    rows: list[dict[str, object]] = []

    for treated_idx, t in treated.iterrows():
        t_time = pd.Timestamp(t[TS_COL])
        hour_distance = circular_hour_distance(int(t["hour"]), controls["hour"])
        date_gap = (pd.to_datetime(controls[TS_COL]) - t_time).abs().dt.total_seconds() / 86400.0
        eligible = (
            controls["is_weekend"].eq(int(t["is_weekend"]))
            & hour_distance.le(max_hour_distance)
            & date_gap.le(max_date_gap_days)
        )
        for column, width in calipers.items():
            if column not in controls.columns or column not in treated.columns:
                continue
            eligible &= (
                pd.to_numeric(controls[column], errors="coerce")
                - float(pd.to_numeric(pd.Series([t[column]]), errors="coerce").iloc[0])
            ).abs().le(float(width))

        pool_idx = controls.index[eligible]
        if pool_idx.empty:
            continue

        treated_values = pd.to_numeric(t[features], errors="coerce").fillna(medians)
        treated_z = (treated_values - medians) / scales
        state_distance = np.sqrt(((control_z.loc[pool_idx] - treated_z) ** 2).mean(axis=1))
        time_penalty = hour_distance.loc[pool_idx].astype(float) / max(max_hour_distance, 1)
        date_penalty = date_gap.loc[pool_idx].astype(float) / max(max_date_gap_days, 1)
        distance = state_distance + 0.10 * time_penalty + 0.05 * date_penalty
        chosen = distance.nsmallest(min(matches, len(distance)))

        for rank, (control_idx, dist) in enumerate(chosen.items(), start=1):
            row: dict[str, object] = {
                "treated_idx": int(treated_idx),
                "treated_id": t_time.isoformat(),
                "treated_ds": t_time,
                "control_idx": int(control_idx),
                "control_ds": controls.loc[control_idx, TS_COL],
                "match_rank": rank,
                "match_distance": float(dist),
            }
            for feature in features:
                row[f"treated_{feature}"] = t.get(feature, np.nan)
                row[f"control_{feature}"] = controls.loc[control_idx].get(feature, np.nan)
            for horizon in HORIZONS:
                for metric in FLOW_METRICS:
                    col = f"{metric}_delta_{horizon}h"
                    if col in treated.columns and col in controls.columns:
                        row[f"treated_{col}"] = t.get(col, np.nan)
                        row[f"control_{col}"] = controls.loc[control_idx].get(col, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_balance(
    spec_name: str,
    treated: pd.DataFrame,
    controls: pd.DataFrame,
    pairs: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    matched_ids = set(pairs["treated_idx"].astype(int))
    matched_treated = treated.loc[treated.index.isin(matched_ids)]
    unique_treated = pairs[["treated_idx", *[f"treated_{f}" for f in features]]].drop_duplicates(
        "treated_idx"
    )
    rows = []
    for feature in features:
        before = standardized_mean_difference(treated[feature], controls[feature])
        after = standardized_mean_difference(
            unique_treated[f"treated_{feature}"], pairs[f"control_{feature}"]
        )
        rows.append(
            {
                "spec": spec_name,
                "feature": feature,
                "smd_before": before,
                "smd_after": after,
                "abs_smd_after": abs(after) if pd.notna(after) else np.nan,
                "matched_activation_episodes": int(matched_treated.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_effects(
    spec_name: str,
    pairs: pd.DataFrame,
    bootstrap_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_rows: list[dict[str, object]] = []
    for treated_id, group in pairs.groupby("treated_id", sort=True):
        for horizon in HORIZONS:
            for metric in FLOW_METRICS:
                tcol = f"treated_{metric}_delta_{horizon}h"
                ccol = f"control_{metric}_delta_{horizon}h"
                if tcol not in group.columns or ccol not in group.columns:
                    continue
                treated_change = pd.to_numeric(group[tcol], errors="coerce").iloc[0]
                control_change = pd.to_numeric(group[ccol], errors="coerce").mean()
                if pd.isna(treated_change) or pd.isna(control_change):
                    continue
                episode_rows.append(
                    {
                        "spec": spec_name,
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
    for (horizon, metric), group in episodes.groupby(["horizon_hours", "metric"]):
        lo, hi = bootstrap_ci(group["observational_effect"], bootstrap_reps)
        effect = float(group["observational_effect"].mean())
        summary_rows.append(
            {
                "spec": spec_name,
                "horizon_hours": int(horizon),
                "metric": metric,
                "matched_activation_episodes": int(group["treated_id"].nunique()),
                "matched_observational_effect": effect,
                "bootstrap_ci_low": lo,
                "bootstrap_ci_high": hi,
                "ci_excludes_zero": bool((hi < 0) or (lo > 0)),
                "lower_is_better": True,
                "causal_interpretation_allowed": False,
            }
        )
    return pd.DataFrame(summary_rows), episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrend-aware matched on-call robustness analysis.")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    raw = truncate_to_explicit_label_coverage(load_dataset())
    raw = add_callability_features(raw, late_call_hour=21, min_rest_buffer_hours=10.0)
    analysis = add_pretrends(add_analysis_columns(raw, HORIZONS))
    treated, controls = candidate_frames(analysis, max(HORIZONS))
    features = numeric_features(treated, controls)

    all_summaries = []
    all_episodes = []
    all_balance = []
    all_pairs = []

    for spec_name, spec in SPECS.items():
        pairs = match_spec(treated, controls, features, spec)
        if pairs.empty:
            raise ValueError(f"No matches for robustness specification {spec_name}.")
        summary, episodes = summarize_effects(spec_name, pairs, args.bootstrap_reps)
        balance = summarize_balance(spec_name, treated, controls, pairs, features)
        pairs.insert(0, "spec", spec_name)
        max_abs_smd = float(balance["abs_smd_after"].max())
        summary["max_abs_smd_after"] = max_abs_smd
        summary["acceptable_balance_lt_0_10"] = max_abs_smd < 0.10
        summary["near_acceptable_balance_lt_0_15"] = max_abs_smd < 0.15
        all_summaries.append(summary)
        all_episodes.append(episodes)
        all_balance.append(balance)
        all_pairs.append(pairs)

    summary = pd.concat(all_summaries, ignore_index=True)
    episodes = pd.concat(all_episodes, ignore_index=True)
    balance = pd.concat(all_balance, ignore_index=True)
    pairs = pd.concat(all_pairs, ignore_index=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "oncall_matched_effect_robustness_summary.csv", index=False)
    episodes.to_csv(out / "oncall_matched_effect_robustness_episodes.csv", index=False)
    balance.to_csv(out / "oncall_matched_effect_robustness_balance.csv", index=False)
    pairs.to_csv(out / "oncall_matched_effect_robustness_pairs.csv", index=False)

    start, end = trusted_hourly_label_bounds()
    print(f"Trusted exact-hour labels: {start} through {end}")
    print("\nPrimary/strict Total-TBS estimates:")
    print(summary[summary["metric"].eq("total_tbs")].to_string(index=False))
    print("\nWorst balance by specification:")
    print(
        balance.groupby("spec", as_index=False)["abs_smd_after"]
        .max()
        .sort_values("abs_smd_after")
        .to_string(index=False)
    )
    print(
        "\nInterpretation guard: these estimates remain observational. The robustness "
        "specifications are designed to expose confounding/regression-to-the-mean, not "
        "to certify a causal treatment effect."
    )


if __name__ == "__main__":
    main()
