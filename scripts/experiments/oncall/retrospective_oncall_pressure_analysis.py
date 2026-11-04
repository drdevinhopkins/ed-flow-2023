from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecast_oncall_probability import (  # noqa: E402
    HORIZONS,
    TS_COL,
    add_horizon_targets,
    add_time_and_trend_features,
    feature_columns,
    load_dataset,
)
from retrospective_oncall_decision_replay import (  # noqa: E402
    FLOW_METRICS,
    add_future_outcomes,
    bad_outcome_thresholds,
    chronological_three_way_split,
    safe_metric,
    train_independent_replay_model,
)

DEFAULT_HORIZON = 6
DEFAULT_PRESSURE_THRESHOLD = 95.0
DEFAULT_BAD_OUTCOME_QUANTILE = 0.90
DEFAULT_OUTCOME_HOURS = 6
DEFAULT_MAJOR_CORE_COUNT = 3
DEFAULT_MATCHES = 5
DEFAULT_TOP_N = 20
PRESSURE_CUTS = (90.0, 95.0, 99.0)
CORE_CONGESTION_METRICS = ("total_tbs", "pod_tbs", "overflow", "WAITINGADM")
MATCH_FEATURES = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "overflow",
    "WAITINGADM",
    "stretcher_occupancy",
    "n_total_scheduled",
    "n_pod",
    "n_vertical",
)


@dataclass(frozen=True)
class PressureConfig:
    horizon: int = DEFAULT_HORIZON
    pressure_threshold: float = DEFAULT_PRESSURE_THRESHOLD
    bad_outcome_quantile: float = DEFAULT_BAD_OUTCOME_QUANTILE
    outcome_hours: int = DEFAULT_OUTCOME_HOURS
    major_core_count: int = DEFAULT_MAJOR_CORE_COUNT
    matches: int = DEFAULT_MATCHES
    top_n: int = DEFAULT_TOP_N


def empirical_percentile(scores: np.ndarray, reference_scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_scores, dtype=float))
    if reference.size == 0:
        raise ValueError("No calibration scores available for pressure percentiles.")
    score_array = np.asarray(scores, dtype=float)
    return np.searchsorted(reference, score_array, side="right") / reference.size * 100.0


def assign_pressure_band(percentile: pd.Series) -> pd.Series:
    return pd.cut(
        percentile,
        bins=[-np.inf, 90.0, 95.0, 99.0, np.inf],
        labels=["routine", "elevated", "high", "very_high"],
        right=False,
    ).astype(str)


def add_congestion_composite(
    replay: pd.DataFrame,
    outcome_hours: int,
    major_core_count: int,
) -> pd.DataFrame:
    out = replay.copy()
    core_bad_cols = [
        f"future_{outcome_hours}h_bad_{metric}"
        for metric in CORE_CONGESTION_METRICS
        if f"future_{outcome_hours}h_bad_{metric}" in out.columns
    ]
    if len(core_bad_cols) < major_core_count:
        raise ValueError(
            f"Only {len(core_bad_cols)} core congestion metrics available; "
            f"need at least {major_core_count}."
        )
    out[f"core_congestion_count_within_{outcome_hours}h"] = (
        out[core_bad_cols].fillna(False).astype(int).sum(axis=1)
    )
    count_col = f"core_congestion_count_within_{outcome_hours}h"
    out[f"major_congestion_within_{outcome_hours}h"] = out[count_col] >= major_core_count
    out[f"extreme_congestion_within_{outcome_hours}h"] = out[count_col] >= len(core_bad_cols)
    return out


def cluster_pressure_episodes(replay: pd.DataFrame, config: PressureConfig) -> pd.DataFrame:
    alerts = replay[
        replay["complete_outcome_window"]
        & (replay["oncall_pressure_percentile"] >= config.pressure_threshold)
    ].copy()
    if alerts.empty:
        return pd.DataFrame()

    alerts = alerts.sort_values(TS_COL).reset_index(drop=True)
    gap_hours = alerts[TS_COL].diff().dt.total_seconds().div(3600)
    alerts["episode_id"] = (gap_hours.isna() | (gap_hours > 1.0)).cumsum()

    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for episode_id, group in alerts.groupby("episode_id", sort=True):
        group = group.sort_values(TS_COL)
        first = group.iloc[0]
        peak = group.sort_values(
            ["oncall_pressure_percentile", "raw_probability"], ascending=False
        ).iloc[0]
        activation = bool(group[f"actual_activation_within_{config.outcome_hours}h"].max())
        major = bool(group[major_col].max())
        extreme = bool(group[extreme_col].max())
        if activation:
            episode_class = "early_or_concordant_activation"
        elif major:
            episode_class = "candidate_missed_opportunity"
        else:
            episode_class = "high_pressure_no_major_congestion"

        row: dict[str, object] = {
            "episode_id": int(episode_id),
            "alert_start": first[TS_COL],
            "alert_end": group.iloc[-1][TS_COL],
            "alert_duration_hours": int(len(group)),
            "peak_pressure_percentile": float(group["oncall_pressure_percentile"].max()),
            "peak_pressure_band": peak["oncall_pressure_band"],
            "peak_raw_probability": float(peak["raw_probability"]),
            "peak_calibrated_probability": float(peak["calibrated_probability"]),
            "peak_pressure_time": peak[TS_COL],
            "actual_oncall_within_window": activation,
            "major_congestion_within_window": major,
            "extreme_congestion_within_window": extreme,
            "episode_class": episode_class,
            "hours_to_actual_activation": pd.to_numeric(
                group["hours_to_actual_activation"], errors="coerce"
            ).min(),
        }
        for metric in FLOW_METRICS:
            if metric in first.index:
                row[f"start_{metric}"] = first[metric]
            future_col = f"future_{config.outcome_hours}h_max_{metric}"
            if future_col in group.columns:
                row[f"subsequent_max_{metric}"] = pd.to_numeric(
                    group[future_col], errors="coerce"
                ).max()
        if "oncall_physician_id" in first.index:
            row["oncall_physician_id"] = first["oncall_physician_id"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["peak_pressure_percentile", "peak_raw_probability", "alert_start"],
        ascending=[False, False, True],
    )


def pressure_band_summary(replay: pd.DataFrame, config: PressureConfig) -> pd.DataFrame:
    valid = replay[replay["complete_outcome_window"]].copy()
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    order = ["routine", "elevated", "high", "very_high"]
    rows: list[dict[str, object]] = []
    for band in order:
        group = valid[valid["oncall_pressure_band"] == band]
        if group.empty:
            continue
        rows.append(
            {
                "pressure_band": band,
                "decision_hours": int(len(group)),
                "share_of_decision_hours": float(len(group) / len(valid)),
                "actual_activation_rate": float(group[activation_col].mean()),
                "major_congestion_rate": float(group[major_col].mean()),
                "extreme_congestion_rate": float(group[extreme_col].mean()),
                "median_raw_score": float(group["raw_probability"].median()),
                "median_calibrated_probability": float(group["calibrated_probability"].median()),
            }
        )
    return pd.DataFrame(rows)


def threshold_summary(replay: pd.DataFrame, config: PressureConfig) -> pd.DataFrame:
    valid = replay[replay["complete_outcome_window"]].copy()
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for threshold in PRESSURE_CUTS:
        group = valid[valid["oncall_pressure_percentile"] >= threshold]
        rows.append(
            {
                "pressure_percentile_threshold": threshold,
                "decision_hours": int(len(group)),
                "decision_hour_share": float(len(group) / len(valid)) if len(valid) else np.nan,
                "actual_activation_rate": float(group[activation_col].mean()) if len(group) else np.nan,
                "major_congestion_rate": float(group[major_col].mean()) if len(group) else np.nan,
                "extreme_congestion_rate": float(group[extreme_col].mean()) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def circular_hour_distance(hour: pd.Series, case_hour: int) -> pd.Series:
    difference = (pd.to_numeric(hour, errors="coerce") - case_hour).abs()
    return np.minimum(difference, 24 - difference)


def matched_noactivation_comparison(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    config: PressureConfig,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()

    valid = replay[replay["complete_outcome_window"]].copy()
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    available_features = [f for f in MATCH_FEATURES if f in valid.columns]
    if len(available_features) < 4:
        raise ValueError("Insufficient operational state features for matched-state analysis.")

    numeric = valid[available_features].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scales = numeric.std(ddof=0).replace(0, np.nan).fillna(1.0)
    standardized = (numeric.fillna(medians) - medians) / scales

    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        if bool(episode.actual_oncall_within_window):
            continue
        case_time = pd.Timestamp(episode.alert_start)
        case_candidates = valid.index[valid[TS_COL] == case_time].tolist()
        if not case_candidates:
            continue
        case_idx = case_candidates[0]
        case = valid.loc[case_idx]

        controls = valid[
            (valid[activation_col] == 0)
            & (valid["oncall_pressure_percentile"] < config.pressure_threshold)
        ].copy()
        controls = controls[
            (pd.to_datetime(controls[TS_COL]) - case_time).abs() > pd.Timedelta(hours=24)
        ]
        if "is_weekend" in controls.columns and "is_weekend" in case.index:
            controls = controls[controls["is_weekend"] == case["is_weekend"]]
        if "hour" in controls.columns and "hour" in case.index:
            hour_distance = circular_hour_distance(controls["hour"], int(case["hour"]))
            narrowed = controls[hour_distance <= 2]
            if len(narrowed) >= config.matches:
                controls = narrowed
        if controls.empty:
            continue

        case_vector = standardized.loc[case_idx, available_features].to_numpy(dtype=float)
        control_matrix = standardized.loc[controls.index, available_features].to_numpy(dtype=float)
        distances = np.sqrt(((control_matrix - case_vector) ** 2).mean(axis=1))
        order = np.argsort(distances)[: config.matches]
        matched = controls.iloc[order]
        matched_distances = distances[order]

        row: dict[str, object] = {
            "episode_id": int(episode.episode_id),
            "case_time": case_time,
            "case_pressure_percentile": float(case["oncall_pressure_percentile"]),
            "case_raw_probability": float(case["raw_probability"]),
            "matched_controls": int(len(matched)),
            "mean_match_distance": float(np.mean(matched_distances)),
            "case_major_congestion": bool(case[major_col]),
            "matched_major_congestion_rate": float(matched[major_col].mean()),
            "case_extreme_congestion": bool(case[extreme_col]),
            "matched_extreme_congestion_rate": float(matched[extreme_col].mean()),
        }
        for metric in ("total_tbs", "overflow", "WAITINGADM"):
            future_col = f"future_{config.outcome_hours}h_max_{metric}"
            if future_col in valid.columns:
                case_value = pd.to_numeric(pd.Series([case[future_col]]), errors="coerce").iloc[0]
                control_mean = pd.to_numeric(matched[future_col], errors="coerce").mean()
                row[f"case_future_max_{metric}"] = case_value
                row[f"matched_mean_future_max_{metric}"] = control_mean
                row[f"case_minus_matched_future_max_{metric}"] = case_value - control_mean
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["case_pressure_percentile", "case_time"], ascending=[False, True]
    ) if rows else pd.DataFrame()


def performance_summary(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    config: PressureConfig,
) -> pd.DataFrame:
    valid = replay[replay["complete_outcome_window"]].copy()
    target = f"oncall_within_{config.horizon}h"
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    high = valid[valid["oncall_pressure_percentile"] >= config.pressure_threshold]
    span_days = max(
        (valid[TS_COL].max() - valid[TS_COL].min()).total_seconds() / 86400.0,
        1.0,
    )
    weeks = span_days / 7.0

    metrics: list[tuple[str, object]] = [
        ("horizon_hours", config.horizon),
        ("pressure_percentile_threshold", config.pressure_threshold),
        ("outcome_window_hours", config.outcome_hours),
        ("bad_outcome_quantile", config.bad_outcome_quantile),
        ("major_core_metric_count", config.major_core_count),
        ("replay_rows", int(len(valid))),
        ("replay_start", valid[TS_COL].min()),
        ("replay_end", valid[TS_COL].max()),
        ("baseline_activation_rate", float(valid[activation_col].mean())),
        ("baseline_major_congestion_rate", float(valid[major_col].mean())),
        ("baseline_extreme_congestion_rate", float(valid[extreme_col].mean())),
        ("raw_roc_auc", safe_metric(roc_auc_score, valid[target], valid["raw_probability"].to_numpy())),
        ("raw_average_precision", safe_metric(average_precision_score, valid[target], valid["raw_probability"].to_numpy())),
        ("raw_brier", float(brier_score_loss(valid[target], valid["raw_probability"]))),
        ("calibrated_roc_auc", safe_metric(roc_auc_score, valid[target], valid["calibrated_probability"].to_numpy())),
        ("calibrated_average_precision", safe_metric(average_precision_score, valid[target], valid["calibrated_probability"].to_numpy())),
        ("calibrated_brier", float(brier_score_loss(valid[target], valid["calibrated_probability"]))),
        ("high_pressure_decision_hours", int(len(high))),
        ("high_pressure_actual_activation_rate", float(high[activation_col].mean()) if len(high) else np.nan),
        ("high_pressure_major_congestion_rate", float(high[major_col].mean()) if len(high) else np.nan),
        ("high_pressure_extreme_congestion_rate", float(high[extreme_col].mean()) if len(high) else np.nan),
        ("episode_count", int(len(episodes))),
        ("episodes_per_week", float(len(episodes) / weeks)),
    ]
    if not episodes.empty:
        metrics.extend(
            [
                ("episode_activation_rate", float(episodes["actual_oncall_within_window"].mean())),
                ("episode_major_congestion_rate", float(episodes["major_congestion_within_window"].mean())),
                ("candidate_missed_opportunity_count", int((episodes["episode_class"] == "candidate_missed_opportunity").sum())),
                ("high_pressure_no_major_congestion_count", int((episodes["episode_class"] == "high_pressure_no_major_congestion").sum())),
                ("early_or_concordant_activation_count", int((episodes["episode_class"] == "early_or_concordant_activation").sum())),
            ]
        )
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_analysis(config: PressureConfig):
    if config.horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    if not 0 < config.pressure_threshold <= 100:
        raise ValueError("pressure_threshold must be in (0, 100].")

    full_df = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    decision_df = full_df[full_df["oncall_active"] == 0].copy()
    target = f"oncall_within_{config.horizon}h"
    decision_df = decision_df.dropna(subset=[target]).reset_index(drop=True)
    features, categorical = feature_columns(decision_df)
    for col in categorical:
        decision_df[col] = decision_df[col].fillna("Unknown").astype(str)

    fit, calibration, replay = chronological_three_way_split(decision_df)
    model, calibrator = train_independent_replay_model(
        fit, calibration, features, categorical, config.horizon
    )
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    replay_raw = model.predict_proba(replay[features])[:, 1]
    replay = replay.copy()
    replay["raw_probability"] = replay_raw
    replay["calibrated_probability"] = calibrator.predict(replay_raw)
    replay["oncall_pressure_percentile"] = empirical_percentile(replay_raw, calibration_raw)
    replay["oncall_pressure_band"] = assign_pressure_band(replay["oncall_pressure_percentile"])

    thresholds = bad_outcome_thresholds(
        pd.concat([fit, calibration], ignore_index=True), config.bad_outcome_quantile
    )
    replay = add_future_outcomes(replay, full_df, thresholds, config.outcome_hours)
    replay = add_congestion_composite(replay, config.outcome_hours, config.major_core_count)
    replay["high_pressure_flag"] = (
        replay["complete_outcome_window"]
        & (replay["oncall_pressure_percentile"] >= config.pressure_threshold)
    )

    episodes = cluster_pressure_episodes(replay, config)
    pressure_bands = pressure_band_summary(replay, config)
    thresholds_summary = threshold_summary(replay, config)
    matched = matched_noactivation_comparison(replay, episodes, config)
    summary = performance_summary(replay, episodes, config)
    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    return replay, episodes, top, pressure_bands, thresholds_summary, matched, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrospective percentile-based on-call pressure analysis."
    )
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--pressure-percentile-threshold", type=float, default=DEFAULT_PRESSURE_THRESHOLD
    )
    parser.add_argument(
        "--bad-outcome-quantile", type=float, default=DEFAULT_BAD_OUTCOME_QUANTILE
    )
    parser.add_argument("--outcome-hours", type=int, default=DEFAULT_OUTCOME_HOURS)
    parser.add_argument("--major-core-count", type=int, default=DEFAULT_MAJOR_CORE_COUNT)
    parser.add_argument("--matches", type=int, default=DEFAULT_MATCHES)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PressureConfig(
        horizon=args.horizon,
        pressure_threshold=args.pressure_percentile_threshold,
        bad_outcome_quantile=args.bad_outcome_quantile,
        outcome_hours=args.outcome_hours,
        major_core_count=args.major_core_count,
        matches=args.matches,
        top_n=args.top_n,
    )
    replay, episodes, top, bands, thresholds, matched, summary = run_analysis(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay.to_csv(output_dir / "oncall_pressure_decision_points.csv", index=False)
    episodes.to_csv(output_dir / "oncall_pressure_episodes.csv", index=False)
    top.to_csv(output_dir / "oncall_pressure_top_cases.csv", index=False)
    bands.to_csv(output_dir / "oncall_pressure_bands.csv", index=False)
    thresholds.to_csv(output_dir / "oncall_pressure_thresholds.csv", index=False)
    matched.to_csv(output_dir / "oncall_pressure_matched_noactivation.csv", index=False)
    summary.to_csv(output_dir / "oncall_pressure_performance.csv", index=False)

    print(summary.to_string(index=False))
    print("\nPressure thresholds:")
    print(thresholds.to_string(index=False))
    if not top.empty:
        show = [
            c for c in (
                "alert_start",
                "peak_pressure_percentile",
                "peak_calibrated_probability",
                "episode_class",
                "hours_to_actual_activation",
                "start_total_tbs",
                "subsequent_max_total_tbs",
                "start_overflow",
                "subsequent_max_overflow",
                "start_WAITINGADM",
                "subsequent_max_WAITINGADM",
            )
            if c in top.columns
        ]
        print("\nTop percentile-based on-call pressure episodes:")
        print(top[show].to_string(index=False))


if __name__ == "__main__":
    main()
