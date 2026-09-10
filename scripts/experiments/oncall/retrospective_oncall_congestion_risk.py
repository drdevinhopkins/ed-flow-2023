from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for path in (SCRIPTS_DIR, EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from forecast_oncall_probability import (  # noqa: E402
    CATBOOST_TASK_TYPE,
    RANDOM_SEED,
    TS_COL,
    add_horizon_targets,
    add_time_and_trend_features,
    load_dataset,
)
from retrospective_oncall_decision_replay import (  # noqa: E402
    add_future_outcomes,
    bad_outcome_thresholds,
    chronological_three_way_split,
    safe_metric,
)
from retrospective_oncall_pressure_analysis import (  # noqa: E402
    MATCH_FEATURES,
    add_congestion_composite,
    assign_pressure_band,
    circular_hour_distance,
    empirical_percentile,
)
from retrospective_oncall_availability_adjusted import (  # noqa: E402
    DEFAULT_LATE_CALL_HOUR,
    DEFAULT_MIN_REST_BUFFER_HOURS,
    add_callability_features,
    prepare_latent_need_model_frame,
)

DEFAULT_OUTCOME_HOURS = 6
DEFAULT_BAD_OUTCOME_QUANTILE = 0.90
DEFAULT_MAJOR_CORE_COUNT = 3
DEFAULT_RISK_PERCENTILE_THRESHOLD = 99.0
DEFAULT_MATCHES = 5
DEFAULT_TOP_N = 30


@dataclass(frozen=True)
class CongestionRiskConfig:
    outcome_hours: int = DEFAULT_OUTCOME_HOURS
    bad_outcome_quantile: float = DEFAULT_BAD_OUTCOME_QUANTILE
    major_core_count: int = DEFAULT_MAJOR_CORE_COUNT
    risk_percentile_threshold: float = DEFAULT_RISK_PERCENTILE_THRESHOLD
    matches: int = DEFAULT_MATCHES
    top_n: int = DEFAULT_TOP_N
    late_call_hour: int = DEFAULT_LATE_CALL_HOUR
    min_rest_buffer_hours: float = DEFAULT_MIN_REST_BUFFER_HOURS


def train_congestion_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    target: str,
) -> tuple[CatBoostClassifier, IsotonicRegression]:
    cat_indices = [features.index(c) for c in categorical]
    params: dict[str, object] = {
        "iterations": 700,
        "depth": 7,
        "learning_rate": 0.04,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": RANDOM_SEED,
        "auto_class_weights": "Balanced",
        "verbose": False,
        "allow_writing_files": False,
        "task_type": CATBOOST_TASK_TYPE,
    }
    if CATBOOST_TASK_TYPE == "GPU":
        params["devices"] = "0"
    model = CatBoostClassifier(**params)
    model.fit(
        fit[features],
        fit[target].astype(int),
        cat_features=cat_indices,
        eval_set=(calibration[features], calibration[target].astype(int)),
        early_stopping_rounds=75,
        verbose=False,
    )
    raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw, calibration[target].astype(int))
    return model, calibrator


def build_labeled_timeline(
    full_df: pd.DataFrame,
    config: CongestionRiskConfig,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    decisions = full_df[full_df["oncall_active"] == 0].copy().reset_index(drop=True)
    prefit, precal, _ = chronological_three_way_split(decisions)
    fit_end = pd.to_datetime(prefit[TS_COL]).max()
    calibration_end = pd.to_datetime(precal[TS_COL]).max()

    thresholds = bad_outcome_thresholds(prefit, config.bad_outcome_quantile)
    labeled = add_future_outcomes(decisions, full_df, thresholds, config.outcome_hours)
    labeled = add_congestion_composite(labeled, config.outcome_hours, config.major_core_count)
    return labeled, fit_end, calibration_end


def split_untreated_training(
    labeled: pd.DataFrame,
    fit_end: pd.Timestamp,
    calibration_end: pd.Timestamp,
    config: CongestionRiskConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activation = f"actual_activation_within_{config.outcome_hours}h"
    complete = labeled["complete_outcome_window"].fillna(False)
    untreated = labeled[complete & (labeled[activation] == 0)].copy()
    ts = pd.to_datetime(untreated[TS_COL])
    fit = untreated[ts <= fit_end].copy()
    calibration = untreated[(ts > fit_end) & (ts <= calibration_end)].copy()

    replay = labeled[
        labeled["complete_outcome_window"]
        & (pd.to_datetime(labeled[TS_COL]) > calibration_end)
    ].copy()
    if fit.empty or calibration.empty or replay.empty:
        raise ValueError("Insufficient fit/calibration/replay rows for untreated congestion model.")
    return fit, calibration, replay


def score_congestion_risk(
    full_df: pd.DataFrame,
    config: CongestionRiskConfig,
):
    labeled, fit_end, calibration_end = build_labeled_timeline(full_df, config)
    model_df, features, categorical = prepare_latent_need_model_frame(labeled)
    fit, calibration, replay = split_untreated_training(
        model_df, fit_end, calibration_end, config
    )
    target = f"major_congestion_within_{config.outcome_hours}h"
    model, calibrator = train_congestion_model(fit, calibration, features, categorical, target)
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    replay_raw = model.predict_proba(replay[features])[:, 1]
    replay = replay.copy()
    replay["untreated_congestion_raw_score"] = replay_raw
    replay["untreated_congestion_probability"] = calibrator.predict(replay_raw)
    replay["oncall_need_pressure_percentile"] = empirical_percentile(replay_raw, calibration_raw)
    replay["oncall_need_pressure_band"] = assign_pressure_band(replay["oncall_need_pressure_percentile"])
    return replay, fit, calibration, target


def episode_class(peak: pd.Series, activation: bool, realized_major: bool) -> str:
    if activation:
        return "high_risk_concordant_activation"
    status = str(peak.get("callability_status", "unknown"))
    if status == "unavailable_no_oncall_scheduled":
        return "high_risk_unavailable"
    if status.startswith("constrained_"):
        return "high_risk_constrained"
    if realized_major:
        return "callable_high_risk_realized_congestion"
    return "callable_high_risk_no_realized_congestion"


def cluster_episodes(replay: pd.DataFrame, config: CongestionRiskConfig) -> pd.DataFrame:
    alerts = replay[
        replay["complete_outcome_window"]
        & (replay["oncall_need_pressure_percentile"] >= config.risk_percentile_threshold)
    ].copy()
    if alerts.empty:
        return pd.DataFrame()
    alerts = alerts.sort_values(TS_COL).reset_index(drop=True)
    gaps = alerts[TS_COL].diff().dt.total_seconds().div(3600)
    alerts["episode_id"] = (gaps.isna() | (gaps > 1.0)).cumsum()
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"

    rows: list[dict[str, object]] = []
    for episode_id, group in alerts.groupby("episode_id", sort=True):
        peak = group.sort_values(
            ["oncall_need_pressure_percentile", "untreated_congestion_raw_score"],
            ascending=False,
        ).iloc[0]
        activation = bool(group[activation_col].max())
        realized_major = bool(group[major_col].max())
        realized_extreme = bool(group[extreme_col].max())
        row = {
            "episode_id": int(episode_id),
            "alert_start": group[TS_COL].min(),
            "alert_end": group[TS_COL].max(),
            "alert_duration_hours": int(len(group)),
            "peak_risk_time": peak[TS_COL],
            "peak_need_pressure_percentile": float(group["oncall_need_pressure_percentile"].max()),
            "peak_untreated_congestion_probability": float(peak["untreated_congestion_probability"]),
            "peak_untreated_congestion_raw_score": float(peak["untreated_congestion_raw_score"]),
            "callability_status_at_peak": peak["callability_status"],
            "oncall_scheduled_at_peak": bool(peak["oncall_scheduled"]),
            "oncall_physician_id_at_peak": peak["oncall_physician_id"],
            "hours_to_next_non_oncall_shift_at_peak": peak["hours_to_next_non_oncall_shift"],
            "actual_oncall_within_window": activation,
            "major_congestion_within_window": realized_major,
            "extreme_congestion_within_window": realized_extreme,
            "episode_class": episode_class(peak, activation, realized_major),
            "hours_to_actual_activation": pd.to_numeric(
                group["hours_to_actual_activation"], errors="coerce"
            ).min(),
        }
        for metric in ("total_tbs", "pod_tbs", "vertical_tbs", "overflow", "WAITINGADM"):
            if metric in peak.index:
                row[f"peak_{metric}"] = peak[metric]
            future_col = f"future_{config.outcome_hours}h_max_{metric}"
            if future_col in group.columns:
                row[f"subsequent_max_{metric}"] = pd.to_numeric(
                    group[future_col], errors="coerce"
                ).max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["peak_need_pressure_percentile", "peak_untreated_congestion_raw_score", "alert_start"],
        ascending=[False, False, True],
    )


def risk_band_summary(replay: pd.DataFrame, config: CongestionRiskConfig) -> pd.DataFrame:
    target = f"major_congestion_within_{config.outcome_hours}h"
    activation = f"actual_activation_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for band in ("routine", "elevated", "high", "very_high"):
        group = replay[replay["oncall_need_pressure_band"] == band]
        untreated = group[group[activation] == 0]
        if group.empty:
            continue
        rows.append({
            "need_pressure_band": band,
            "decision_hours": int(len(group)),
            "untreated_decision_hours": int(len(untreated)),
            "actual_activation_rate": float(group[activation].mean()),
            "realized_major_congestion_rate_all": float(group[target].mean()),
            "realized_major_congestion_rate_untreated": float(untreated[target].mean()) if len(untreated) else np.nan,
            "median_untreated_congestion_probability": float(group["untreated_congestion_probability"].median()),
        })
    return pd.DataFrame(rows)


def matched_high_risk_callable(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    config: CongestionRiskConfig,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    target = f"major_congestion_within_{config.outcome_hours}h"
    activation = f"actual_activation_within_{config.outcome_hours}h"
    valid = replay[replay["complete_outcome_window"]].copy()
    features = [f for f in MATCH_FEATURES if f in valid.columns and f != "n_total_scheduled"]
    if "n_total_working_excl_oncall" in valid.columns:
        features.append("n_total_working_excl_oncall")
    numeric = valid[features].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scales = numeric.std(ddof=0).replace(0, np.nan).fillna(1.0)
    standardized = (numeric.fillna(medians) - medians) / scales

    rows: list[dict[str, object]] = []
    cases = episodes[
        episodes["episode_class"] == "callable_high_risk_realized_congestion"
    ]
    for episode in cases.itertuples(index=False):
        case_time = pd.Timestamp(episode.peak_risk_time)
        idxs = valid.index[valid[TS_COL] == case_time].tolist()
        if not idxs:
            continue
        case_idx = idxs[0]
        case = valid.loc[case_idx]
        controls = valid[
            (valid[activation] == 0)
            & valid["oncall_callable_proxy"]
            & (valid["oncall_need_pressure_percentile"] < config.risk_percentile_threshold)
        ].copy()
        controls = controls[
            (pd.to_datetime(controls[TS_COL]) - case_time).abs() > pd.Timedelta(hours=24)
        ]
        if "is_weekend" in controls.columns:
            controls = controls[controls["is_weekend"] == case["is_weekend"]]
        if "hour" in controls.columns:
            hd = circular_hour_distance(controls["hour"], int(case["hour"]))
            narrowed = controls[hd <= 2]
            if len(narrowed) >= config.matches:
                controls = narrowed
        if controls.empty:
            continue
        case_vec = standardized.loc[case_idx, features].to_numpy(dtype=float)
        matrix = standardized.loc[controls.index, features].to_numpy(dtype=float)
        distances = np.sqrt(((matrix - case_vec) ** 2).mean(axis=1))
        order = np.argsort(distances)[: config.matches]
        matched = controls.iloc[order]
        row: dict[str, object] = {
            "episode_id": int(episode.episode_id),
            "case_time": case_time,
            "case_pressure_percentile": float(case["oncall_need_pressure_percentile"]),
            "case_untreated_congestion_probability": float(case["untreated_congestion_probability"]),
            "matched_controls": int(len(matched)),
            "mean_match_distance": float(np.mean(distances[order])),
            "case_major_congestion": bool(case[target]),
            "matched_major_congestion_rate": float(matched[target].mean()),
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
    return pd.DataFrame(rows)


def performance_summary(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    target: str,
    config: CongestionRiskConfig,
) -> pd.DataFrame:
    activation = f"actual_activation_within_{config.outcome_hours}h"
    untreated = replay[replay[activation] == 0].copy()
    high = replay[replay["oncall_need_pressure_percentile"] >= config.risk_percentile_threshold]
    high_untreated = high[high[activation] == 0]
    y = untreated[target].astype(int)
    raw = untreated["untreated_congestion_raw_score"].to_numpy()
    prob = untreated["untreated_congestion_probability"].to_numpy()
    metrics: list[tuple[str, object]] = [
        ("outcome_window_hours", config.outcome_hours),
        ("risk_percentile_threshold", config.risk_percentile_threshold),
        ("late_call_hour_proxy", config.late_call_hour),
        ("min_rest_buffer_hours_proxy", config.min_rest_buffer_hours),
        ("replay_rows", int(len(replay))),
        ("untreated_replay_rows", int(len(untreated))),
        ("untreated_major_congestion_base_rate", float(y.mean())),
        ("untreated_roc_auc", safe_metric(roc_auc_score, y, raw)),
        ("untreated_average_precision", safe_metric(average_precision_score, y, raw)),
        ("untreated_brier_calibrated", float(brier_score_loss(y, prob))),
        ("high_risk_decision_hours", int(len(high))),
        ("high_risk_untreated_hours", int(len(high_untreated))),
        ("high_risk_realized_major_congestion_rate_untreated", float(high_untreated[target].mean()) if len(high_untreated) else np.nan),
        ("high_risk_actual_activation_rate", float(high[activation].mean()) if len(high) else np.nan),
        ("episode_count", int(len(episodes))),
    ]
    if not episodes.empty:
        counts = episodes["episode_class"].value_counts()
        for label in (
            "high_risk_concordant_activation",
            "callable_high_risk_realized_congestion",
            "callable_high_risk_no_realized_congestion",
            "high_risk_unavailable",
            "high_risk_constrained",
        ):
            metrics.append((f"episode_count_{label}", int(counts.get(label, 0))))
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_analysis(config: CongestionRiskConfig):
    full = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    full = add_callability_features(full, config.late_call_hour, config.min_rest_buffer_hours)
    replay, fit, calibration, target = score_congestion_risk(full, config)
    episodes = cluster_episodes(replay, config)
    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    bands = risk_band_summary(replay, config)
    matched = matched_high_risk_callable(replay, episodes, config)
    summary = performance_summary(replay, episodes, target, config)
    return replay, episodes, top, bands, matched, summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retrospective untreated-congestion risk model for on-call decision support."
    )
    p.add_argument("--outcome-hours", type=int, default=DEFAULT_OUTCOME_HOURS)
    p.add_argument("--bad-outcome-quantile", type=float, default=DEFAULT_BAD_OUTCOME_QUANTILE)
    p.add_argument("--major-core-count", type=int, default=DEFAULT_MAJOR_CORE_COUNT)
    p.add_argument("--risk-percentile-threshold", type=float, default=DEFAULT_RISK_PERCENTILE_THRESHOLD)
    p.add_argument("--matches", type=int, default=DEFAULT_MATCHES)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--late-call-hour", type=int, default=DEFAULT_LATE_CALL_HOUR)
    p.add_argument("--min-rest-buffer-hours", type=float, default=DEFAULT_MIN_REST_BUFFER_HOURS)
    p.add_argument("--output-dir", default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = CongestionRiskConfig(
        outcome_hours=args.outcome_hours,
        bad_outcome_quantile=args.bad_outcome_quantile,
        major_core_count=args.major_core_count,
        risk_percentile_threshold=args.risk_percentile_threshold,
        matches=args.matches,
        top_n=args.top_n,
        late_call_hour=args.late_call_hour,
        min_rest_buffer_hours=args.min_rest_buffer_hours,
    )
    replay, episodes, top, bands, matched, summary = run_analysis(config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay.to_csv(out / "oncall_congestion_risk_decision_points.csv", index=False)
    episodes.to_csv(out / "oncall_congestion_risk_episodes.csv", index=False)
    top.to_csv(out / "oncall_congestion_risk_top_cases.csv", index=False)
    bands.to_csv(out / "oncall_congestion_risk_bands.csv", index=False)
    matched.to_csv(out / "oncall_congestion_risk_matched_callable.csv", index=False)
    summary.to_csv(out / "oncall_congestion_risk_performance.csv", index=False)
    print(summary.to_string(index=False))
    print("\nRisk bands:")
    print(bands.to_string(index=False))
    if not top.empty:
        cols = [c for c in (
            "alert_start", "peak_risk_time", "peak_need_pressure_percentile",
            "peak_untreated_congestion_probability", "callability_status_at_peak",
            "episode_class", "oncall_physician_id_at_peak", "actual_oncall_within_window",
            "major_congestion_within_window", "peak_total_tbs", "subsequent_max_total_tbs",
            "peak_overflow", "subsequent_max_overflow", "peak_WAITINGADM",
            "subsequent_max_WAITINGADM",
        ) if c in top.columns]
        print("\nTop untreated-congestion risk episodes:")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
