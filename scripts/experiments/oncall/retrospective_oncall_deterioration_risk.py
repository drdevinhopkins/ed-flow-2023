from __future__ import annotations

"""Retrospective on-call need replay focused on *deterioration*, not persistence.

The untreated congestion-risk model is useful for identifying busy states, but its
binary outcome (future threshold crossing) is often already nearly determined by the
current ED state. This experiment asks a stricter question: among hours where on-call
is not activated, can the current state predict a clinically meaningful worsening of
physician-facing backlog over the next six hours?

Historical on-call activation is deliberately not the target. Availability and
callability are applied only after the need score is produced.
"""

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
from retrospective_oncall_congestion_risk import train_congestion_model  # noqa: E402
from retrospective_oncall_congestion_risk_leakage_safe import (  # noqa: E402
    is_future_or_outcome_feature,
)

DEFAULT_OUTCOME_HOURS = 6
DEFAULT_BAD_OUTCOME_QUANTILE = 0.90
DEFAULT_POSITIVE_DELTA_QUANTILE = 0.50
DEFAULT_RISK_PERCENTILE_THRESHOLD = 99.0
DEFAULT_MATCHES = 5
DEFAULT_TOP_N = 30

# Physician-facing backlog domains. WAITINGADM is retained as context but deliberately
# excluded from the primary deterioration target because an ED on-call physician cannot
# directly resolve inpatient boarding.
DETERIORATION_METRICS = ("total_tbs", "pod_tbs", "vertical_tbs", "overflow")
CONTEXT_METRICS = ("WAITINGADM",)
MATCH_FEATURES = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "overflow",
    "WAITINGADM",
    "n_pod",
    "n_vertical",
    "n_total_working_excl_oncall",
)


@dataclass(frozen=True)
class DeteriorationConfig:
    outcome_hours: int = DEFAULT_OUTCOME_HOURS
    bad_outcome_quantile: float = DEFAULT_BAD_OUTCOME_QUANTILE
    positive_delta_quantile: float = DEFAULT_POSITIVE_DELTA_QUANTILE
    risk_percentile_threshold: float = DEFAULT_RISK_PERCENTILE_THRESHOLD
    matches: int = DEFAULT_MATCHES
    top_n: int = DEFAULT_TOP_N
    late_call_hour: int = DEFAULT_LATE_CALL_HOUR
    min_rest_buffer_hours: float = DEFAULT_MIN_REST_BUFFER_HOURS


def build_labeled_timeline(full_df: pd.DataFrame, config: DeteriorationConfig):
    decisions = full_df[full_df["oncall_active"] == 0].copy().reset_index(drop=True)
    prefit, precal, _ = chronological_three_way_split(decisions)
    fit_end = pd.to_datetime(prefit[TS_COL]).max()
    calibration_end = pd.to_datetime(precal[TS_COL]).max()

    threshold_metrics = tuple(dict.fromkeys(DETERIORATION_METRICS + CONTEXT_METRICS))
    absolute_thresholds = bad_outcome_thresholds(
        prefit, config.bad_outcome_quantile, metrics=threshold_metrics
    )
    labeled = add_future_outcomes(
        decisions, full_df, absolute_thresholds, config.outcome_hours
    )

    # Current absolute severity is descriptive and becomes a separate rescue-state
    # signal. It is not the deterioration-model target.
    current_flags = []
    for metric in DETERIORATION_METRICS:
        threshold = absolute_thresholds.get(metric)
        if threshold is None or metric not in labeled.columns:
            continue
        flag = f"current_bad_{metric}"
        labeled[flag] = pd.to_numeric(labeled[metric], errors="coerce") >= threshold
        current_flags.append(flag)
    labeled["current_backlog_severity_count"] = (
        labeled[current_flags].fillna(False).astype(int).sum(axis=1)
        if current_flags else 0
    )
    labeled["current_rescue_state"] = labeled["current_backlog_severity_count"] >= 3

    return labeled, prefit, fit_end, calibration_end, absolute_thresholds


def derive_delta_thresholds(
    labeled: pd.DataFrame,
    fit_end: pd.Timestamp,
    config: DeteriorationConfig,
) -> dict[str, float]:
    activation = f"actual_activation_within_{config.outcome_hours}h"
    fit = labeled[
        labeled["complete_outcome_window"].fillna(False)
        & (labeled[activation] == 0)
        & (pd.to_datetime(labeled[TS_COL]) <= fit_end)
    ].copy()
    thresholds: dict[str, float] = {}
    for metric in DETERIORATION_METRICS:
        future_col = f"future_{config.outcome_hours}h_max_{metric}"
        if metric not in fit.columns or future_col not in fit.columns:
            continue
        current = pd.to_numeric(fit[metric], errors="coerce")
        future = pd.to_numeric(fit[future_col], errors="coerce")
        delta = (future - current).dropna()
        positive = delta[delta > 0]
        if positive.empty:
            continue
        threshold = float(positive.quantile(config.positive_delta_quantile))
        thresholds[metric] = max(1.0, threshold)
    if len(thresholds) < len(DETERIORATION_METRICS):
        missing = sorted(set(DETERIORATION_METRICS) - set(thresholds))
        raise ValueError(f"Could not derive deterioration thresholds for {missing}")
    return thresholds


def add_deterioration_labels(
    labeled: pd.DataFrame,
    delta_thresholds: dict[str, float],
    config: DeteriorationConfig,
) -> pd.DataFrame:
    out = labeled.copy()
    flags: dict[str, pd.Series] = {}
    for metric, threshold in delta_thresholds.items():
        future_col = f"future_{config.outcome_hours}h_max_{metric}"
        delta_col = f"future_{config.outcome_hours}h_delta_{metric}"
        flag_col = f"future_{config.outcome_hours}h_deterioration_{metric}"
        out[delta_col] = (
            pd.to_numeric(out[future_col], errors="coerce")
            - pd.to_numeric(out[metric], errors="coerce")
        )
        out[flag_col] = out[delta_col] >= threshold
        flags[metric] = out[flag_col]

    out[f"deterioration_count_within_{config.outcome_hours}h"] = (
        pd.DataFrame(flags).fillna(False).astype(int).sum(axis=1)
    )

    # A substantial deterioration requires total TBS to rise meaningfully, at least
    # one of POD/vertical backlog to rise, and either overflow to rise or Total TBS to
    # make a particularly large move (roughly the upper quartile of positive fit moves).
    total_flag = flags["total_tbs"]
    zone_flag = flags["pod_tbs"] | flags["vertical_tbs"]
    overflow_flag = flags["overflow"]
    total_delta = out[f"future_{config.outcome_hours}h_delta_total_tbs"]
    large_total_move = total_delta >= (2.0 * delta_thresholds["total_tbs"])
    target = f"major_deterioration_within_{config.outcome_hours}h"
    out[target] = total_flag & zone_flag & (overflow_flag | large_total_move)
    return out


def leakage_safe_model_frame(df: pd.DataFrame):
    model_df, features, categorical = prepare_latent_need_model_frame(df)
    extra_prefixes = (
        "current_bad_",
        "current_backlog_severity_count",
        "current_rescue_state",
        "deterioration_count_within_",
        "major_deterioration_within_",
    )
    features = [
        feature for feature in features
        if not is_future_or_outcome_feature(feature)
        and not feature.startswith(extra_prefixes)
    ]
    categorical = [feature for feature in categorical if feature in features]
    forbidden = [
        feature for feature in features
        if is_future_or_outcome_feature(feature)
        or feature.startswith(extra_prefixes)
    ]
    if forbidden:
        raise AssertionError(f"Outcome leakage features remain: {forbidden}")
    if not features:
        raise ValueError("No leakage-safe deterioration-model features remain.")
    return model_df, features, categorical


def split_untreated(
    model_df: pd.DataFrame,
    fit_end: pd.Timestamp,
    calibration_end: pd.Timestamp,
    config: DeteriorationConfig,
):
    activation = f"actual_activation_within_{config.outcome_hours}h"
    eligible = model_df[
        model_df["complete_outcome_window"].fillna(False)
        & (model_df[activation] == 0)
    ].copy()
    ts = pd.to_datetime(eligible[TS_COL])
    fit = eligible[ts <= fit_end].copy()
    calibration = eligible[(ts > fit_end) & (ts <= calibration_end)].copy()
    replay = model_df[
        model_df["complete_outcome_window"].fillna(False)
        & (pd.to_datetime(model_df[TS_COL]) > calibration_end)
    ].copy()
    if fit.empty or calibration.empty or replay.empty:
        raise ValueError("Insufficient fit/calibration/replay rows for deterioration model")
    return fit, calibration, replay


def score_deterioration(full_df: pd.DataFrame, config: DeteriorationConfig):
    labeled, _, fit_end, calibration_end, absolute_thresholds = build_labeled_timeline(
        full_df, config
    )
    delta_thresholds = derive_delta_thresholds(labeled, fit_end, config)
    labeled = add_deterioration_labels(labeled, delta_thresholds, config)
    model_df, features, categorical = leakage_safe_model_frame(labeled)
    fit, calibration, replay = split_untreated(
        model_df, fit_end, calibration_end, config
    )
    target = f"major_deterioration_within_{config.outcome_hours}h"
    model, calibrator = train_congestion_model(
        fit, calibration, features, categorical, target
    )
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    replay_raw = model.predict_proba(replay[features])[:, 1]
    replay = replay.copy()
    replay["untreated_deterioration_raw_score"] = replay_raw
    replay["untreated_deterioration_probability"] = calibrator.predict(replay_raw)
    replay["deterioration_pressure_percentile"] = empirical_percentile(
        replay_raw, calibration_raw
    )
    replay["deterioration_pressure_band"] = assign_pressure_band(
        replay["deterioration_pressure_percentile"]
    )
    replay["deterioration_model_feature_count"] = len(features)
    return replay, fit, calibration, target, delta_thresholds, absolute_thresholds


def classify_episode(peak: pd.Series, activation: bool, realized: bool) -> str:
    if activation:
        return "deterioration_risk_concordant_activation"
    status = str(peak.get("callability_status", "unknown"))
    if status == "unavailable_no_oncall_scheduled":
        return "deterioration_risk_unavailable"
    if status.startswith("constrained_"):
        return "deterioration_risk_constrained"
    if realized:
        return "callable_deterioration_realized"
    return "callable_deterioration_not_realized"


def cluster_episodes(replay: pd.DataFrame, target: str, config: DeteriorationConfig):
    alerts = replay[
        replay["complete_outcome_window"].fillna(False)
        & (replay["deterioration_pressure_percentile"] >= config.risk_percentile_threshold)
    ].copy()
    if alerts.empty:
        return pd.DataFrame()
    alerts = alerts.sort_values(TS_COL).reset_index(drop=True)
    gaps = alerts[TS_COL].diff().dt.total_seconds().div(3600)
    alerts["episode_id"] = (gaps.isna() | (gaps > 1.0)).cumsum()
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for episode_id, group in alerts.groupby("episode_id", sort=True):
        peak = group.sort_values(
            ["deterioration_pressure_percentile", "untreated_deterioration_raw_score"],
            ascending=False,
        ).iloc[0]
        activation = bool(group[activation_col].max())
        realized = bool(group[target].max())
        row: dict[str, object] = {
            "episode_id": int(episode_id),
            "alert_start": group[TS_COL].min(),
            "alert_end": group[TS_COL].max(),
            "alert_duration_hours": int(len(group)),
            "peak_risk_time": peak[TS_COL],
            "peak_deterioration_pressure_percentile": float(
                group["deterioration_pressure_percentile"].max()
            ),
            "peak_untreated_deterioration_probability": float(
                peak["untreated_deterioration_probability"]
            ),
            "callability_status_at_peak": peak["callability_status"],
            "oncall_scheduled_at_peak": bool(peak["oncall_scheduled"]),
            "oncall_physician_id_at_peak": peak["oncall_physician_id"],
            "hours_to_next_non_oncall_shift_at_peak": peak["hours_to_next_non_oncall_shift"],
            "actual_oncall_within_window": activation,
            "major_deterioration_within_window": realized,
            "current_rescue_state_at_peak": bool(peak["current_rescue_state"]),
            "episode_class": classify_episode(peak, activation, realized),
            "hours_to_actual_activation": pd.to_numeric(
                group["hours_to_actual_activation"], errors="coerce"
            ).min(),
        }
        for metric in DETERIORATION_METRICS + CONTEXT_METRICS:
            if metric in peak.index:
                row[f"peak_{metric}"] = peak[metric]
            delta_col = f"future_{config.outcome_hours}h_delta_{metric}"
            max_col = f"future_{config.outcome_hours}h_max_{metric}"
            if delta_col in peak.index:
                row[f"future_delta_{metric}"] = peak[delta_col]
            if max_col in peak.index:
                row[f"future_max_{metric}"] = peak[max_col]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["peak_deterioration_pressure_percentile", "peak_risk_time"],
        ascending=[False, True],
    )


def matched_callable(replay: pd.DataFrame, episodes: pd.DataFrame, target: str, config: DeteriorationConfig):
    if episodes.empty:
        return pd.DataFrame()
    activation = f"actual_activation_within_{config.outcome_hours}h"
    valid = replay[replay["complete_outcome_window"].fillna(False)].copy()
    features = [feature for feature in MATCH_FEATURES if feature in valid.columns]
    numeric = valid[features].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scales = numeric.std(ddof=0).replace(0, np.nan).fillna(1.0)
    standardized = (numeric.fillna(medians) - medians) / scales
    rows: list[dict[str, object]] = []

    cases = episodes[
        episodes["episode_class"] == "callable_deterioration_realized"
    ]
    for episode in cases.itertuples(index=False):
        case_time = pd.Timestamp(episode.peak_risk_time)
        candidates = valid.index[valid[TS_COL] == case_time].tolist()
        if not candidates:
            continue
        case_idx = candidates[0]
        case = valid.loc[case_idx]
        controls = valid[
            (valid[activation] == 0)
            & valid["oncall_callable_proxy"]
            & (valid["deterioration_pressure_percentile"] < config.risk_percentile_threshold)
        ].copy()
        controls = controls[
            (pd.to_datetime(controls[TS_COL]) - case_time).abs() > pd.Timedelta(hours=24)
        ]
        if "is_weekend" in controls.columns:
            controls = controls[controls["is_weekend"] == case["is_weekend"]]
        if "hour" in controls.columns:
            hour_distance = circular_hour_distance(controls["hour"], int(case["hour"]))
            narrowed = controls[hour_distance <= 2]
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
            "case_pressure_percentile": float(case["deterioration_pressure_percentile"]),
            "matched_controls": int(len(matched)),
            "mean_match_distance": float(np.mean(distances[order])),
            "case_major_deterioration": bool(case[target]),
            "matched_major_deterioration_rate": float(matched[target].mean()),
        }
        for metric in DETERIORATION_METRICS + CONTEXT_METRICS:
            delta_col = f"future_{config.outcome_hours}h_delta_{metric}"
            if delta_col in valid.columns:
                case_delta = pd.to_numeric(pd.Series([case[delta_col]]), errors="coerce").iloc[0]
                matched_delta = pd.to_numeric(matched[delta_col], errors="coerce").mean()
                row[f"case_future_delta_{metric}"] = case_delta
                row[f"matched_mean_future_delta_{metric}"] = matched_delta
                row[f"case_minus_matched_future_delta_{metric}"] = case_delta - matched_delta
        rows.append(row)
    return pd.DataFrame(rows)


def band_summary(replay: pd.DataFrame, target: str, config: DeteriorationConfig):
    activation = f"actual_activation_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for band in ("routine", "elevated", "high", "very_high"):
        group = replay[replay["deterioration_pressure_band"] == band]
        untreated = group[group[activation] == 0]
        if group.empty:
            continue
        rows.append({
            "deterioration_pressure_band": band,
            "decision_hours": int(len(group)),
            "untreated_decision_hours": int(len(untreated)),
            "actual_activation_rate": float(group[activation].mean()),
            "realized_deterioration_rate_all": float(group[target].mean()),
            "realized_deterioration_rate_untreated": float(untreated[target].mean()) if len(untreated) else np.nan,
            "current_rescue_state_rate": float(group["current_rescue_state"].mean()),
            "median_deterioration_probability": float(group["untreated_deterioration_probability"].median()),
        })
    return pd.DataFrame(rows)


def performance_summary(replay: pd.DataFrame, episodes: pd.DataFrame, matched: pd.DataFrame, target: str, delta_thresholds: dict[str, float], config: DeteriorationConfig):
    activation = f"actual_activation_within_{config.outcome_hours}h"
    untreated = replay[replay[activation] == 0].copy()
    high = replay[replay["deterioration_pressure_percentile"] >= config.risk_percentile_threshold]
    high_untreated = high[high[activation] == 0]
    y = untreated[target].astype(int)
    raw = untreated["untreated_deterioration_raw_score"].to_numpy()
    prob = untreated["untreated_deterioration_probability"].to_numpy()
    metrics: list[tuple[str, object]] = [
        ("outcome_window_hours", config.outcome_hours),
        ("risk_percentile_threshold", config.risk_percentile_threshold),
        ("positive_delta_quantile", config.positive_delta_quantile),
        ("untreated_replay_rows", int(len(untreated))),
        ("untreated_deterioration_base_rate", float(y.mean())),
        ("untreated_roc_auc", safe_metric(roc_auc_score, y, raw)),
        ("untreated_average_precision", safe_metric(average_precision_score, y, raw)),
        ("untreated_brier_calibrated", float(brier_score_loss(y, prob))),
        ("high_risk_decision_hours", int(len(high))),
        ("high_risk_untreated_hours", int(len(high_untreated))),
        ("high_risk_realized_deterioration_rate_untreated", float(high_untreated[target].mean()) if len(high_untreated) else np.nan),
        ("high_risk_actual_activation_rate", float(high[activation].mean()) if len(high) else np.nan),
        ("current_rescue_state_rate_replay", float(replay["current_rescue_state"].mean())),
        ("episode_count", int(len(episodes))),
    ]
    for metric, threshold in delta_thresholds.items():
        metrics.append((f"fit_positive_delta_threshold_{metric}", threshold))
    if not episodes.empty:
        counts = episodes["episode_class"].value_counts()
        for label in (
            "deterioration_risk_concordant_activation",
            "callable_deterioration_realized",
            "callable_deterioration_not_realized",
            "deterioration_risk_unavailable",
            "deterioration_risk_constrained",
        ):
            metrics.append((f"episode_count_{label}", int(counts.get(label, 0))))
    if not matched.empty:
        metrics.extend([
            ("matched_control_mean_deterioration_rate", float(matched["matched_major_deterioration_rate"].mean())),
            ("matched_case_minus_control_total_tbs_delta", float(matched["case_minus_matched_future_delta_total_tbs"].mean())),
            ("matched_case_minus_control_overflow_delta", float(matched["case_minus_matched_future_delta_overflow"].mean())),
        ])
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_analysis(config: DeteriorationConfig):
    full = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    full = add_callability_features(full, config.late_call_hour, config.min_rest_buffer_hours)
    replay, fit, calibration, target, delta_thresholds, absolute_thresholds = score_deterioration(full, config)
    episodes = cluster_episodes(replay, target, config)
    matched = matched_callable(replay, episodes, target, config)
    bands = band_summary(replay, target, config)
    summary = performance_summary(replay, episodes, matched, target, delta_thresholds, config)
    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    threshold_rows = [
        {"metric": metric, "absolute_bad_threshold": absolute_thresholds.get(metric), "positive_delta_threshold": delta_thresholds.get(metric)}
        for metric in tuple(dict.fromkeys(DETERIORATION_METRICS + CONTEXT_METRICS))
    ]
    thresholds = pd.DataFrame(threshold_rows)
    return replay, episodes, top, matched, bands, summary, thresholds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage-safe untreated deterioration replay for ED on-call need.")
    p.add_argument("--outcome-hours", type=int, default=DEFAULT_OUTCOME_HOURS)
    p.add_argument("--bad-outcome-quantile", type=float, default=DEFAULT_BAD_OUTCOME_QUANTILE)
    p.add_argument("--positive-delta-quantile", type=float, default=DEFAULT_POSITIVE_DELTA_QUANTILE)
    p.add_argument("--risk-percentile-threshold", type=float, default=DEFAULT_RISK_PERCENTILE_THRESHOLD)
    p.add_argument("--matches", type=int, default=DEFAULT_MATCHES)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--late-call-hour", type=int, default=DEFAULT_LATE_CALL_HOUR)
    p.add_argument("--min-rest-buffer-hours", type=float, default=DEFAULT_MIN_REST_BUFFER_HOURS)
    p.add_argument("--output-dir", default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = DeteriorationConfig(
        outcome_hours=args.outcome_hours,
        bad_outcome_quantile=args.bad_outcome_quantile,
        positive_delta_quantile=args.positive_delta_quantile,
        risk_percentile_threshold=args.risk_percentile_threshold,
        matches=args.matches,
        top_n=args.top_n,
        late_call_hour=args.late_call_hour,
        min_rest_buffer_hours=args.min_rest_buffer_hours,
    )
    replay, episodes, top, matched, bands, summary, thresholds = run_analysis(config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay.to_csv(out / "oncall_deterioration_risk_decision_points.csv", index=False)
    episodes.to_csv(out / "oncall_deterioration_risk_episodes.csv", index=False)
    top.to_csv(out / "oncall_deterioration_risk_top_cases.csv", index=False)
    matched.to_csv(out / "oncall_deterioration_risk_matched_callable.csv", index=False)
    bands.to_csv(out / "oncall_deterioration_risk_bands.csv", index=False)
    summary.to_csv(out / "oncall_deterioration_risk_performance.csv", index=False)
    thresholds.to_csv(out / "oncall_deterioration_risk_thresholds.csv", index=False)
    print(summary.to_string(index=False))
    print("\nDeterioration thresholds:")
    print(thresholds.to_string(index=False))
    print("\nDeterioration risk bands:")
    print(bands.to_string(index=False))
    if not top.empty:
        cols = [c for c in (
            "alert_start", "peak_risk_time", "peak_deterioration_pressure_percentile",
            "peak_untreated_deterioration_probability", "callability_status_at_peak",
            "episode_class", "current_rescue_state_at_peak", "peak_total_tbs",
            "future_delta_total_tbs", "peak_overflow", "future_delta_overflow",
            "peak_WAITINGADM", "future_max_WAITINGADM",
        ) if c in top.columns]
        print("\nTop deterioration-risk episodes:")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
