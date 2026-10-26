from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from forecast_oncall_probability import (  # noqa: E402
    CATBOOST_TASK_TYPE,
    HORIZONS,
    RANDOM_SEED,
    TS_COL,
    add_horizon_targets,
    add_time_and_trend_features,
    feature_columns,
    load_dataset,
)

FIT_FRACTION = 0.70
CALIBRATION_FRACTION = 0.10
DEFAULT_HORIZON = 6
DEFAULT_ALERT_THRESHOLD = 0.70
DEFAULT_BAD_OUTCOME_QUANTILE = 0.90
DEFAULT_OUTCOME_HOURS = 6
DEFAULT_TOP_N = 20
FLOW_METRICS = (
    "total_tbs", "pod_tbs", "vertical_tbs", "stretcher_occupancy", "overflow", "WAITINGADM",
)

@dataclass(frozen=True)
class ReplayConfig:
    horizon: int = DEFAULT_HORIZON
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD
    bad_outcome_quantile: float = DEFAULT_BAD_OUTCOME_QUANTILE
    outcome_hours: int = DEFAULT_OUTCOME_HOURS
    top_n: int = DEFAULT_TOP_N


def chronological_three_way_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    fit_end = int(n * FIT_FRACTION)
    calibration_end = int(n * (FIT_FRACTION + CALIBRATION_FRACTION))
    if fit_end <= 0 or calibration_end <= fit_end or calibration_end >= n:
        raise ValueError("Insufficient rows for fit/calibration/replay split.")
    return df.iloc[:fit_end].copy(), df.iloc[fit_end:calibration_end].copy(), df.iloc[calibration_end:].copy()


def safe_metric(metric_fn, y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(metric_fn(y_true, y_prob))


def train_independent_replay_model(fit, calibration, features, categorical, horizon):
    target = f"oncall_within_{horizon}h"
    cat_indices = [features.index(c) for c in categorical]
    params = {
        "iterations": 700, "depth": 7, "learning_rate": 0.04, "loss_function": "Logloss",
        "eval_metric": "AUC", "random_seed": RANDOM_SEED, "auto_class_weights": "Balanced",
        "verbose": False, "allow_writing_files": False, "task_type": CATBOOST_TASK_TYPE,
    }
    if CATBOOST_TASK_TYPE == "GPU":
        params["devices"] = "0"
    model = CatBoostClassifier(**params)
    model.fit(fit[features], fit[target].astype(int), cat_features=cat_indices,
              eval_set=(calibration[features], calibration[target].astype(int)),
              early_stopping_rounds=75, verbose=False)
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calibration_raw, calibration[target].astype(int))
    return model, calibrator


def bad_outcome_thresholds(history: pd.DataFrame, quantile: float, metrics: Iterable[str] = FLOW_METRICS):
    thresholds = {}
    for metric in metrics:
        if metric in history.columns:
            values = pd.to_numeric(history[metric], errors="coerce").dropna()
            if not values.empty:
                thresholds[metric] = float(values.quantile(quantile))
    return thresholds


def add_future_outcomes(decision_points, full_timeline, thresholds, outcome_hours):
    out = decision_points.copy().reset_index(drop=True)
    timeline = full_timeline.copy()
    timeline[TS_COL] = pd.to_datetime(timeline[TS_COL], errors="coerce")
    timeline = timeline.dropna(subset=[TS_COL]).sort_values(TS_COL).drop_duplicates(TS_COL, keep="last").set_index(TS_COL)
    future_max = {m: [] for m in thresholds}
    future_bad = {m: [] for m in thresholds}
    activation_any, activation_delay, complete_windows = [], [], []
    for ts in pd.to_datetime(out[TS_COL]):
        expected = pd.date_range(ts + pd.Timedelta(hours=1), ts + pd.Timedelta(hours=outcome_hours), freq="h")
        window = timeline.reindex(expected)
        complete = expected.isin(timeline.index).all()
        complete_windows.append(bool(complete))
        active = pd.to_numeric(window.get("oncall_active"), errors="coerce").fillna(0)
        active_steps = np.flatnonzero(active.to_numpy() >= 1)
        activation_any.append(int(len(active_steps) > 0))
        activation_delay.append(float(active_steps[0] + 1) if len(active_steps) else np.nan)
        for metric, threshold in thresholds.items():
            values = pd.to_numeric(window[metric], errors="coerce") if metric in window.columns else pd.Series(dtype=float)
            maximum = float(values.max()) if values.notna().any() else np.nan
            future_max[metric].append(maximum)
            future_bad[metric].append(bool(pd.notna(maximum) and maximum >= threshold))
    for metric in thresholds:
        out[f"future_{outcome_hours}h_max_{metric}"] = future_max[metric]
        out[f"future_{outcome_hours}h_bad_{metric}"] = future_bad[metric]
    bad_cols = [c for c in out if c.startswith(f"future_{outcome_hours}h_bad_")]
    out[f"severe_flow_within_{outcome_hours}h"] = out[bad_cols].any(axis=1) if bad_cols else False
    out[f"actual_activation_within_{outcome_hours}h"] = activation_any
    out["hours_to_actual_activation"] = activation_delay
    out["complete_outcome_window"] = complete_windows
    return out


def assign_alert_band(prob):
    return pd.cut(prob, bins=[-np.inf, 0.25, 0.50, 0.70, np.inf],
                  labels=["low", "watch", "consider", "strong"], right=False).astype(str)


def build_alert_episodes(replay, config):
    alerts = replay[(replay["calibrated_probability"] >= config.alert_threshold) & replay["complete_outcome_window"]].copy()
    if alerts.empty:
        return pd.DataFrame()
    alerts = alerts.sort_values(TS_COL).reset_index(drop=True)
    gap_hours = alerts[TS_COL].diff().dt.total_seconds().div(3600)
    alerts["episode_id"] = (gap_hours.isna() | (gap_hours > 1.0)).cumsum()
    rows = []
    for episode_id, group in alerts.groupby("episode_id", sort=True):
        first, last = group.iloc[0], group.iloc[-1]
        peak = group.loc[group["calibrated_probability"].idxmax()]
        actual_activation = bool(group[f"actual_activation_within_{config.outcome_hours}h"].max())
        severe_flow = bool(group[f"severe_flow_within_{config.outcome_hours}h"].max())
        row = {
            "episode_id": int(episode_id), "alert_start": first[TS_COL], "alert_end": last[TS_COL],
            "alert_duration_hours": int(len(group)), "peak_probability": float(group["calibrated_probability"].max()),
            "peak_raw_probability": float(group["raw_probability"].max()), "peak_probability_time": peak[TS_COL],
            "actual_oncall_within_window": actual_activation, "severe_flow_within_window": severe_flow,
            "episode_class": "early_or_concordant_activation" if actual_activation else (
                "candidate_missed_opportunity" if severe_flow else "false_alarm_candidate"),
            "hours_to_actual_activation": pd.to_numeric(group["hours_to_actual_activation"], errors="coerce").min(),
        }
        for metric in FLOW_METRICS:
            if metric in first.index:
                row[f"start_{metric}"] = first[metric]
            future_col = f"future_{config.outcome_hours}h_max_{metric}"
            if future_col in group.columns:
                row[f"subsequent_max_{metric}"] = pd.to_numeric(group[future_col], errors="coerce").max()
        if "oncall_physician_id" in first.index:
            row["oncall_physician_id"] = first["oncall_physician_id"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["peak_probability", "alert_start"], ascending=[False, True])


def summarize_replay(replay, episodes, config):
    target = f"oncall_within_{config.horizon}h"
    severe = f"severe_flow_within_{config.outcome_hours}h"
    valid = replay[replay["complete_outcome_window"]].copy()
    if valid.empty:
        raise ValueError("No replay rows have a complete future outcome window.")
    alert_mask = valid["calibrated_probability"] >= config.alert_threshold
    span_days = max((valid[TS_COL].max() - valid[TS_COL].min()).total_seconds() / 86400.0, 1.0)
    weeks = span_days / 7.0
    metrics = [
        ("horizon_hours", config.horizon), ("alert_threshold", config.alert_threshold),
        ("outcome_window_hours", config.outcome_hours), ("bad_outcome_quantile", config.bad_outcome_quantile),
        ("replay_rows", len(valid)), ("replay_start", valid[TS_COL].min()), ("replay_end", valid[TS_COL].max()),
        ("raw_roc_auc", safe_metric(roc_auc_score, valid[target], valid["raw_probability"].to_numpy())),
        ("raw_average_precision", safe_metric(average_precision_score, valid[target], valid["raw_probability"].to_numpy())),
        ("raw_brier", float(brier_score_loss(valid[target], valid["raw_probability"]))),
        ("calibrated_roc_auc", safe_metric(roc_auc_score, valid[target], valid["calibrated_probability"].to_numpy())),
        ("calibrated_average_precision", safe_metric(average_precision_score, valid[target], valid["calibrated_probability"].to_numpy())),
        ("calibrated_brier", float(brier_score_loss(valid[target], valid["calibrated_probability"]))),
        ("alert_hours", int(alert_mask.sum())), ("episode_count", int(len(episodes))),
        ("episodes_per_week", float(len(episodes) / weeks)),
        ("alert_hour_ppv_actual_activation", float(valid.loc[alert_mask, target].mean()) if alert_mask.any() else np.nan),
        ("alert_hour_ppv_severe_flow", float(valid.loc[alert_mask, severe].mean()) if alert_mask.any() else np.nan),
    ]
    if not episodes.empty:
        metrics.extend([
            ("episode_ppv_actual_activation", float(episodes["actual_oncall_within_window"].mean())),
            ("episode_ppv_severe_flow", float(episodes["severe_flow_within_window"].mean())),
            ("candidate_missed_opportunity_count", int((episodes["episode_class"] == "candidate_missed_opportunity").sum())),
            ("false_alarm_candidate_count", int((episodes["episode_class"] == "false_alarm_candidate").sum())),
            ("early_or_concordant_activation_count", int((episodes["episode_class"] == "early_or_concordant_activation").sum())),
            ("median_hours_to_actual_activation", pd.to_numeric(episodes["hours_to_actual_activation"], errors="coerce").median()),
        ])
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_replay(config):
    if config.horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    full_df = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    decision_df = full_df[full_df["oncall_active"] == 0].copy()
    target = f"oncall_within_{config.horizon}h"
    decision_df = decision_df.dropna(subset=[target]).reset_index(drop=True)
    features, categorical = feature_columns(decision_df)
    for col in categorical:
        decision_df[col] = decision_df[col].fillna("Unknown").astype(str)
    fit, calibration, replay = chronological_three_way_split(decision_df)
    model, calibrator = train_independent_replay_model(fit, calibration, features, categorical, config.horizon)
    raw = model.predict_proba(replay[features])[:, 1]
    replay = replay.copy()
    replay["raw_probability"] = raw
    replay["calibrated_probability"] = calibrator.predict(raw)
    replay["alert_band"] = assign_alert_band(replay["calibrated_probability"])
    thresholds = bad_outcome_thresholds(pd.concat([fit, calibration], ignore_index=True), config.bad_outcome_quantile)
    replay = add_future_outcomes(replay, full_df, thresholds, config.outcome_hours)
    replay["would_recommend_oncall"] = replay["complete_outcome_window"] & (replay["calibrated_probability"] >= config.alert_threshold)
    episodes = build_alert_episodes(replay, config)
    summary = summarize_replay(replay, episodes, config)
    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    return replay, top, summary


def parse_args():
    p = argparse.ArgumentParser(description="Independent retrospective replay of on-call recommendation episodes.")
    p.add_argument("--horizon", type=int, choices=HORIZONS, default=DEFAULT_HORIZON)
    p.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    p.add_argument("--bad-outcome-quantile", type=float, default=DEFAULT_BAD_OUTCOME_QUANTILE)
    p.add_argument("--outcome-hours", type=int, default=DEFAULT_OUTCOME_HOURS)
    p.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    p.add_argument("--output-dir", default=".")
    return p.parse_args()


def main():
    args = parse_args()
    config = ReplayConfig(args.horizon, args.alert_threshold, args.bad_outcome_quantile, args.outcome_hours, args.top_n)
    replay, top, summary = run_replay(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output_dir / "oncall_retrospective_decision_points.csv", index=False)
    top.to_csv(output_dir / "oncall_retrospective_top_cases.csv", index=False)
    summary.to_csv(output_dir / "oncall_retrospective_performance.csv", index=False)
    print(summary.to_string(index=False))
    if not top.empty:
        display_cols = [c for c in (
            "alert_start", "peak_probability", "episode_class", "hours_to_actual_activation",
            "start_total_tbs", "subsequent_max_total_tbs", "start_stretcher_occupancy",
            "subsequent_max_stretcher_occupancy", "start_overflow", "subsequent_max_overflow") if c in top.columns]
        print("\nTop retrospective recommendation episodes:")
        print(top[display_cols].to_string(index=False))

if __name__ == "__main__":
    main()
