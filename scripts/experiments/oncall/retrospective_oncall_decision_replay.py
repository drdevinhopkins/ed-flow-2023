from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# Reuse the production feature construction so the replay evaluates the same signal.
from scripts.forecast_oncall_probability import (
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
REPLAY_FRACTION = 0.20
DEFAULT_HORIZON = 6
DEFAULT_ALERT_THRESHOLD = 0.70
DEFAULT_BAD_OUTCOME_QUANTILE = 0.90
DEFAULT_OUTCOME_HOURS = 6
DEFAULT_TOP_N = 20

FLOW_METRICS = (
    "total_tbs",
    "pod_tbs",
    "vertical_tbs",
    "stretcher_occupancy",
    "overflow",
    "WAITINGADM",
)


@dataclass(frozen=True)
class ReplayConfig:
    horizon: int = DEFAULT_HORIZON
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD
    bad_outcome_quantile: float = DEFAULT_BAD_OUTCOME_QUANTILE
    outcome_hours: int = DEFAULT_OUTCOME_HOURS
    top_n: int = DEFAULT_TOP_N


def chronological_three_way_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    fit_end = int(n * FIT_FRACTION)
    calibration_end = int(n * (FIT_FRACTION + CALIBRATION_FRACTION))
    if fit_end <= 0 or calibration_end <= fit_end or calibration_end >= n:
        raise ValueError("Insufficient rows for fit/calibration/replay split.")
    return (
        df.iloc[:fit_end].copy(),
        df.iloc[fit_end:calibration_end].copy(),
        df.iloc[calibration_end:].copy(),
    )


def safe_metric(metric_fn, y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(metric_fn(y_true, y_prob))


def train_independent_replay_model(
    fit: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    horizon: int,
) -> tuple[CatBoostClassifier, IsotonicRegression]:
    target = f"oncall_within_{horizon}h"
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
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(calibration_raw, calibration[target].astype(int))
    return model, calibrator


def bad_outcome_thresholds(
    history: pd.DataFrame,
    quantile: float,
    metrics: Iterable[str] = FLOW_METRICS,
) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for metric in metrics:
        if metric not in history.columns:
            continue
        values = pd.to_numeric(history[metric], errors="coerce").dropna()
        if not values.empty:
            thresholds[metric] = float(values.quantile(quantile))
    return thresholds


def add_future_outcomes(
    replay: pd.DataFrame,
    thresholds: dict[str, float],
    outcome_hours: int,
) -> pd.DataFrame:
    out = replay.copy().reset_index(drop=True)
    active = pd.to_numeric(out["oncall_active"], errors="coerce").fillna(0).astype(int)

    for metric, threshold in thresholds.items():
        values = pd.to_numeric(out[metric], errors="coerce")
        future_max = pd.concat(
            [values.shift(-step) for step in range(1, outcome_hours + 1)], axis=1
        ).max(axis=1, skipna=True)
        out[f"future_{outcome_hours}h_max_{metric}"] = future_max
        out[f"future_{outcome_hours}h_bad_{metric}"] = future_max >= threshold

    bad_cols = [c for c in out.columns if c.startswith(f"future_{outcome_hours}h_bad_")]
    out[f"severe_flow_within_{outcome_hours}h"] = out[bad_cols].any(axis=1) if bad_cols else False

    future_active = pd.concat(
        [active.shift(-step) for step in range(1, outcome_hours + 1)], axis=1
    )
    out[f"actual_activation_within_{outcome_hours}h"] = future_active.max(axis=1, skipna=True).fillna(0).astype(int)

    activation_delay: list[float] = []
    for idx in range(len(out)):
        delay = np.nan
        for step in range(1, outcome_hours + 1):
            j = idx + step
            if j < len(out) and active.iloc[j] == 1:
                delay = float(step)
                break
        activation_delay.append(delay)
    out["hours_to_actual_activation"] = activation_delay
    return out


def assign_alert_band(prob: pd.Series) -> pd.Series:
    return pd.cut(
        prob,
        bins=[-np.inf, 0.25, 0.50, 0.70, np.inf],
        labels=["low", "watch", "consider", "strong"],
        right=False,
    ).astype(str)


def build_alert_episodes(
    replay: pd.DataFrame,
    config: ReplayConfig,
) -> pd.DataFrame:
    alerts = replay[replay["calibrated_probability"] >= config.alert_threshold].copy()
    if alerts.empty:
        return pd.DataFrame()

    alerts = alerts.sort_values(TS_COL).reset_index(drop=True)
    gap_hours = alerts[TS_COL].diff().dt.total_seconds().div(3600)
    alerts["episode_id"] = (gap_hours.isna() | (gap_hours > 1.0)).cumsum()

    rows: list[dict[str, object]] = []
    for episode_id, group in alerts.groupby("episode_id", sort=True):
        first = group.iloc[0]
        last = group.iloc[-1]
        peak_idx = group["calibrated_probability"].idxmax()
        peak = alerts.loc[peak_idx]
        actual_activation = bool(group[f"actual_activation_within_{config.outcome_hours}h"].max())
        severe_flow = bool(group[f"severe_flow_within_{config.outcome_hours}h"].max())

        row: dict[str, object] = {
            "episode_id": int(episode_id),
            "alert_start": first[TS_COL],
            "alert_end": last[TS_COL],
            "alert_duration_hours": int(len(group)),
            "peak_probability": float(group["calibrated_probability"].max()),
            "peak_raw_probability": float(group["raw_probability"].max()),
            "peak_probability_time": peak[TS_COL],
            "actual_oncall_within_window": actual_activation,
            "severe_flow_within_window": severe_flow,
            "episode_class": (
                "early_or_concordant_activation" if actual_activation
                else "candidate_missed_opportunity" if severe_flow
                else "false_alarm_candidate"
            ),
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
        ["peak_probability", "alert_start"], ascending=[False, True]
    )


def summarize_replay(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    config: ReplayConfig,
) -> pd.DataFrame:
    target = f"oncall_within_{config.horizon}h"
    severe = f"severe_flow_within_{config.outcome_hours}h"
    alert_mask = replay["calibrated_probability"] >= config.alert_threshold

    span_days = max(
        (replay[TS_COL].max() - replay[TS_COL].min()).total_seconds() / 86400.0,
        1.0,
    )
    weeks = span_days / 7.0

    metrics: list[tuple[str, object]] = [
        ("horizon_hours", config.horizon),
        ("alert_threshold", config.alert_threshold),
        ("outcome_window_hours", config.outcome_hours),
        ("bad_outcome_quantile", config.bad_outcome_quantile),
        ("replay_rows", len(replay)),
        ("replay_start", replay[TS_COL].min()),
        ("replay_end", replay[TS_COL].max()),
        ("raw_roc_auc", safe_metric(roc_auc_score, replay[target], replay["raw_probability"].to_numpy())),
        ("raw_average_precision", safe_metric(average_precision_score, replay[target], replay["raw_probability"].to_numpy())),
        ("raw_brier", float(brier_score_loss(replay[target], replay["raw_probability"]))),
        ("calibrated_roc_auc", safe_metric(roc_auc_score, replay[target], replay["calibrated_probability"].to_numpy())),
        ("calibrated_average_precision", safe_metric(average_precision_score, replay[target], replay["calibrated_probability"].to_numpy())),
        ("calibrated_brier", float(brier_score_loss(replay[target], replay["calibrated_probability"]))),
        ("alert_hours", int(alert_mask.sum())),
        ("episode_count", int(len(episodes))),
        ("episodes_per_week", float(len(episodes) / weeks)),
        ("alert_hour_ppv_actual_activation", float(replay.loc[alert_mask, target].mean()) if alert_mask.any() else np.nan),
        ("alert_hour_ppv_severe_flow", float(replay.loc[alert_mask, severe].mean()) if alert_mask.any() else np.nan),
    ]

    if not episodes.empty:
        metrics.extend(
            [
                ("episode_ppv_actual_activation", float(episodes["actual_oncall_within_window"].mean())),
                ("episode_ppv_severe_flow", float(episodes["severe_flow_within_window"].mean())),
                ("candidate_missed_opportunity_count", int((episodes["episode_class"] == "candidate_missed_opportunity").sum())),
                ("false_alarm_candidate_count", int((episodes["episode_class"] == "false_alarm_candidate").sum())),
                ("early_or_concordant_activation_count", int((episodes["episode_class"] == "early_or_concordant_activation").sum())),
                ("median_hours_to_actual_activation", pd.to_numeric(episodes["hours_to_actual_activation"], errors="coerce").median()),
            ]
        )
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_replay(config: ReplayConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")

    df = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    df = df[df["oncall_active"] == 0].copy()
    target = f"oncall_within_{config.horizon}h"
    df = df.dropna(subset=[target]).reset_index(drop=True)

    features, categorical = feature_columns(df)
    for col in categorical:
        df[col] = df[col].fillna("Unknown").astype(str)

    fit, calibration, replay = chronological_three_way_split(df)
    model, calibrator = train_independent_replay_model(
        fit, calibration, features, categorical, config.horizon
    )

    raw = model.predict_proba(replay[features])[:, 1]
    replay = replay.copy()
    replay["raw_probability"] = raw
    replay["calibrated_probability"] = calibrator.predict(raw)
    replay["alert_band"] = assign_alert_band(replay["calibrated_probability"])

    thresholds = bad_outcome_thresholds(
        pd.concat([fit, calibration], ignore_index=True), config.bad_outcome_quantile
    )
    replay = add_future_outcomes(replay, thresholds, config.outcome_hours)
    replay["would_recommend_oncall"] = replay["calibrated_probability"] >= config.alert_threshold

    episodes = build_alert_episodes(replay, config)
    summary = summarize_replay(replay, episodes, config)

    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    return replay, top, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Independent retrospective replay of on-call recommendation episodes."
    )
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=DEFAULT_HORIZON)
    parser.add_argument("--alert-threshold", type=float, default=DEFAULT_ALERT_THRESHOLD)
    parser.add_argument("--bad-outcome-quantile", type=float, default=DEFAULT_BAD_OUTCOME_QUANTILE)
    parser.add_argument("--outcome-hours", type=int, default=DEFAULT_OUTCOME_HOURS)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ReplayConfig(
        horizon=args.horizon,
        alert_threshold=args.alert_threshold,
        bad_outcome_quantile=args.bad_outcome_quantile,
        outcome_hours=args.outcome_hours,
        top_n=args.top_n,
    )
    replay, top, summary = run_replay(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    replay.to_csv(output_dir / "oncall_retrospective_decision_points.csv", index=False)
    top.to_csv(output_dir / "oncall_retrospective_top_cases.csv", index=False)
    summary.to_csv(output_dir / "oncall_retrospective_performance.csv", index=False)

    print(summary.to_string(index=False))
    if not top.empty:
        display_cols = [
            c for c in (
                "alert_start",
                "peak_probability",
                "episode_class",
                "hours_to_actual_activation",
                "start_total_tbs",
                "subsequent_max_total_tbs",
                "start_stretcher_occupancy",
                "subsequent_max_stretcher_occupancy",
                "start_overflow",
                "subsequent_max_overflow",
            )
            if c in top.columns
        ]
        print("\nTop retrospective recommendation episodes:")
        print(top[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
