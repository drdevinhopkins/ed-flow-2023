from __future__ import annotations

"""Leakage-safe on-call deterioration replay adjusted for the normal diurnal ED ramp.

A global future-max delta threshold strongly labels routine morning accumulation as
'deterioration'. This variant defines meaningful worsening relative to the historical
untreated distribution for the same decision hour. The target therefore asks whether
backlog rises unusually much for *this time of day*, not merely whether it rises.
"""

import argparse
import sys
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
import retrospective_oncall_deterioration_risk as base  # noqa: E402
from retrospective_oncall_availability_adjusted import add_callability_features  # noqa: E402
from retrospective_oncall_pressure_analysis import empirical_percentile, assign_pressure_band  # noqa: E402
from retrospective_oncall_decision_replay import safe_metric  # noqa: E402

DEFAULT_HOUR_DELTA_QUANTILE = 0.80
DEFAULT_LARGE_TOTAL_QUANTILE = 0.90


def derive_hour_thresholds(
    labeled: pd.DataFrame,
    fit_end: pd.Timestamp,
    config: base.DeteriorationConfig,
    hour_quantile: float,
    large_total_quantile: float,
) -> pd.DataFrame:
    activation = f"actual_activation_within_{config.outcome_hours}h"
    fit = labeled[
        labeled["complete_outcome_window"].fillna(False)
        & (labeled[activation] == 0)
        & (pd.to_datetime(labeled[TS_COL]) <= fit_end)
    ].copy()
    if "hour" not in fit.columns:
        fit["hour"] = pd.to_datetime(fit[TS_COL]).dt.hour

    rows: list[dict[str, object]] = []
    for metric in base.DETERIORATION_METRICS:
        future_col = f"future_{config.outcome_hours}h_max_{metric}"
        delta = (
            pd.to_numeric(fit[future_col], errors="coerce")
            - pd.to_numeric(fit[metric], errors="coerce")
        )
        temp = pd.DataFrame({"hour": fit["hour"], "delta": delta}).dropna()
        if temp.empty:
            raise ValueError(f"No fit deltas available for {metric}")
        global_threshold = max(1.0, float(temp["delta"].quantile(hour_quantile)))
        global_large = max(
            global_threshold,
            float(temp["delta"].quantile(large_total_quantile)),
        )
        for hour in range(24):
            values = temp.loc[temp["hour"] == hour, "delta"]
            threshold = (
                max(1.0, float(values.quantile(hour_quantile)))
                if len(values) >= 20 else global_threshold
            )
            large = (
                max(threshold, float(values.quantile(large_total_quantile)))
                if len(values) >= 20 else global_large
            )
            rows.append({
                "metric": metric,
                "hour": hour,
                "deterioration_delta_threshold": threshold,
                "large_delta_threshold": large,
                "fit_rows_for_hour": int(len(values)),
            })
    return pd.DataFrame(rows)


def add_hour_adjusted_labels(
    labeled: pd.DataFrame,
    hour_thresholds: pd.DataFrame,
    config: base.DeteriorationConfig,
) -> pd.DataFrame:
    out = labeled.copy()
    if "hour" not in out.columns:
        out["hour"] = pd.to_datetime(out[TS_COL]).dt.hour
    flags: dict[str, pd.Series] = {}

    for metric in base.DETERIORATION_METRICS:
        lookup = hour_thresholds[hour_thresholds["metric"] == metric].set_index("hour")
        threshold = out["hour"].map(lookup["deterioration_delta_threshold"])
        large = out["hour"].map(lookup["large_delta_threshold"])
        future_col = f"future_{config.outcome_hours}h_max_{metric}"
        delta_col = f"future_{config.outcome_hours}h_delta_{metric}"
        threshold_col = f"hour_adjusted_delta_threshold_{metric}"
        flag_col = f"future_{config.outcome_hours}h_deterioration_{metric}"
        out[delta_col] = (
            pd.to_numeric(out[future_col], errors="coerce")
            - pd.to_numeric(out[metric], errors="coerce")
        )
        out[threshold_col] = threshold
        out[flag_col] = out[delta_col] >= threshold
        if metric == "total_tbs":
            out["hour_adjusted_large_delta_threshold_total_tbs"] = large
        flags[metric] = out[flag_col]

    out[f"deterioration_count_within_{config.outcome_hours}h"] = (
        pd.DataFrame(flags).fillna(False).astype(int).sum(axis=1)
    )
    total_flag = flags["total_tbs"]
    zone_flag = flags["pod_tbs"] | flags["vertical_tbs"]
    overflow_flag = flags["overflow"]
    total_delta = out[f"future_{config.outcome_hours}h_delta_total_tbs"]
    large_total = total_delta >= out["hour_adjusted_large_delta_threshold_total_tbs"]
    target = f"major_deterioration_within_{config.outcome_hours}h"
    out[target] = total_flag & zone_flag & (overflow_flag | large_total)
    return out


def leakage_safe_features(df: pd.DataFrame):
    model_df, features, categorical = base.leakage_safe_model_frame(df)
    forbidden_prefixes = (
        "hour_adjusted_delta_threshold_",
        "hour_adjusted_large_delta_threshold_",
    )
    features = [f for f in features if not f.startswith(forbidden_prefixes)]
    categorical = [f for f in categorical if f in features]
    return model_df, features, categorical


def score_hour_adjusted(
    full_df: pd.DataFrame,
    config: base.DeteriorationConfig,
    hour_quantile: float,
    large_total_quantile: float,
):
    labeled, _, fit_end, calibration_end, absolute_thresholds = base.build_labeled_timeline(
        full_df, config
    )
    hour_thresholds = derive_hour_thresholds(
        labeled, fit_end, config, hour_quantile, large_total_quantile
    )
    labeled = add_hour_adjusted_labels(labeled, hour_thresholds, config)
    model_df, features, categorical = leakage_safe_features(labeled)
    fit, calibration, replay = base.split_untreated(
        model_df, fit_end, calibration_end, config
    )
    target = f"major_deterioration_within_{config.outcome_hours}h"
    model, calibrator = base.train_congestion_model(
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
    return replay, fit, calibration, target, hour_thresholds, absolute_thresholds


def run_analysis(
    config: base.DeteriorationConfig,
    hour_quantile: float,
    large_total_quantile: float,
):
    full = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    full = add_callability_features(full, config.late_call_hour, config.min_rest_buffer_hours)
    replay, fit, calibration, target, hour_thresholds, absolute_thresholds = score_hour_adjusted(
        full, config, hour_quantile, large_total_quantile
    )
    episodes = base.cluster_episodes(replay, target, config)
    matched = base.matched_callable(replay, episodes, target, config)
    bands = base.band_summary(replay, target, config)

    # Reuse the common performance report while representing the hour-specific
    # thresholds by their fit-period medians. The full 24xmetric table is exported too.
    median_thresholds = (
        hour_thresholds.groupby("metric")["deterioration_delta_threshold"].median().to_dict()
    )
    summary = base.performance_summary(
        replay, episodes, matched, target, median_thresholds, config
    )
    summary = pd.concat([
        summary,
        pd.DataFrame([
            {"metric": "hour_adjusted_delta_quantile", "value": hour_quantile},
            {"metric": "hour_adjusted_large_total_quantile", "value": large_total_quantile},
        ]),
    ], ignore_index=True)

    high = replay[replay["deterioration_pressure_percentile"] >= config.risk_percentile_threshold].copy()
    if not high.empty:
        counts = pd.to_datetime(high[TS_COL]).dt.hour.value_counts(normalize=True)
        summary = pd.concat([
            summary,
            pd.DataFrame([
                {"metric": "high_risk_most_common_hour", "value": int(counts.index[0])},
                {"metric": "high_risk_most_common_hour_share", "value": float(counts.iloc[0])},
            ]),
        ], ignore_index=True)

    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    abs_rows = pd.DataFrame([
        {"metric": metric, "absolute_bad_threshold": threshold}
        for metric, threshold in absolute_thresholds.items()
    ])
    return replay, episodes, top, matched, bands, summary, hour_thresholds, abs_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hour-adjusted untreated deterioration replay for ED on-call need.")
    p.add_argument("--outcome-hours", type=int, default=base.DEFAULT_OUTCOME_HOURS)
    p.add_argument("--bad-outcome-quantile", type=float, default=base.DEFAULT_BAD_OUTCOME_QUANTILE)
    p.add_argument("--hour-delta-quantile", type=float, default=DEFAULT_HOUR_DELTA_QUANTILE)
    p.add_argument("--large-total-quantile", type=float, default=DEFAULT_LARGE_TOTAL_QUANTILE)
    p.add_argument("--risk-percentile-threshold", type=float, default=base.DEFAULT_RISK_PERCENTILE_THRESHOLD)
    p.add_argument("--matches", type=int, default=base.DEFAULT_MATCHES)
    p.add_argument("--top-n", type=int, default=base.DEFAULT_TOP_N)
    p.add_argument("--late-call-hour", type=int, default=base.DEFAULT_LATE_CALL_HOUR)
    p.add_argument("--min-rest-buffer-hours", type=float, default=base.DEFAULT_MIN_REST_BUFFER_HOURS)
    p.add_argument("--output-dir", default=".")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = base.DeteriorationConfig(
        outcome_hours=args.outcome_hours,
        bad_outcome_quantile=args.bad_outcome_quantile,
        positive_delta_quantile=args.hour_delta_quantile,
        risk_percentile_threshold=args.risk_percentile_threshold,
        matches=args.matches,
        top_n=args.top_n,
        late_call_hour=args.late_call_hour,
        min_rest_buffer_hours=args.min_rest_buffer_hours,
    )
    replay, episodes, top, matched, bands, summary, hour_thresholds, absolute = run_analysis(
        config, args.hour_delta_quantile, args.large_total_quantile
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    replay.to_csv(out / "oncall_deterioration_hour_adjusted_decision_points.csv", index=False)
    episodes.to_csv(out / "oncall_deterioration_hour_adjusted_episodes.csv", index=False)
    top.to_csv(out / "oncall_deterioration_hour_adjusted_top_cases.csv", index=False)
    matched.to_csv(out / "oncall_deterioration_hour_adjusted_matched_callable.csv", index=False)
    bands.to_csv(out / "oncall_deterioration_hour_adjusted_bands.csv", index=False)
    summary.to_csv(out / "oncall_deterioration_hour_adjusted_performance.csv", index=False)
    hour_thresholds.to_csv(out / "oncall_deterioration_hour_adjusted_thresholds.csv", index=False)
    absolute.to_csv(out / "oncall_deterioration_hour_adjusted_absolute_thresholds.csv", index=False)
    print(summary.to_string(index=False))
    print("\nHour-adjusted deterioration bands:")
    print(bands.to_string(index=False))
    if not top.empty:
        cols = [c for c in (
            "alert_start", "peak_risk_time", "peak_deterioration_pressure_percentile",
            "peak_untreated_deterioration_probability", "callability_status_at_peak",
            "episode_class", "current_rescue_state_at_peak", "peak_total_tbs",
            "future_delta_total_tbs", "peak_overflow", "future_delta_overflow",
        ) if c in top.columns]
        print("\nTop hour-adjusted deterioration-risk episodes:")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
