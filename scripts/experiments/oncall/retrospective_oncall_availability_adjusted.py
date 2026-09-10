from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    add_future_outcomes,
    bad_outcome_thresholds,
    chronological_three_way_split,
    train_independent_replay_model,
)
from retrospective_oncall_pressure_analysis import (  # noqa: E402
    CORE_CONGESTION_METRICS,
    FLOW_METRICS,
    MATCH_FEATURES,
    add_congestion_composite,
    assign_pressure_band,
    circular_hour_distance,
    empirical_percentile,
)

DEFAULT_HORIZON = 6
DEFAULT_PRESSURE_THRESHOLD = 99.0
DEFAULT_BAD_OUTCOME_QUANTILE = 0.90
DEFAULT_OUTCOME_HOURS = 6
DEFAULT_MAJOR_CORE_COUNT = 3
DEFAULT_MATCHES = 5
DEFAULT_TOP_N = 30
DEFAULT_LATE_CALL_HOUR = 21
DEFAULT_MIN_REST_BUFFER_HOURS = 10.0
NEXT_SHIFT_LOOKAHEAD_HOURS = 36
SENSITIVITY_LATE_HOURS = (20, 21, 22)
SENSITIVITY_REST_BUFFERS = (8.0, 10.0, 12.0)


@dataclass(frozen=True)
class AvailabilityConfig:
    horizon: int = DEFAULT_HORIZON
    pressure_threshold: float = DEFAULT_PRESSURE_THRESHOLD
    bad_outcome_quantile: float = DEFAULT_BAD_OUTCOME_QUANTILE
    outcome_hours: int = DEFAULT_OUTCOME_HOURS
    major_core_count: int = DEFAULT_MAJOR_CORE_COUNT
    matches: int = DEFAULT_MATCHES
    top_n: int = DEFAULT_TOP_N
    late_call_hour: int = DEFAULT_LATE_CALL_HOUR
    min_rest_buffer_hours: float = DEFAULT_MIN_REST_BUFFER_HOURS


def normalize_oncall_id(series: pd.Series) -> pd.Series:
    text = series.fillna("None").astype(str).str.strip()
    return text.replace({"": "None", "nan": "None", "NaN": "None", "Unknown": "None"})


def add_scheduled_availability(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["oncall_physician_id"] = normalize_oncall_id(out["oncall_physician_id"])
    out["oncall_scheduled"] = out["oncall_physician_id"].ne("None")
    return out


def hours_to_next_non_oncall_shift(df: pd.DataFrame) -> pd.Series:
    """Proxy next-work interval from the hourly physician-role matrix.

    For the physician assigned on-call at a decision hour, find the next future hour
    (within 36 h) where that physician is working a non-on-call role. This does not
    claim contractual rest requirements; it is used only as an operational
    sensitivity proxy for 'may have to work in the morning'.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    physician_cols = {c.removeprefix("physician__"): c for c in df if c.startswith("physician__")}
    if not physician_cols:
        return out

    role_frame = df[list(physician_cols.values())].fillna("NotWorking").astype(str)
    ids = normalize_oncall_id(df["oncall_physician_id"])
    n = len(df)
    for i in range(n):
        physician_id = ids.iat[i]
        column = physician_cols.get(physician_id)
        if not column:
            continue
        upper = min(n, i + NEXT_SHIFT_LOOKAHEAD_HOURS + 1)
        future_roles = role_frame[column].iloc[i + 1 : upper]
        mask = ~future_roles.isin(["NotWorking", "oncall"])
        if mask.any():
            first_pos = int(np.flatnonzero(mask.to_numpy())[0]) + 1
            out.iat[i] = float(first_pos)
    return out


def classify_callability(
    df: pd.DataFrame,
    late_call_hour: int,
    min_rest_buffer_hours: float,
) -> pd.Series:
    scheduled = df["oncall_scheduled"].fillna(False).astype(bool)
    hour = pd.to_datetime(df[TS_COL]).dt.hour
    next_shift = pd.to_numeric(df["hours_to_next_non_oncall_shift"], errors="coerce")
    late = hour >= late_call_hour
    next_shift_constrained = next_shift.notna() & (next_shift <= min_rest_buffer_hours)

    status = pd.Series("callable_proxy", index=df.index, dtype=object)
    status.loc[~scheduled] = "unavailable_no_oncall_scheduled"
    status.loc[scheduled & late] = "constrained_late_evening"
    status.loc[scheduled & next_shift_constrained] = "constrained_next_shift"
    status.loc[scheduled & late & next_shift_constrained] = "constrained_late_and_next_shift"
    return status


def add_callability_features(
    df: pd.DataFrame,
    late_call_hour: int,
    min_rest_buffer_hours: float,
) -> pd.DataFrame:
    out = add_scheduled_availability(df)
    out["hours_to_next_non_oncall_shift"] = hours_to_next_non_oncall_shift(out)
    out["callability_status"] = classify_callability(out, late_call_hour, min_rest_buffer_hours)
    out["oncall_callable_proxy"] = out["callability_status"].eq("callable_proxy")
    return out


def prepare_latent_need_model_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Hide on-call availability/identity from the pressure model.

    Other working-physician identities remain available because actual ED staffing can
    plausibly affect need. A physician's 'oncall' role is replaced by NotWorking so the
    model cannot infer who is assigned on-call from the identity matrix.
    """
    model_df = df.copy()
    physician_cols = [c for c in model_df if c.startswith("physician__")]
    for col in physician_cols:
        values = model_df[col].fillna("NotWorking").astype(str)
        model_df[col] = values.mask(values.eq("oncall"), "NotWorking")

    if "n_total_scheduled" in model_df.columns:
        total = pd.to_numeric(model_df["n_total_scheduled"], errors="coerce")
        n_oncall = pd.to_numeric(model_df.get("n_oncall", 0), errors="coerce").fillna(0)
        model_df["n_total_working_excl_oncall"] = total - n_oncall

    features, categorical = feature_columns(model_df)
    excluded = {
        "oncall_physician_id",
        "n_oncall",
        "n_total_scheduled",
        "oncall_active_lag1",
        "oncall_activations_prior_24h",
        "oncall_scheduled",
        "oncall_callable_proxy",
        "hours_to_next_non_oncall_shift",
    }
    features = [f for f in features if f not in excluded]
    categorical = [c for c in categorical if c in features and c != "oncall_physician_id"]
    for col in categorical:
        model_df[col] = model_df[col].fillna("Unknown").astype(str)
    return model_df, features, categorical


def eligible_training_rows(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target = f"oncall_within_{horizon}h"
    return df[
        (df["oncall_active"] == 0)
        & df["oncall_callable_proxy"]
        & df[target].notna()
    ].copy().reset_index(drop=True)


def score_replay(
    full_df: pd.DataFrame,
    config: AvailabilityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df, features, categorical = prepare_latent_need_model_frame(full_df)
    eligible = eligible_training_rows(model_df, config.horizon)
    fit, calibration, eligible_replay = chronological_three_way_split(eligible)
    model, calibrator = train_independent_replay_model(
        fit, calibration, features, categorical, config.horizon
    )
    calibration_raw = model.predict_proba(calibration[features])[:, 1]

    replay_start = pd.to_datetime(eligible_replay[TS_COL]).min()
    replay = model_df[
        (model_df["oncall_active"] == 0)
        & (pd.to_datetime(model_df[TS_COL]) >= replay_start)
        & model_df[f"oncall_within_{config.horizon}h"].notna()
    ].copy().reset_index(drop=True)
    raw = model.predict_proba(replay[features])[:, 1]
    replay["raw_need_proxy_score"] = raw
    replay["calibrated_activation_probability_among_callable"] = calibrator.predict(raw)
    replay["oncall_need_pressure_percentile"] = empirical_percentile(raw, calibration_raw)
    replay["oncall_need_pressure_band"] = assign_pressure_band(replay["oncall_need_pressure_percentile"])
    return replay, fit, calibration


def classify_episode(first: pd.Series, activation: bool, major: bool) -> str:
    if activation:
        return "early_or_concordant_activation"
    status = str(first.get("callability_status", "unknown"))
    if status == "unavailable_no_oncall_scheduled":
        return "high_need_unavailable"
    if status.startswith("constrained_"):
        return "high_need_constrained"
    if major:
        return "candidate_missed_opportunity_callable"
    return "callable_high_pressure_no_major_congestion"


def cluster_episodes(replay: pd.DataFrame, config: AvailabilityConfig) -> pd.DataFrame:
    alerts = replay[
        replay["complete_outcome_window"]
        & (replay["oncall_need_pressure_percentile"] >= config.pressure_threshold)
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
        group = group.sort_values(TS_COL)
        peak = group.sort_values(
            ["oncall_need_pressure_percentile", "raw_need_proxy_score"], ascending=False
        ).iloc[0]
        first = peak
        activation = bool(group[activation_col].max())
        major = bool(group[major_col].max())
        extreme = bool(group[extreme_col].max())
        row: dict[str, object] = {
            "episode_id": int(episode_id),
            "alert_start": group.iloc[0][TS_COL],
            "alert_end": group.iloc[-1][TS_COL],
            "alert_duration_hours": int(len(group)),
            "peak_pressure_time": peak[TS_COL],
            "peak_need_pressure_percentile": float(group["oncall_need_pressure_percentile"].max()),
            "peak_need_pressure_band": peak["oncall_need_pressure_band"],
            "peak_raw_need_proxy_score": float(peak["raw_need_proxy_score"]),
            "peak_calibrated_activation_probability_among_callable": float(
                peak["calibrated_activation_probability_among_callable"]
            ),
            "callability_status_at_peak": peak["callability_status"],
            "oncall_scheduled_at_peak": bool(peak["oncall_scheduled"]),
            "oncall_physician_id_at_peak": peak["oncall_physician_id"],
            "hours_to_next_non_oncall_shift_at_peak": peak["hours_to_next_non_oncall_shift"],
            "actual_oncall_within_window": activation,
            "major_congestion_within_window": major,
            "extreme_congestion_within_window": extreme,
            "episode_class": classify_episode(first, activation, major),
            "hours_to_actual_activation": pd.to_numeric(
                group["hours_to_actual_activation"], errors="coerce"
            ).min(),
            "any_callable_hour_in_episode": bool(group["oncall_callable_proxy"].any()),
        }
        for metric in FLOW_METRICS:
            if metric in peak.index:
                row[f"peak_{metric}"] = peak[metric]
            future_col = f"future_{config.outcome_hours}h_max_{metric}"
            if future_col in group.columns:
                row[f"subsequent_max_{metric}"] = pd.to_numeric(
                    group[future_col], errors="coerce"
                ).max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["peak_need_pressure_percentile", "peak_raw_need_proxy_score", "alert_start"],
        ascending=[False, False, True],
    )


def availability_summary(replay: pd.DataFrame, config: AvailabilityConfig) -> pd.DataFrame:
    valid = replay[replay["complete_outcome_window"]].copy()
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for threshold in (90.0, 95.0, 99.0):
        high = valid[valid["oncall_need_pressure_percentile"] >= threshold]
        for status, group in high.groupby("callability_status", dropna=False):
            rows.append({
                "pressure_threshold": threshold,
                "callability_status": status,
                "decision_hours": int(len(group)),
                "actual_activation_rate": float(group[activation_col].mean()) if len(group) else np.nan,
                "major_congestion_rate": float(group[major_col].mean()) if len(group) else np.nan,
            })
    return pd.DataFrame(rows)


def matched_callable_noactivation(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    config: AvailabilityConfig,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    valid = replay[replay["complete_outcome_window"]].copy()
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    extreme_col = f"extreme_congestion_within_{config.outcome_hours}h"
    available_features = [f for f in MATCH_FEATURES if f in valid.columns and f != "n_total_scheduled"]
    if "n_total_working_excl_oncall" in valid.columns:
        available_features.append("n_total_working_excl_oncall")
    numeric = valid[available_features].apply(pd.to_numeric, errors="coerce")
    medians = numeric.median()
    scales = numeric.std(ddof=0).replace(0, np.nan).fillna(1.0)
    standardized = (numeric.fillna(medians) - medians) / scales

    rows: list[dict[str, object]] = []
    cases = episodes[episodes["episode_class"] == "candidate_missed_opportunity_callable"]
    for episode in cases.itertuples(index=False):
        case_time = pd.Timestamp(episode.peak_pressure_time)
        indices = valid.index[valid[TS_COL] == case_time].tolist()
        if not indices:
            continue
        case_idx = indices[0]
        case = valid.loc[case_idx]
        controls = valid[
            (valid[activation_col] == 0)
            & valid["oncall_callable_proxy"]
            & (valid["oncall_need_pressure_percentile"] < config.pressure_threshold)
        ].copy()
        controls = controls[
            (pd.to_datetime(controls[TS_COL]) - case_time).abs() > pd.Timedelta(hours=24)
        ]
        if "is_weekend" in controls.columns:
            controls = controls[controls["is_weekend"] == case["is_weekend"]]
        if "hour" in controls.columns:
            distance = circular_hour_distance(controls["hour"], int(case["hour"]))
            narrowed = controls[distance <= 2]
            if len(narrowed) >= config.matches:
                controls = narrowed
        if controls.empty:
            continue
        case_vector = standardized.loc[case_idx, available_features].to_numpy(dtype=float)
        matrix = standardized.loc[controls.index, available_features].to_numpy(dtype=float)
        distances = np.sqrt(((matrix - case_vector) ** 2).mean(axis=1))
        order = np.argsort(distances)[: config.matches]
        matched = controls.iloc[order]
        row: dict[str, object] = {
            "episode_id": int(episode.episode_id),
            "case_time": case_time,
            "case_pressure_percentile": float(case["oncall_need_pressure_percentile"]),
            "matched_controls": int(len(matched)),
            "mean_match_distance": float(np.mean(distances[order])),
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
    return pd.DataFrame(rows)


def callability_sensitivity(replay: pd.DataFrame, config: AvailabilityConfig) -> pd.DataFrame:
    valid = replay[
        replay["complete_outcome_window"]
        & (replay["oncall_need_pressure_percentile"] >= config.pressure_threshold)
    ].copy()
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    rows: list[dict[str, object]] = []
    for late_hour in SENSITIVITY_LATE_HOURS:
        for rest_buffer in SENSITIVITY_REST_BUFFERS:
            status = classify_callability(valid, late_hour, rest_buffer)
            callable_mask = status.eq("callable_proxy")
            callable = valid[callable_mask]
            rows.append({
                "late_call_hour": late_hour,
                "min_rest_buffer_hours": rest_buffer,
                "high_pressure_hours": int(len(valid)),
                "callable_high_pressure_hours": int(callable_mask.sum()),
                "callable_share": float(callable_mask.mean()) if len(valid) else np.nan,
                "callable_actual_activation_rate": float(callable[activation_col].mean()) if len(callable) else np.nan,
                "callable_major_congestion_rate": float(callable[major_col].mean()) if len(callable) else np.nan,
            })
    return pd.DataFrame(rows)


def performance_summary(
    replay: pd.DataFrame,
    episodes: pd.DataFrame,
    config: AvailabilityConfig,
) -> pd.DataFrame:
    valid = replay[replay["complete_outcome_window"]].copy()
    high = valid[valid["oncall_need_pressure_percentile"] >= config.pressure_threshold]
    major_col = f"major_congestion_within_{config.outcome_hours}h"
    activation_col = f"actual_activation_within_{config.outcome_hours}h"
    metrics: list[tuple[str, object]] = [
        ("horizon_hours", config.horizon),
        ("pressure_percentile_threshold", config.pressure_threshold),
        ("late_call_hour_proxy", config.late_call_hour),
        ("min_rest_buffer_hours_proxy", config.min_rest_buffer_hours),
        ("replay_rows_all_decision_hours", int(len(valid))),
        ("replay_rows_oncall_scheduled", int(valid["oncall_scheduled"].sum())),
        ("replay_rows_callable_proxy", int(valid["oncall_callable_proxy"].sum())),
        ("high_pressure_decision_hours", int(len(high))),
        ("high_pressure_major_congestion_rate", float(high[major_col].mean()) if len(high) else np.nan),
        ("high_pressure_actual_activation_rate", float(high[activation_col].mean()) if len(high) else np.nan),
        ("episode_count", int(len(episodes))),
    ]
    if not episodes.empty:
        counts = episodes["episode_class"].value_counts()
        for label in (
            "early_or_concordant_activation",
            "candidate_missed_opportunity_callable",
            "high_need_unavailable",
            "high_need_constrained",
            "callable_high_pressure_no_major_congestion",
        ):
            metrics.append((f"episode_count_{label}", int(counts.get(label, 0))))
    return pd.DataFrame(metrics, columns=["metric", "value"])


def run_analysis(config: AvailabilityConfig):
    full_df = add_horizon_targets(add_time_and_trend_features(load_dataset()))
    full_df = add_callability_features(
        full_df, config.late_call_hour, config.min_rest_buffer_hours
    )
    replay, fit, calibration = score_replay(full_df, config)
    thresholds = bad_outcome_thresholds(
        pd.concat([fit, calibration], ignore_index=True), config.bad_outcome_quantile
    )
    replay = add_future_outcomes(replay, full_df, thresholds, config.outcome_hours)
    replay = add_congestion_composite(replay, config.outcome_hours, config.major_core_count)
    replay["high_need_pressure_flag"] = (
        replay["complete_outcome_window"]
        & (replay["oncall_need_pressure_percentile"] >= config.pressure_threshold)
    )
    episodes = cluster_episodes(replay, config)
    top = episodes.head(config.top_n).copy() if not episodes.empty else episodes.copy()
    availability = availability_summary(replay, config)
    matched = matched_callable_noactivation(replay, episodes, config)
    sensitivity = callability_sensitivity(replay, config)
    summary = performance_summary(replay, episodes, config)
    return replay, episodes, top, availability, matched, sensitivity, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Availability-adjusted retrospective on-call need-pressure replay."
    )
    parser.add_argument("--horizon", type=int, choices=HORIZONS, default=DEFAULT_HORIZON)
    parser.add_argument("--pressure-percentile-threshold", type=float, default=DEFAULT_PRESSURE_THRESHOLD)
    parser.add_argument("--bad-outcome-quantile", type=float, default=DEFAULT_BAD_OUTCOME_QUANTILE)
    parser.add_argument("--outcome-hours", type=int, default=DEFAULT_OUTCOME_HOURS)
    parser.add_argument("--major-core-count", type=int, default=DEFAULT_MAJOR_CORE_COUNT)
    parser.add_argument("--matches", type=int, default=DEFAULT_MATCHES)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--late-call-hour", type=int, default=DEFAULT_LATE_CALL_HOUR)
    parser.add_argument("--min-rest-buffer-hours", type=float, default=DEFAULT_MIN_REST_BUFFER_HOURS)
    parser.add_argument("--output-dir", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AvailabilityConfig(
        horizon=args.horizon,
        pressure_threshold=args.pressure_percentile_threshold,
        bad_outcome_quantile=args.bad_outcome_quantile,
        outcome_hours=args.outcome_hours,
        major_core_count=args.major_core_count,
        matches=args.matches,
        top_n=args.top_n,
        late_call_hour=args.late_call_hour,
        min_rest_buffer_hours=args.min_rest_buffer_hours,
    )
    replay, episodes, top, availability, matched, sensitivity, summary = run_analysis(config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    replay.to_csv(output_dir / "oncall_availability_adjusted_decision_points.csv", index=False)
    episodes.to_csv(output_dir / "oncall_availability_adjusted_episodes.csv", index=False)
    top.to_csv(output_dir / "oncall_availability_adjusted_top_cases.csv", index=False)
    availability.to_csv(output_dir / "oncall_availability_adjusted_summary.csv", index=False)
    matched.to_csv(output_dir / "oncall_availability_adjusted_matched_callable.csv", index=False)
    sensitivity.to_csv(output_dir / "oncall_callability_sensitivity.csv", index=False)
    summary.to_csv(output_dir / "oncall_availability_adjusted_performance.csv", index=False)

    print(summary.to_string(index=False))
    print("\nHigh-pressure hours by callability:")
    print(availability[availability["pressure_threshold"] == config.pressure_threshold].to_string(index=False))
    print("\nCallability sensitivity:")
    print(sensitivity.to_string(index=False))
    if not top.empty:
        cols = [c for c in (
            "alert_start", "peak_pressure_time", "peak_need_pressure_percentile",
            "callability_status_at_peak", "episode_class", "oncall_physician_id_at_peak",
            "hours_to_next_non_oncall_shift_at_peak", "actual_oncall_within_window",
            "major_congestion_within_window", "peak_total_tbs", "subsequent_max_total_tbs",
            "peak_overflow", "subsequent_max_overflow", "peak_WAITINGADM",
            "subsequent_max_WAITINGADM",
        ) if c in top.columns]
        print("\nTop availability-adjusted episodes:")
        print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
