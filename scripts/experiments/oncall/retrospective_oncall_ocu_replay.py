from __future__ import annotations

"""Post-label-cutoff validation of on-call pressure using schedule ocU day labels.

The exact hourly on-call-use label table ends in May 2026.  The schedule subsequently
contains an ``ocU`` (OC busy used) marker that confirms that on-call was used on a
calendar day but does not identify the activation hour.  This analysis therefore:

1. trains the availability-adjusted activation-pressure model only inside the
   explicit hourly-label coverage window;
2. removes historical activation-lag features from the pressure model;
3. scores later operational hours without treating absent hourly labels as zeros;
4. aggregates those scores by calendar day and compares them with positive ``ocU``
   days only; and
5. audits nominal OC/B2 schedule context without inferring why a schedule exception
   occurred.

Missing ocU rows remain UNKNOWN.  They are not negative labels and are never used to
estimate specificity or PPV.
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

from forecast_oncall_probability import (  # noqa: E402
    SHIFT_DATA_URL,
    SHIFT_TYPES,
    TS_COL,
    add_horizon_targets,
    add_time_and_trend_features,
    load_dataset,
)
from retrospective_oncall_availability_adjusted import (  # noqa: E402
    add_callability_features,
    eligible_training_rows,
    prepare_latent_need_model_frame,
)
from retrospective_oncall_decision_replay import (  # noqa: E402
    chronological_three_way_split,
    train_independent_replay_model,
)
from retrospective_oncall_label_coverage import explicit_label_bounds  # noqa: E402
from retrospective_oncall_ocu_day_labels import load_ocu_day_labels  # noqa: E402
from retrospective_oncall_pressure_analysis import (  # noqa: E402
    assign_pressure_band,
    empirical_percentile,
)

DEFAULT_HORIZON = 6
DEFAULT_START_HOUR = 8
DEFAULT_END_HOUR = 22
DEFAULT_LATE_CALL_HOUR = 21
DEFAULT_MIN_REST_BUFFER_HOURS = 10.0
PRESSURE_THRESHOLDS = (90.0, 95.0, 99.0)
OCU_CODE = "ocu"
B2_CODE = "b2"


def _text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _join(values: pd.Series) -> str:
    items = sorted({str(value).strip() for value in values if str(value).strip()})
    return "|".join(items)


def train_label_safe_pressure_model(
    full_features: pd.DataFrame,
    horizon: int,
) -> tuple[object, object, np.ndarray, list[str], list[str], pd.Timestamp]:
    _, label_end = explicit_label_bounds()
    known = full_features[pd.to_datetime(full_features[TS_COL]) <= label_end].copy()
    known = add_horizon_targets(known)
    known_model, features, categorical = prepare_latent_need_model_frame(known)
    eligible = eligible_training_rows(known_model, horizon)
    fit, calibration, _ = chronological_three_way_split(eligible)
    model, calibrator = train_independent_replay_model(
        fit, calibration, features, categorical, horizon
    )
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    return model, calibrator, calibration_raw, features, categorical, label_end


def score_post_cutoff(
    full_features: pd.DataFrame,
    horizon: int,
    start_hour: int,
    end_hour: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    model, calibrator, calibration_raw, features, categorical, label_end = (
        train_label_safe_pressure_model(full_features, horizon)
    )

    model_frame, _, _ = prepare_latent_need_model_frame(full_features)
    for col in categorical:
        model_frame[col] = model_frame[col].fillna("Unknown").astype(str)

    ts = pd.to_datetime(model_frame[TS_COL], errors="coerce")
    post = model_frame[ts > label_end].copy()
    post = post[pd.to_datetime(post[TS_COL]).dt.hour.between(start_hour, end_hour)].copy()
    post = post.dropna(subset=features, how="all")
    if post.empty:
        raise ValueError("No post-cutoff operational rows available for ocU replay.")

    raw = model.predict_proba(post[features])[:, 1]
    post["raw_need_proxy_score"] = raw
    post["calibrated_activation_probability_among_callable"] = calibrator.predict(raw)
    post["oncall_need_pressure_percentile"] = empirical_percentile(raw, calibration_raw)
    post["oncall_need_pressure_band"] = assign_pressure_band(
        post["oncall_need_pressure_percentile"]
    )
    post["date"] = pd.to_datetime(post[TS_COL]).dt.date
    post["hour"] = pd.to_datetime(post[TS_COL]).dt.hour
    return post.reset_index(drop=True), label_end


def _first_crossing(group: pd.DataFrame, threshold: float):
    hits = group[group["oncall_need_pressure_percentile"] >= threshold]
    if hits.empty:
        return pd.NaT
    return pd.to_datetime(hits[TS_COL]).min()


def aggregate_daily_scores(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, group in hourly.groupby("date", sort=True):
        group = group.sort_values(TS_COL)
        peak = group.loc[group["oncall_need_pressure_percentile"].idxmax()]
        row: dict[str, object] = {
            "date": date,
            "scored_hours": int(len(group)),
            "max_need_pressure_percentile": float(group["oncall_need_pressure_percentile"].max()),
            "max_raw_need_proxy_score": float(group["raw_need_proxy_score"].max()),
            "max_calibrated_activation_probability_among_callable": float(
                group["calibrated_activation_probability_among_callable"].max()
            ),
            "peak_pressure_time": peak[TS_COL],
            "peak_callability_status": peak.get("callability_status", "unknown"),
            "peak_oncall_scheduled": bool(peak.get("oncall_scheduled", False)),
            "peak_oncall_physician_id": peak.get("oncall_physician_id", "None"),
        }
        for threshold in PRESSURE_THRESHOLDS:
            key = int(threshold)
            row[f"any_p{key}"] = bool(
                (group["oncall_need_pressure_percentile"] >= threshold).any()
            )
            row[f"first_p{key}_time"] = _first_crossing(group, threshold)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_schedule_context(dates: pd.Series, ocu_daily: pd.DataFrame) -> pd.DataFrame:
    shifts = pd.read_csv(SHIFT_DATA_URL)
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"], errors="coerce")
    shifts = shifts.dropna(subset=["shift_start"]).copy()
    shifts["date"] = shifts["shift_start"].dt.date
    shifts["shift_code_norm"] = _text(shifts["shift_short_name"]).str.casefold()
    shifts["physician_name"] = (
        _text(shifts["first_name"]) + " " + _text(shifts["last_name"])
    ).str.strip()

    nominal_oncall_codes = {
        code.casefold() for code, role in SHIFT_TYPES.items() if role == "oncall"
    }
    ocu_by_date = ocu_daily.set_index("date")

    rows: list[dict[str, object]] = []
    for date in sorted(set(dates)):
        day = shifts[shifts["date"] == date].copy()
        nominal = day[day["shift_code_norm"].isin(nominal_oncall_codes)]
        b2 = day[day["shift_code_norm"].eq(B2_CODE)]
        ocu_row = ocu_by_date.loc[date] if date in ocu_by_date.index else None
        ocu_names: set[str] = set()
        if ocu_row is not None:
            ocu_names = {
                name.strip()
                for name in str(ocu_row.get("ocu_physician_ids", "")).split("|")
                if name.strip()
            }
        nominal_names = {
            name for name in _text(nominal["physician_name"]) if name
        }

        regular = day[
            ~day["shift_code_norm"].isin(nominal_oncall_codes | {OCU_CODE})
        ]
        ocu_other_work = regular[regular["physician_name"].isin(ocu_names)] if ocu_names else regular.iloc[0:0]

        weekday = pd.Timestamp(date).dayofweek
        b2_expected_mon_thu = weekday in (0, 1, 2, 3)
        nominal_present = not nominal.empty
        positive = bool(ocu_row is not None and ocu_row.get("ocu_used_day", False))
        if not positive:
            context = "no_positive_ocu_label"
        elif not nominal_present:
            context = "ocu_positive_no_nominal_oc_row"
        elif ocu_names.intersection(nominal_names):
            context = "ocu_positive_nominal_oc_same_physician"
        else:
            context = "ocu_positive_nominal_oc_different_physician"

        rows.append(
            {
                "date": date,
                "nominal_oc_row_present": nominal_present,
                "nominal_oc_physicians": "|".join(sorted(nominal_names)),
                "ocu_physician_matches_nominal_oc": bool(ocu_names.intersection(nominal_names)),
                "ocu_physician_has_other_regular_shift_same_day": bool(len(ocu_other_work)),
                "ocu_physician_other_shift_codes": _join(ocu_other_work["shift_short_name"]),
                "b2_expected_mon_thu": b2_expected_mon_thu,
                "b2_row_present": bool(len(b2)),
                "b2_absent_on_mon_thu": bool(b2_expected_mon_thu and b2.empty),
                "schedule_context_class": context,
            }
        )
    return pd.DataFrame(rows)


def build_daily_validation(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = aggregate_daily_scores(hourly)
    ocu_daily, _ = load_ocu_day_labels()
    context = build_schedule_context(scores["date"], ocu_daily)

    keep = [
        "date",
        "ocu_rows",
        "ocu_assigned_rows",
        "ocu_physician_ids",
        "ocu_used_day",
        "ocu_day_label",
        "label_resolution",
        "activation_timestamp_known",
        "label_source",
        "label_confidence",
    ]
    merged = scores.merge(ocu_daily[keep], on="date", how="left").merge(context, on="date", how="left")
    merged["day_label_status"] = np.where(
        merged["ocu_used_day"].fillna(False),
        "ocu_positive_used_day",
        "unknown_no_positive_ocu_row",
    )
    merged["activation_timestamp_known"] = merged["activation_timestamp_known"].fillna(False)
    positives = merged[merged["day_label_status"].eq("ocu_positive_used_day")].copy()
    return merged, positives


def build_summary(
    daily: pd.DataFrame,
    positives: pd.DataFrame,
    label_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[tuple[str, object]] = [
        ("explicit_hourly_label_end", label_end),
        ("post_cutoff_first_scored_day", daily["date"].min()),
        ("post_cutoff_last_scored_day", daily["date"].max()),
        ("post_cutoff_scored_days", int(len(daily))),
        ("post_cutoff_ocu_positive_days", int(len(positives))),
        ("post_cutoff_unlabeled_days_not_treated_as_negative", int(len(daily) - len(positives))),
        ("activation_timestamp_known_for_ocu", False),
    ]
    for threshold in PRESSURE_THRESHOLDS:
        key = int(threshold)
        rows.extend(
            [
                (
                    f"ocu_positive_days_with_p{key}_same_day",
                    int(positives[f"any_p{key}"].sum()) if len(positives) else 0,
                ),
                (
                    f"ocu_positive_day_p{key}_capture_rate",
                    float(positives[f"any_p{key}"].mean()) if len(positives) else np.nan,
                ),
                (
                    f"all_scored_days_with_p{key}_descriptive_only",
                    int(daily[f"any_p{key}"].sum()),
                ),
            ]
        )
    if len(positives):
        rows.extend(
            [
                ("ocu_positive_median_daily_max_pressure_percentile", float(positives["max_need_pressure_percentile"].median())),
                ("ocu_positive_days_without_nominal_oc_row", int((~positives["nominal_oc_row_present"]).sum())),
                ("ocu_positive_days_ocu_physician_other_regular_shift", int(positives["ocu_physician_has_other_regular_shift_same_day"].sum())),
                ("ocu_positive_mon_thu_days_with_b2_absent", int(positives["b2_absent_on_mon_thu"].sum())),
            ]
        )
    rows.extend(
        [
            ("specificity_estimable_from_ocu", False),
            ("ppv_estimable_from_ocu", False),
            ("lead_time_estimable_from_ocu", False),
        ]
    )
    return pd.DataFrame(rows, columns=["metric", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate post-May on-call pressure using ocU day labels.")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--start-hour", type=int, default=DEFAULT_START_HOUR)
    parser.add_argument("--end-hour", type=int, default=DEFAULT_END_HOUR)
    parser.add_argument("--late-call-hour", type=int, default=DEFAULT_LATE_CALL_HOUR)
    parser.add_argument("--min-rest-buffer-hours", type=float, default=DEFAULT_MIN_REST_BUFFER_HOURS)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    full = add_time_and_trend_features(load_dataset())
    full = add_callability_features(full, args.late_call_hour, args.min_rest_buffer_hours)
    hourly, label_end = score_post_cutoff(full, args.horizon, args.start_hour, args.end_hour)
    daily, positives = build_daily_validation(hourly)
    summary = build_summary(daily, positives, label_end)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out / "oncall_ocu_postcutoff_hourly_scores.csv", index=False)
    daily.to_csv(out / "oncall_ocu_postcutoff_daily_validation.csv", index=False)
    positives.to_csv(out / "oncall_ocu_positive_day_cases.csv", index=False)
    summary.to_csv(out / "oncall_ocu_postcutoff_summary.csv", index=False)

    print(summary.to_string(index=False))
    print("\nocU positive post-cutoff days:")
    if positives.empty:
        print("none")
    else:
        show = [
            "date",
            "ocu_physician_ids",
            "max_need_pressure_percentile",
            "first_p90_time",
            "first_p95_time",
            "first_p99_time",
            "nominal_oc_row_present",
            "ocu_physician_matches_nominal_oc",
            "ocu_physician_has_other_regular_shift_same_day",
            "b2_absent_on_mon_thu",
            "schedule_context_class",
        ]
        print(positives[show].to_string(index=False))


if __name__ == "__main__":
    main()
