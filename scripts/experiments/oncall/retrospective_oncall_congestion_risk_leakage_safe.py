from __future__ import annotations

"""Leakage-safe entry point for the untreated congestion-risk replay.

The initial congestion-risk experiment intentionally reused the production feature
collector. Because the dataframe had already been decorated with future-outcome
columns, the generic numeric feature collector admitted those labels and produced
an obviously impossible perfect held-out score. This entry point explicitly removes
all future/outcome-derived columns before fitting and also excludes the currently
suspect stretcher_occupancy proxy from matched-state distance calculations.
"""

import retrospective_oncall_congestion_risk as base


def is_future_or_outcome_feature(name: str) -> bool:
    exact = {
        "hours_to_actual_activation",
        "complete_outcome_window",
    }
    prefixes = (
        "future_",
        "severe_flow_within_",
        "actual_activation_within_",
        "core_congestion_count_within_",
        "major_congestion_within_",
        "extreme_congestion_within_",
    )
    return name in exact or name.startswith(prefixes)


def leakage_safe_score_congestion_risk(full_df, config):
    labeled, fit_end, calibration_end = base.build_labeled_timeline(full_df, config)
    model_df, features, categorical = base.prepare_latent_need_model_frame(labeled)

    features = [feature for feature in features if not is_future_or_outcome_feature(feature)]
    categorical = [feature for feature in categorical if feature in features]
    if not features:
        raise ValueError("No leakage-safe congestion-risk features remain.")

    forbidden = [feature for feature in features if is_future_or_outcome_feature(feature)]
    if forbidden:
        raise AssertionError(f"Future/outcome leakage features remain: {forbidden}")

    fit, calibration, replay = base.split_untreated_training(
        model_df, fit_end, calibration_end, config
    )
    target = f"major_congestion_within_{config.outcome_hours}h"
    model, calibrator = base.train_congestion_model(
        fit, calibration, features, categorical, target
    )
    calibration_raw = model.predict_proba(calibration[features])[:, 1]
    replay_raw = model.predict_proba(replay[features])[:, 1]

    replay = replay.copy()
    replay["untreated_congestion_raw_score"] = replay_raw
    replay["untreated_congestion_probability"] = calibrator.predict(replay_raw)
    replay["oncall_need_pressure_percentile"] = base.empirical_percentile(
        replay_raw, calibration_raw
    )
    replay["oncall_need_pressure_band"] = base.assign_pressure_band(
        replay["oncall_need_pressure_percentile"]
    )
    replay["congestion_model_feature_count"] = len(features)
    return replay, fit, calibration, target


def main() -> None:
    base.score_congestion_risk = leakage_safe_score_congestion_risk
    # TTStr/53 generates impossible >100% 'occupancy' values in this dataset. Keep
    # that proxy out of matching until the source metric/capacity semantics are verified.
    base.MATCH_FEATURES = tuple(
        feature for feature in base.MATCH_FEATURES if feature != "stretcher_occupancy"
    )
    base.main()


if __name__ == "__main__":
    main()
