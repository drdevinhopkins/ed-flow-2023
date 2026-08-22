from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterable

import holidays
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models" / "oncall_probability"
ID_COL = "id"
TS_COL = "ds"
SERIES_ID = "jgh"
HORIZONS = (4, 6, 8)
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42

HOURLY_DATA_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
SHIFT_DATA_URL = (
    "https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/"
    "all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1"
)
WEATHER_DATA_URL = (
    "https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/"
    "weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1"
)
ONCALL_LABELS_PATH = REPO_ROOT / "hourly_oncall_used_for_busy_since_2022.csv"

SHIFT_TYPES = {
    "W1": "flow", "X1": "pod", "X3": "pod", "X4": "vertical", "X2": "vertical",
    "WOC1": "oncall", "WOC2": "oncall", "WOC3": "oncall", "X5": "pod",
    "W3": "overlap", "Y1": "pod", "Y3": "pod", "Y4": "vertical",
    "Y2": "vertical", "Y5": "pod", "Z1": "night", "Z2": "night", "D1": "pod",
    "R1": "pod", "P1": "vertical", "D2": "vertical", "OC1": "oncall",
    "OC2": "oncall", "V1": "flow", "A1": "pod", "G1": "vertical", "E1": "pod",
    "R2": "pod", "A2": "pod", "P2": "vertical", "E2": "vertical",
    "N1": "night", "N2": "night", "L2": "overlap", "L4": "overlap",
    "H1": "teaching", "B1": "vertical", "L1": "overlap", "W5": "overlap",
    "L6": "overlap", "B2": "vertical",
}
ROLE_TYPES = ("flow", "pod", "vertical", "overlap", "teaching", "night", "oncall")


def add_holiday_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dates = pd.to_datetime(out[TS_COL], errors="coerce").dt.date
    valid_dates = [d for d in dates if pd.notna(d)]
    if not valid_dates:
        raise ValueError("No valid timestamps available for holiday features.")
    years = range(min(d.year for d in valid_dates), max(d.year for d in valid_dates) + 1)
    qc = holidays.Canada(subdiv="QC", years=years, observed=True)
    jewish = holidays.Israel(years=years, observed=True)
    out["is_qc_holiday"] = ["yes" if d in qc else "no" for d in dates]
    out["is_jewish_holiday"] = ["yes" if d in jewish else "no" for d in dates]
    return out


def derive_flow_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def derive_sum(name: str, components: Iterable[str]) -> None:
        components = list(components)
        if name not in out.columns and all(c in out.columns for c in components):
            out[name] = out[components].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    derive_sum(
        "total_tbs",
        ["TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS",
         "RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"],
    )
    derive_sum(
        "pod_tbs",
        ["TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS"],
    )
    derive_sum("vertical_tbs", ["RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"])
    derive_sum("overflow", ["POST_POD1", "TRG_HALLWAY1"])
    return out


def build_staffing_features(all_shifts_df: pd.DataFrame) -> pd.DataFrame:
    shifts = all_shifts_df.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"], errors="coerce").dt.round("h")
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"], errors="coerce").dt.round("h")
    shifts["shift_type"] = shifts["shift_short_name"].map(SHIFT_TYPES)
    shifts["physician_id"] = (
        shifts["first_name"].fillna("").astype(str).str.strip()
        + shifts["last_name"].fillna("").astype(str).str.strip()
    )
    shifts = shifts.dropna(subset=["shift_start", "shift_end", "shift_type"])
    shifts = shifts[shifts["physician_id"] != ""]

    rows: list[dict[str, object]] = []
    for row in shifts.itertuples(index=False):
        for hour in pd.date_range(row.shift_start, row.shift_end, freq="h", inclusive="left"):
            rows.append({TS_COL: hour, "physician_id": row.physician_id, "shift_type": row.shift_type})

    expanded = pd.DataFrame(rows)
    if expanded.empty:
        raise ValueError("No staffing hours could be generated.")

    physician_matrix = (
        expanded.pivot_table(index=TS_COL, columns="physician_id", values="shift_type", aggfunc="first")
        .fillna("NotWorking")
        .add_prefix("physician__")
    )
    role_counts = (
        expanded.groupby([TS_COL, "shift_type"]).size().unstack(fill_value=0)
        .reindex(columns=ROLE_TYPES, fill_value=0).add_prefix("n_")
    )
    role_counts["n_total_scheduled"] = role_counts.sum(axis=1)
    oncall_ids = (
        expanded[expanded["shift_type"] == "oncall"]
        .groupby(TS_COL)["physician_id"]
        .agg(lambda x: "|".join(sorted(set(x))))
        .rename("oncall_physician_id")
    )
    staffing = physician_matrix.join(role_counts, how="outer").join(oncall_ids, how="left")
    staffing["oncall_physician_id"] = staffing["oncall_physician_id"].fillna("None")
    return staffing.reset_index()


def load_dataset() -> pd.DataFrame:
    hourly = pd.read_csv(HOURLY_DATA_URL)
    hourly[TS_COL] = pd.to_datetime(hourly[TS_COL], errors="coerce").dt.floor("h")
    hourly = hourly.dropna(subset=[TS_COL]).sort_values(TS_COL)
    hourly = derive_flow_metrics(hourly)

    shifts = pd.read_csv(SHIFT_DATA_URL)
    staffing = build_staffing_features(shifts)

    weather = pd.read_csv(WEATHER_DATA_URL)
    weather[TS_COL] = pd.to_datetime(weather[TS_COL], errors="coerce").dt.floor("h")
    weather = weather.dropna(subset=[TS_COL]).sort_values(TS_COL)

    labels = pd.read_csv(ONCALL_LABELS_PATH)
    labels[TS_COL] = pd.to_datetime(labels[TS_COL], errors="coerce").dt.floor("h")
    labels = labels.rename(columns={"oncall-used-for-busy": "oncall_active"})
    labels = labels[[TS_COL, "oncall_active"]].dropna(subset=[TS_COL])

    df = hourly.merge(staffing, on=TS_COL, how="inner").merge(weather, on=TS_COL, how="inner")
    df = df.merge(labels, on=TS_COL, how="left")

    # Assumption inherited from the source label design: no label row means no activation.
    # If the capture process was incomplete for any historical interval, restrict the dataset
    # to the verified-complete period before interpreting the probabilities operationally.
    df["oncall_active"] = pd.to_numeric(df["oncall_active"], errors="coerce").fillna(0).clip(0, 1)
    df[ID_COL] = SERIES_ID
    df = add_holiday_flags(df)
    return df.sort_values(TS_COL).drop_duplicates(TS_COL, keep="last").reset_index(drop=True)


def add_time_and_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[TS_COL])
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    preferred = [
        "total_tbs", "pod_tbs", "vertical_tbs", "overflow", "stretcher_occupancy",
        "WAITINGADM", "Inflow_Total", "TRG_HALLWAY_TBS", "RESUS",
    ]
    numeric_candidates = [c for c in preferred if c in out.columns]
    for col in numeric_candidates:
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = values
        out[f"{col}_lag1"] = values.shift(1)
        out[f"{col}_lag2"] = values.shift(2)
        out[f"{col}_lag4"] = values.shift(4)
        out[f"{col}_delta1"] = values - values.shift(1)
        out[f"{col}_delta4"] = values - values.shift(4)
        out[f"{col}_mean4"] = values.shift(1).rolling(4, min_periods=1).mean()

    out["oncall_active_lag1"] = out["oncall_active"].shift(1)
    out["oncall_activations_prior_24h"] = out["oncall_active"].shift(1).rolling(24, min_periods=1).sum()
    return out


def add_horizon_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    active = out["oncall_active"].astype(int)
    for horizon in HORIZONS:
        future_cols = [active.shift(-step) for step in range(1, horizon + 1)]
        target_frame = pd.concat(future_cols, axis=1)
        out[f"oncall_within_{horizon}h"] = target_frame.max(axis=1, skipna=False)
    return out


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {ID_COL, TS_COL, "oncall_active"}
    excluded.update(f"oncall_within_{h}h" for h in HORIZONS)

    categorical = [
        c for c in df.columns
        if c.startswith("physician__")
        or c in {"oncall_physician_id", "is_qc_holiday", "is_jewish_holiday"}
    ]
    numeric = [
        c for c in df.columns
        if c not in excluded and c not in categorical and pd.api.types.is_numeric_dtype(df[c])
    ]
    features = numeric + categorical
    return features, categorical


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_at = int(len(df) * (1 - VALIDATION_FRACTION))
    if split_at <= 0 or split_at >= len(df):
        raise ValueError("Insufficient data for chronological train/validation split.")
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()


def safe_metric(metric_fn, y_true: pd.Series, y_prob: np.ndarray) -> float | None:
    if pd.Series(y_true).nunique() < 2:
        return None
    return float(metric_fn(y_true, y_prob))


def train_horizon(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    horizon: int,
) -> tuple[CatBoostClassifier, IsotonicRegression, dict[str, object]]:
    target = f"oncall_within_{horizon}h"
    cat_indices = [features.index(c) for c in categorical]

    model = CatBoostClassifier(
        iterations=700,
        depth=7,
        learning_rate=0.04,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        train[features],
        train[target].astype(int),
        cat_features=cat_indices,
        eval_set=(validation[features], validation[target].astype(int)),
        early_stopping_rounds=75,
        verbose=False,
    )

    raw_prob = model.predict_proba(validation[features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_prob, validation[target].astype(int))
    calibrated_prob = calibrator.predict(raw_prob)

    metrics = {
        "horizon_hours": horizon,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "train_event_rate": float(train[target].mean()),
        "validation_event_rate": float(validation[target].mean()),
        "roc_auc_raw": safe_metric(roc_auc_score, validation[target], raw_prob),
        "average_precision_raw": safe_metric(average_precision_score, validation[target], raw_prob),
        "brier_raw": float(brier_score_loss(validation[target], raw_prob)),
        "brier_calibrated": float(brier_score_loss(validation[target], calibrated_prob)),
    }
    return model, calibrator, metrics


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = add_horizon_targets(add_time_and_trend_features(load_dataset()))

    # A decision-support probability is only meaningful before activation. Rows where on-call
    # is already active are excluded from model training and from the live prediction row.
    df = df[df["oncall_active"] == 0].copy()
    df = df.dropna(subset=[f"oncall_within_{h}h" for h in HORIZONS])

    features, categorical = feature_columns(df)
    if not features:
        raise ValueError("No usable model features were found.")

    for col in categorical:
        df[col] = df[col].fillna("Unknown").astype(str)

    train, validation = chronological_split(df)
    current = df.iloc[[-1]].copy()
    probabilities: list[dict[str, object]] = []
    validation_metrics: list[dict[str, object]] = []

    for horizon in HORIZONS:
        model, calibrator, metrics = train_horizon(
            train, validation, features, categorical, horizon
        )
        raw = float(model.predict_proba(current[features])[:, 1][0])
        calibrated = float(calibrator.predict([raw])[0])

        model.save_model(str(MODEL_DIR / f"oncall_within_{horizon}h.cbm"))
        with open(MODEL_DIR / f"oncall_within_{horizon}h_calibrator.pkl", "wb") as f:
            pickle.dump(calibrator, f)

        probabilities.append(
            {
                TS_COL: current[TS_COL].iloc[0],
                "horizon_hours": horizon,
                "raw_probability": raw,
                "calibrated_probability": calibrated,
                "oncall_physician_id": current["oncall_physician_id"].iloc[0]
                if "oncall_physician_id" in current.columns else "Unknown",
            }
        )
        validation_metrics.append(metrics)

    pd.DataFrame(probabilities).to_csv("oncall_need_probability.csv", index=False)
    pd.DataFrame(validation_metrics).to_csv("oncall_need_probability_validation.csv", index=False)

    metadata = {
        "features": features,
        "categorical_features": categorical,
        "horizons_hours": list(HORIZONS),
        "validation_fraction": VALIDATION_FRACTION,
        "note": (
            "Probabilities are calibrated on a chronological holdout. They predict historical "
            "on-call activation behavior, not a causal requirement for additional staffing."
        ),
    }
    with open(MODEL_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved oncall_need_probability.csv")
    print("Saved oncall_need_probability_validation.csv")
    print(f"Saved models and calibration artifacts under {MODEL_DIR}")


if __name__ == "__main__":
    main()
