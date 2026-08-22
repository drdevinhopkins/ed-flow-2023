from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import dropbox
import holidays
import requests
from dotenv import load_dotenv
import pandas as pd
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from utils import upload

load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[1]
ID_COL = "id"
TS_COL = "ds"
SERIES_ID = "jgh"
PREDICTION_LENGTH = 8
SCENARIO_DURATIONS = (4, 6, 8)
QUANTILES = (0.1, 0.5, 0.9)

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


def load_pipeline() -> Chronos2Pipeline:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading amazon/chronos-2 on {device}")
    return BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map=device)


def add_holiday_flags(
    df: pd.DataFrame,
    ts_col: str = TS_COL,
    local_tz: str = "America/Montreal",
    observed: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce")

    if getattr(out[ts_col].dt, "tz", None) is not None:
        dates = out[ts_col].dt.tz_convert(local_tz).dt.date
    else:
        dates = out[ts_col].dt.date

    valid_dates = [d for d in dates if pd.notna(d)]
    if not valid_dates:
        raise ValueError("No valid datetimes found when building holiday covariates.")

    years = range(min(d.year for d in valid_dates), max(d.year for d in valid_dates) + 1)
    qc_holidays = holidays.Canada(subdiv="QC", years=years, observed=observed)
    jewish_holidays = holidays.Israel(years=years, observed=observed)

    out["is_qc_holiday"] = ["yes" if d in qc_holidays else "no" for d in dates]
    out["is_jewish_holiday"] = ["yes" if d in jewish_holidays else "no" for d in dates]
    return out


def derive_flow_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create canonical flow metrics where source columns allow it."""
    out = df.copy()

    total_tbs_components = [
        "TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS",
        "RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS",
    ]
    pod_tbs_components = [
        "TRG_HALLWAY_TBS", "POD_GREEN_TBS", "POD_YELLOW_TBS", "POD_ORANGE_TBS",
    ]
    vertical_tbs_components = ["RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"]

    def derive_sum(name: str, components: Iterable[str]) -> None:
        components = list(components)
        if name not in out.columns and all(c in out.columns for c in components):
            out[name] = out[components].sum(axis=1)

    derive_sum("total_tbs", total_tbs_components)
    derive_sum("pod_tbs", pod_tbs_components)
    derive_sum("vertical_tbs", vertical_tbs_components)
    derive_sum("overflow", ["POST_POD1", "TRG_HALLWAY1"])

    aliases = {
        "total_tbs": ("total_tbs", "Total_TBS", "TOTAL_TBS"),
        "pod_tbs": ("pod_tbs", "POD_TBS", "Pod_TBS"),
        "vertical_tbs": ("vertical_tbs", "VERTICAL_TBS", "VERT_TBS", "vert_tbs"),
        "stretcher_occupancy": (
            "stretcher_occupancy",
            "STRETCHER_OCCUPANCY",
            "Stretcher_Occupancy",
            "stretcher_occupancy_pct",
            "STRETCHER_OCCUPANCY_PCT",
        ),
        "overflow": ("overflow", "OVERFLOW", "Overflow"),
    }

    targets: list[str] = []
    for canonical, candidates in aliases.items():
        if canonical in out.columns:
            targets.append(canonical)
            continue
        source = next((c for c in candidates if c in out.columns), None)
        if source:
            out[canonical] = pd.to_numeric(out[source], errors="coerce")
            targets.append(canonical)
        else:
            print(f"Warning: could not resolve flow metric '{canonical}'; it will be skipped.")

    if not targets:
        raise ValueError("None of the requested flow metrics could be resolved from the hourly dataset.")

    return out, targets


def build_staffing_features(all_shifts_df: pd.DataFrame) -> pd.DataFrame:
    """Preserve physician identity and add structural staffing covariates."""
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

    expanded_rows: list[dict[str, object]] = []
    for row in shifts.itertuples(index=False):
        for hour in pd.date_range(row.shift_start, row.shift_end, freq="h", inclusive="left"):
            expanded_rows.append(
                {
                    TS_COL: hour,
                    "physician_id": row.physician_id,
                    "shift_type": row.shift_type,
                }
            )

    expanded = pd.DataFrame(expanded_rows)
    if expanded.empty:
        raise ValueError("No valid physician shift-hours could be built.")

    # Identity is intentionally retained: each physician gets a categorical role/NotWorking feature.
    physician_matrix = (
        expanded.pivot_table(
            index=TS_COL,
            columns="physician_id",
            values="shift_type",
            aggfunc="first",
        )
        .fillna("NotWorking")
        .add_prefix("physician__")
    )

    role_counts = (
        expanded.groupby([TS_COL, "shift_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ROLE_TYPES, fill_value=0)
        .add_prefix("n_")
    )
    role_counts["n_total_scheduled"] = role_counts.sum(axis=1)

    # Explicitly expose who is scheduled on-call so Chronos can learn physician-specific
    # differences in activation and downstream impact without relying only on sparse columns.
    oncall_ids = (
        expanded[expanded["shift_type"] == "oncall"]
        .groupby(TS_COL)["physician_id"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .rename("oncall_physician_id")
    )

    staffing = physician_matrix.join(role_counts, how="outer").join(oncall_ids, how="left")
    staffing["oncall_physician_id"] = staffing["oncall_physician_id"].fillna("None")
    return staffing.reset_index().sort_values(TS_COL)


def regularize_history(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    out = df.sort_values(TS_COL).copy()
    full_idx = pd.date_range(out[TS_COL].min(), out[TS_COL].max(), freq="h")
    out = out.set_index(TS_COL).reindex(full_idx)
    out.index.name = TS_COL
    out[ID_COL] = SERIES_ID

    for target in targets:
        out[target] = (
            pd.to_numeric(out[target], errors="coerce")
            .interpolate(limit_direction="both")
        )

    return out.reset_index()


def require_exact_future_hours(
    frame: pd.DataFrame,
    future_hours: pd.DatetimeIndex,
    name: str,
) -> pd.DataFrame:
    indexed = frame.drop_duplicates(TS_COL, keep="last").set_index(TS_COL)
    future = indexed.reindex(future_hours)
    missing = future.index[future.isna().all(axis=1)]
    if len(missing):
        rendered = ", ".join(ts.isoformat() for ts in missing)
        raise ValueError(f"{name} is missing required future hours: {rendered}")
    future.index.name = TS_COL
    return future.reset_index()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_csv(HOURLY_DATA_URL)
    hourly[TS_COL] = pd.to_datetime(hourly[TS_COL], errors="coerce").dt.floor("h")
    hourly = hourly.dropna(subset=[TS_COL]).sort_values(TS_COL)
    hourly[ID_COL] = SERIES_ID

    shifts = pd.read_csv(SHIFT_DATA_URL)
    weather = pd.read_csv(WEATHER_DATA_URL)
    weather[TS_COL] = pd.to_datetime(weather[TS_COL], errors="coerce").dt.floor("h")
    weather = weather.dropna(subset=[TS_COL]).sort_values(TS_COL)

    return hourly, shifts, weather


def add_historical_oncall_activation(hourly: pd.DataFrame) -> pd.DataFrame:
    labels = pd.read_csv(ONCALL_LABELS_PATH)
    labels[TS_COL] = pd.to_datetime(labels[TS_COL], errors="coerce").dt.floor("h")
    labels = labels.rename(columns={"oncall-used-for-busy": "oncall_active"})
    labels = labels[[TS_COL, "oncall_active"]].dropna(subset=[TS_COL])

    out = hourly.merge(labels, on=TS_COL, how="left")

    # This assumes a missing row in the label file means "on-call was not activated".
    # If historical label capture was incomplete, restrict the training window before using
    # this script; do not silently train on an incompletely observed period.
    out["oncall_active"] = out["oncall_active"].fillna(0).astype(float)
    return out


def make_future_base(
    history: pd.DataFrame,
    staffing: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    cutoff = history[TS_COL].max()
    future_hours = pd.date_range(
        cutoff + pd.Timedelta(hours=1),
        periods=PREDICTION_LENGTH,
        freq="h",
    )

    future_staffing = require_exact_future_hours(staffing, future_hours, "staffing schedule")
    future_weather = require_exact_future_hours(weather, future_hours, "weather forecast")

    future = future_staffing.merge(future_weather, on=TS_COL, how="inner", validate="one_to_one")
    future[ID_COL] = SERIES_ID
    future = add_holiday_flags(future)
    return future


def forecast_scenario(
    pipeline: Chronos2Pipeline,
    history: pd.DataFrame,
    future_base: pd.DataFrame,
    targets: list[str],
    active_hours: int,
) -> pd.DataFrame:
    future = future_base.copy()
    future["oncall_active"] = 0.0
    if active_hours:
        future.loc[future.index[:active_hours], "oncall_active"] = 1.0

    # Chronos future_df should contain only columns available historically (plus id/timestamp),
    # not future target values.
    future_columns = [
        c for c in future.columns
        if c in history.columns and c not in targets
    ]
    future = future[future_columns]

    forecast = pipeline.predict_df(
        history,
        prediction_length=PREDICTION_LENGTH,
        future_df=future,
        id_column=ID_COL,
        timestamp_column=TS_COL,
        target=targets,
        quantile_levels=list(QUANTILES),
    )

    scenario = "no_oncall" if active_hours == 0 else f"oncall_{active_hours}h"
    forecast = forecast.copy()
    forecast["scenario"] = scenario
    forecast["oncall_active_hours"] = active_hours
    return forecast


def build_comparison(all_forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed hourly deltas and 4/6/8-hour endpoint summaries."""
    value_columns = [
        c for c in ("predictions", "0.1", "0.5", "0.9")
        if c in all_forecasts.columns
    ]
    if "predictions" not in value_columns:
        raise ValueError("Chronos forecast output did not contain a 'predictions' column.")

    detail = all_forecasts[
        [TS_COL, "target_name", "scenario", "oncall_active_hours", *value_columns]
    ].copy()

    baseline = detail[detail["scenario"] == "no_oncall"][
        [TS_COL, "target_name", "predictions"]
    ].rename(columns={"predictions": "no_oncall_prediction"})

    detail = detail.merge(baseline, on=[TS_COL, "target_name"], how="left")
    detail["estimated_improvement"] = (
        detail["no_oncall_prediction"] - detail["predictions"]
    )
    detail["hours_ahead"] = (
        detail.groupby(["scenario", "target_name"]).cumcount() + 1
    )

    summary = detail[
        detail["hours_ahead"].isin(SCENARIO_DURATIONS)
        & detail["scenario"].isin([f"oncall_{h}h" for h in SCENARIO_DURATIONS])
    ].copy()
    summary = summary[
        [
            "scenario",
            "oncall_active_hours",
            "hours_ahead",
            "target_name",
            "no_oncall_prediction",
            "predictions",
            "estimated_improvement",
        ]
    ].rename(columns={"predictions": "with_oncall_prediction"})

    return detail, summary


def upload_outputs(output_paths: Iterable[str]) -> None:
    dropbox_app_key = os.environ.get("DROPBOX_APP_KEY")
    dropbox_app_secret = os.environ.get("DROPBOX_APP_SECRET")
    dropbox_refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")

    token_url = "https://api.dropboxapi.com/oauth2/token"
    params = {
        "grant_type": "refresh_token",
        "refresh_token": dropbox_refresh_token,
        "client_id": dropbox_app_key,
        "client_secret": dropbox_app_secret,
    }
    response = requests.post(token_url, data=params, timeout=30)
    response.raise_for_status()
    dropbox_access_token = response.json()["access_token"]
    dbx = dropbox.Dropbox(dropbox_access_token)

    for output_path in output_paths:
        upload(dbx, output_path, "", "", Path(output_path).name, overwrite=True)


def main() -> None:
    hourly, shifts, weather = load_inputs()
    hourly, targets = derive_flow_metrics(hourly)
    hourly = add_historical_oncall_activation(hourly)

    staffing = build_staffing_features(shifts)

    # Inner joins intentionally define the last timestamp where all historical covariates exist.
    history = hourly.merge(staffing, on=TS_COL, how="inner")
    history = add_holiday_flags(history).merge(weather, on=TS_COL, how="inner")
    history = (
        history.sort_values(TS_COL)
        .drop_duplicates([ID_COL, TS_COL], keep="last")
    )
    history = regularize_history(history, targets)

    # Keep synthetic hourly rows so Chronos can infer a strict frequency. Fill each
    # missing covariate according to its semantics instead of dropping timestamps.
    for column in history.columns:
        if column in targets or column in (TS_COL, ID_COL):
            continue
        missing = history[column].isna()
        if not missing.any():
            continue
        if column.startswith("physician__"):
            history.loc[missing, column] = "NotWorking"
        elif column == "oncall_physician_id":
            history.loc[missing, column] = "None"
        elif column in {"is_qc_holiday", "is_jewish_holiday"}:
            continue
        elif column == "oncall_active" or column.startswith("n_"):
            history.loc[missing, column] = 0.0
        elif pd.api.types.is_numeric_dtype(history[column]):
            history[column] = pd.to_numeric(history[column], errors="coerce").interpolate(
                limit_direction="both"
            )
        else:
            history[column] = history[column].ffill().bfill()

    history = add_holiday_flags(history)

    cutoff = history[TS_COL].max()
    print(f"Historical cutoff: {cutoff}")
    print(f"Targets: {', '.join(targets)}")

    future_base = make_future_base(history, staffing, weather)

    pipeline = load_pipeline()
    forecasts = [
        forecast_scenario(pipeline, history, future_base, targets, active_hours=0),
        *[
            forecast_scenario(pipeline, history, future_base, targets, active_hours=h)
            for h in SCENARIO_DURATIONS
        ],
    ]

    all_forecasts = pd.concat(forecasts, ignore_index=True)
    detail, summary = build_comparison(all_forecasts)

    detail.to_csv("oncall_impact_forecast.csv", index=False)
    summary.to_csv("oncall_impact_summary.csv", index=False)
    upload_outputs(("oncall_impact_forecast.csv", "oncall_impact_summary.csv"))

    print("Saved oncall_impact_forecast.csv")
    print("Saved oncall_impact_summary.csv")
    print(
        "Important: these are model-based counterfactual scenarios, not causal treatment "
        "effects. Historical on-call activation is confounded by ED severity."
    )


if __name__ == "__main__":
    main()
