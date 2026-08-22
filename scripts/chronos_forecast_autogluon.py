from __future__ import annotations

import os
import tempfile
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from dotenv import load_dotenv

load_dotenv()

FLOW_URL = "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
SHIFTS_URL = "https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1"
WEATHER_URL = "https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1"
PREDICTION_LENGTH = 24
DIRECT_FORECAST_PATH = Path(os.environ.get("DIRECT_FORECAST_PATH", "chronos_forecast.csv"))
AUTOGLUON_FORECAST_PATH = Path(os.environ.get("AUTOGLUON_FORECAST_PATH", "chronos_forecast_autogluon.csv"))
COMPARISON_PATH = Path(os.environ.get("COMPARISON_PATH", "chronos_autogluon_comparison.csv"))
MODEL_PATH = os.environ.get("AUTOGLUON_CHRONOS2_MODEL", "amazon/chronos-2")
MODEL_DIR = Path(os.environ.get("AUTOGLUON_CHRONOS2_PATH", str(Path(tempfile.gettempdir()) / "ed-flow-autogluon-chronos2")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Operational outcomes that matter for the ED flow forecast. Keep this list
# intentionally small so model evaluation remains interpretable and efficient.
FLOW_TARGETS = [
    "Total_TBS",
    "POD_TBS",
    "Vertical_TBS",
    "TTStr",
    "Overflow",
    "WAITINGADM",
]

SHIFT_TYPES = {
    "W1":"flow", "X1":"pod", "X3":"pod", "X4":"vertical", "X2":"vertical",
    "WOC1":"oncall", "WOC2":"oncall", "WOC3":"oncall", "X5":"pod", "W3":"overlap",
    "Y1":"pod", "Y3":"pod", "Y4":"vertical", "Y2":"vertical", "Y5":"pod",
    "Z1":"night", "Z2":"night", "D1":"pod", "R1":"pod", "P1":"vertical",
    "D2":"vertical", "OC1":"oncall", "OC2":"oncall", "V1":"flow", "A1":"pod",
    "G1":"vertical", "E1":"pod", "R2":"pod", "A2":"pod", "P2":"vertical",
    "E2":"vertical", "N1":"night", "N2":"night", "L2":"overlap", "L4":"overlap",
    "H1":"teaching", "B1":"vertical", "L1":"overlap", "W5":"overlap", "L6":"overlap", "B2":"vertical",
}


def validate_flow_targets(frame: pd.DataFrame) -> list[str]:
    """Return the configured targets, failing loudly if the input schema changed."""
    missing = [target for target in FLOW_TARGETS if target not in frame.columns]
    if missing:
        available = ", ".join(sorted(frame.columns))
        raise ValueError(
            "Missing required flow target column(s): "
            f"{', '.join(missing)}. Available columns: {available}"
        )

    non_numeric = [target for target in FLOW_TARGETS if not pd.api.types.is_numeric_dtype(frame[target])]
    if non_numeric:
        print(f"Coercing configured flow targets to numeric: {', '.join(non_numeric)}")
        for target in non_numeric:
            frame[target] = pd.to_numeric(frame[target], errors="coerce")

    empty = [target for target in FLOW_TARGETS if frame[target].notna().sum() == 0]
    if empty:
        raise ValueError(f"Configured flow target(s) contain no numeric observations: {', '.join(empty)}")

    return list(FLOW_TARGETS)


def add_holidays(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    dates = pd.to_datetime(out["ds"], errors="coerce").dt.date
    years = dates.dropna().map(lambda value: value.year)
    if years.empty:
        raise ValueError("No valid timestamps available for holiday features")
    year_range = range(int(years.min()), int(years.max()) + 1)
    qc = holidays.Canada(subdiv="QC", years=year_range, observed=True)
    il = holidays.Israel(years=year_range, observed=True)
    out["is_qc_holiday"] = dates.map(lambda value: int(value in qc) if pd.notna(value) else 0)
    out["is_jewish_holiday"] = dates.map(lambda value: int(value in il) if pd.notna(value) else 0)
    return out


def staffing_features(shifts: pd.DataFrame) -> pd.DataFrame:
    shifts = shifts.copy()
    shifts["shift_start"] = pd.to_datetime(shifts["shift_start"]).dt.round("h")
    shifts["shift_end"] = pd.to_datetime(shifts["shift_end"]).dt.round("h")
    shifts["shift_type"] = shifts["shift_short_name"].map(SHIFT_TYPES).fillna("unknown")
    rows = []
    for row in shifts.itertuples(index=False):
        for timestamp in pd.date_range(row.shift_start, row.shift_end, freq="h", inclusive="left"):
            rows.append({"ds": timestamp, "shift_type": row.shift_type})
    if not rows:
        raise ValueError("No staffing hours were generated")
    return (pd.DataFrame(rows).groupby(["ds", "shift_type"]).size()
            .unstack(fill_value=0).add_prefix("staffing_").add_suffix("_count").reset_index())


def fill_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").ffill().bfill().fillna(0.0)
    return out


def to_long(frame: pd.DataFrame, targets: list[str], covariates: list[str]) -> TimeSeriesDataFrame:
    long = frame[["ds", *covariates, *targets]].melt(
        id_vars=["ds", *covariates], value_vars=targets, var_name="item_id", value_name="target"
    )
    long["target"] = pd.to_numeric(long["target"], errors="coerce").ffill().bfill().fillna(0.0)
    return TimeSeriesDataFrame.from_data_frame(long, id_column="item_id", timestamp_column="ds")


def future_long(frame: pd.DataFrame, targets: list[str], covariates: list[str]) -> TimeSeriesDataFrame:
    future = frame.assign(_key=1).merge(pd.DataFrame({"item_id": targets, "_key": 1}), on="_key").drop(columns="_key")
    return TimeSeriesDataFrame.from_data_frame(future[["item_id", "ds", *covariates]], id_column="item_id", timestamp_column="ds")


def prediction_column(frame: pd.DataFrame) -> str:
    for column in ("0.5", "mean"):
        if column in frame:
            return column
    numeric = [column for column in frame if column not in {"item_id", "timestamp"} and pd.api.types.is_numeric_dtype(frame[column])]
    if not numeric:
        raise ValueError(f"No numeric prediction column found: {list(frame.columns)}")
    return numeric[0]


def compare(autogluon: pd.DataFrame) -> None:
    if not DIRECT_FORECAST_PATH.exists():
        print(f"Direct forecast not found; skipped comparison: {DIRECT_FORECAST_PATH}")
        return
    direct = pd.read_csv(DIRECT_FORECAST_PATH)
    direct["ds"] = pd.to_datetime(direct["ds"], errors="coerce")
    direct_column = os.environ.get("DIRECT_FORECAST_COLUMN", "forecast_all_vars_with_future")
    if direct_column not in direct:
        raise ValueError(f"Missing direct forecast column {direct_column!r}")
    direct = direct[direct["target_name"].isin(FLOW_TARGETS)]
    direct = direct[["ds", "target_name", direct_column]].rename(columns={direct_column: "direct_chronos_forecast"})
    result = autogluon.merge(direct, on=["ds", "target_name"], how="inner")
    result["difference"] = result["autogluon_forecast"] - result["direct_chronos_forecast"]
    result["absolute_difference"] = result["difference"].abs()
    result["percent_difference"] = result["difference"] / result["direct_chronos_forecast"].abs().replace(0, np.nan) * 100
    result.to_csv(COMPARISON_PATH, index=False)
    if not result.empty:
        print(f"Compared {len(result)} rows across {result['target_name'].nunique()} targets")
        print(f"Mean absolute model difference: {result['difference'].abs().mean():.6f}")
        print(f"RMSE between model outputs: {np.sqrt((result['difference'] ** 2).mean()):.6f}")
    print(f"Saved comparison: {COMPARISON_PATH}")


def main() -> None:
    print(f"Using AutoGluon Chronos2 device: {DEVICE}")
    print(f"AutoGluon Chronos2 model: {MODEL_PATH}")

    flow = pd.read_csv(FLOW_URL)
    flow["ds"] = pd.to_datetime(flow["ds"], errors="coerce")
    flow = flow.dropna(subset=["ds"]).sort_values("ds")
    targets = validate_flow_targets(flow)
    print(f"Forecasting {len(targets)} operational targets: {', '.join(targets)}")

    staffing = staffing_features(pd.read_csv(SHIFTS_URL))
    weather = pd.read_csv(WEATHER_URL)
    weather["ds"] = pd.to_datetime(weather["ds"], errors="coerce")
    weather_columns = [column for column in weather if column != "ds"]

    history = flow[["ds", *targets]].merge(staffing, on="ds", how="left").merge(weather, on="ds", how="left")
    history = add_holidays(history)
    staffing_columns = [column for column in staffing if column != "ds"]
    covariates = [*staffing_columns, *weather_columns, "is_qc_holiday", "is_jewish_holiday"]
    history = fill_numeric(history, [*targets, *covariates])

    last = history["ds"].max()
    index = pd.date_range(history["ds"].min(), last, freq="h", name="ds")
    history = history.set_index("ds").reindex(index).reset_index()
    history = fill_numeric(add_holidays(history), [*targets, *covariates])

    future_index = pd.date_range(last + pd.Timedelta(hours=1), periods=PREDICTION_LENGTH, freq="h", name="ds")
    future = pd.DataFrame(index=future_index).reset_index().merge(staffing, on="ds", how="left").merge(weather, on="ds", how="left")
    future = add_holidays(future)
    future = fill_numeric(future, [*staffing_columns, *weather_columns])
    future[["is_qc_holiday", "is_jewish_holiday"]] = future[["is_qc_holiday", "is_jewish_holiday"]].fillna(0)
    if len(future) != PREDICTION_LENGTH:
        raise ValueError("Could not construct a complete 24-hour future covariate frame")

    train = to_long(history, targets, covariates)
    known = future_long(future, targets, covariates)
    predictor = TimeSeriesPredictor(
        target="target",
        known_covariates_names=covariates,
        prediction_length=PREDICTION_LENGTH,
        freq="h",
        eval_metric="WQL",
        quantile_levels=[0.5],
        path=MODEL_DIR,
        verbosity=2,
    )
    predictor.fit(
        train,
        hyperparameters={
            "Chronos2": {
                "model_path": MODEL_PATH,
                "device": DEVICE,
                "batch_size": int(os.environ.get("AUTOGLUON_BATCH_SIZE", "32")),
            }
        },
        skip_model_selection=True,
        enable_ensemble=False,
        verbosity=2,
    )
    predictions = predictor.predict(train, known_covariates=known).reset_index()
    column = prediction_column(predictions)
    output = predictions.rename(
        columns={"timestamp": "ds", "item_id": "target_name", column: "autogluon_forecast"}
    )[["ds", "target_name", "autogluon_forecast"]]
    output = output[output["target_name"].isin(FLOW_TARGETS)]
    output.to_csv(AUTOGLUON_FORECAST_PATH, index=False)
    print(f"Saved AutoGluon forecast: {AUTOGLUON_FORECAST_PATH}")
    compare(output)


if __name__ == "__main__":
    main()
