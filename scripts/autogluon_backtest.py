from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from chronos_forecast_autogluon import (
    DEVICE,
    FLOW_TARGETS,
    FLOW_URL,
    MODEL_PATH,
    PREDICTION_LENGTH,
    SHIFTS_URL,
    WEATHER_URL,
    add_holidays,
    fill_numeric,
    future_long,
    prediction_column,
    staffing_features,
    to_long,
    validate_flow_targets,
)

VALIDATION_WINDOWS = int(os.environ.get("AUTOGLUON_VALIDATION_WINDOWS", "3"))
HOLDOUT_WINDOWS = int(os.environ.get("AUTOGLUON_HOLDOUT_WINDOWS", "4"))
VAL_STEP_SIZE = int(os.environ.get("AUTOGLUON_VAL_STEP_SIZE", str(PREDICTION_LENGTH)))
BATCH_SIZE = int(os.environ.get("AUTOGLUON_BATCH_SIZE", "32"))
TIME_LIMIT = int(os.environ.get("AUTOGLUON_TIME_LIMIT", "0")) or None
HORIZON_CHECKPOINTS = (1, 4, 8, 12, 24)
QUANTILES = (0.1, 0.5, 0.9)

MODEL_DIR = Path(
    os.environ.get(
        "AUTOGLUON_BACKTEST_MODEL_PATH",
        str(Path(tempfile.gettempdir()) / "ed-flow-autogluon-backtest"),
    )
)
LEADERBOARD_PATH = Path(os.environ.get("AUTOGLUON_LEADERBOARD_PATH", "autogluon_leaderboard.csv"))
BACKTEST_ROWS_PATH = Path(os.environ.get("AUTOGLUON_BACKTEST_ROWS_PATH", "autogluon_backtest_rows.csv"))
BACKTEST_SUMMARY_PATH = Path(os.environ.get("AUTOGLUON_BACKTEST_SUMMARY_PATH", "autogluon_backtest_summary.csv"))
BACKTEST_HORIZONS_PATH = Path(os.environ.get("AUTOGLUON_BACKTEST_HORIZONS_PATH", "autogluon_backtest_horizons.csv"))


def canonicalize_flow_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Expose the six operational targets under stable names used by AutoGluon."""
    out = frame.copy()

    aliases = {
        "Total_TBS": "total_tbs",
        "POD_TBS": "pod_tbs",
        "Vertical_TBS": "vert_tbs",
        "Overflow": "overflow",
    }
    for target, source in aliases.items():
        if target not in out.columns and source in out.columns:
            out[target] = pd.to_numeric(out[source], errors="coerce")

    component_sets = {
        "Total_TBS": [
            "TRG_HALLWAY_TBS",
            "POD_GREEN_TBS",
            "POD_YELLOW_TBS",
            "POD_ORANGE_TBS",
            "RAZ_TBS",
            "AMBVERTTBS",
            "QTrack_TBS",
            "Garage_TBS",
        ],
        "POD_TBS": [
            "TRG_HALLWAY_TBS",
            "POD_GREEN_TBS",
            "POD_YELLOW_TBS",
            "POD_ORANGE_TBS",
        ],
        "Vertical_TBS": ["RAZ_TBS", "AMBVERTTBS", "QTrack_TBS", "Garage_TBS"],
        "Overflow": ["TRG_HALLWAY1", "POST_POD1"],
    }
    for target, components in component_sets.items():
        if target in out.columns:
            continue
        missing = [column for column in components if column not in out.columns]
        if not missing:
            out[target] = out[components].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    return out


def prepare_history() -> tuple[pd.DataFrame, list[str], list[str]]:
    flow = pd.read_csv(FLOW_URL)
    flow["ds"] = pd.to_datetime(flow["ds"], errors="coerce")
    flow = flow.dropna(subset=["ds"]).sort_values("ds")
    flow = canonicalize_flow_targets(flow)
    targets = validate_flow_targets(flow)

    staffing = staffing_features(pd.read_csv(SHIFTS_URL))
    staffing_columns = [column for column in staffing.columns if column != "ds"]

    weather = pd.read_csv(WEATHER_URL)
    weather["ds"] = pd.to_datetime(weather["ds"], errors="coerce")
    weather_columns = [column for column in weather.columns if column != "ds"]

    history = (
        flow[["ds", *targets]]
        .merge(staffing, on="ds", how="left")
        .merge(weather, on="ds", how="left")
    )
    history = add_holidays(history)
    covariates = [
        *staffing_columns,
        *weather_columns,
        "is_qc_holiday",
        "is_jewish_holiday",
    ]

    full_index = pd.date_range(history["ds"].min(), history["ds"].max(), freq="h", name="ds")
    history = history.set_index("ds").reindex(full_index).reset_index()
    history = add_holidays(history)
    history = fill_numeric(history, [*targets, *covariates])

    return history, targets, covariates


def split_train_holdout(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_hours = HOLDOUT_WINDOWS * PREDICTION_LENGTH
    minimum_train_hours = (VALIDATION_WINDOWS + 2) * PREDICTION_LENGTH
    if len(history) <= holdout_hours + minimum_train_hours:
        raise ValueError(
            "Not enough history for the requested internal validation and external holdout windows: "
            f"rows={len(history)}, validation_windows={VALIDATION_WINDOWS}, "
            f"holdout_windows={HOLDOUT_WINDOWS}."
        )

    split_at = len(history) - holdout_hours
    return history.iloc[:split_at].copy(), history.iloc[split_at:].copy()


def model_hyperparameters() -> dict[str, dict]:
    return {
        "Chronos2": {
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
        },
        "SeasonalNaive": {},
        "AutoETS": {},
        "Theta": {},
        "DirectTabular": {
            "model_name": "CAT",
        },
    }


def fit_predictor(train_history: pd.DataFrame, targets: list[str], covariates: list[str]) -> TimeSeriesPredictor:
    train = to_long(train_history, targets, covariates)
    predictor = TimeSeriesPredictor(
        target="target",
        known_covariates_names=covariates,
        prediction_length=PREDICTION_LENGTH,
        freq="h",
        eval_metric="WQL",
        quantile_levels=list(QUANTILES),
        path=MODEL_DIR,
        verbosity=2,
    )
    predictor.fit(
        train,
        hyperparameters=model_hyperparameters(),
        num_val_windows=VALIDATION_WINDOWS,
        val_step_size=VAL_STEP_SIZE,
        refit_every_n_windows=1,
        enable_ensemble=True,
        skip_model_selection=False,
        time_limit=TIME_LIMIT,
        verbosity=2,
    )
    return predictor


def _prediction_frame(
    predictor: TimeSeriesPredictor,
    context: TimeSeriesDataFrame,
    known: TimeSeriesDataFrame,
    model: str,
) -> pd.DataFrame:
    predictions = predictor.predict(context, known_covariates=known, model=model).reset_index()
    point_column = "mean" if "mean" in predictions.columns else prediction_column(predictions)

    output = predictions.rename(
        columns={
            "timestamp": "ds",
            "item_id": "target_name",
            point_column: "prediction",
        }
    )
    keep = ["ds", "target_name", "prediction"]
    for quantile in QUANTILES:
        source = str(quantile)
        if source in output.columns:
            destination = f"q{int(quantile * 100):02d}"
            output = output.rename(columns={source: destination})
            keep.append(destination)
    return output[keep]


def rolling_holdout_predictions(
    predictor: TimeSeriesPredictor,
    history: pd.DataFrame,
    train_history: pd.DataFrame,
    targets: list[str],
    covariates: list[str],
    models: list[str],
) -> pd.DataFrame:
    first_holdout = len(train_history)
    rows: list[pd.DataFrame] = []

    for window in range(HOLDOUT_WINDOWS):
        forecast_start_idx = first_holdout + window * PREDICTION_LENGTH
        forecast_end_idx = forecast_start_idx + PREDICTION_LENGTH
        if forecast_end_idx > len(history):
            break

        context_frame = history.iloc[:forecast_start_idx].copy()
        future_frame = history.iloc[forecast_start_idx:forecast_end_idx].copy()
        cutoff = context_frame["ds"].iloc[-1]

        context = to_long(context_frame, targets, covariates)
        known = future_long(future_frame[["ds", *covariates]], targets, covariates)
        actual = future_frame[["ds", *targets]].melt(
            id_vars="ds",
            value_vars=targets,
            var_name="target_name",
            value_name="actual",
        )

        for model in models:
            print(f"Backtest window {window + 1}/{HOLDOUT_WINDOWS}: {model} from cutoff {cutoff}")
            try:
                forecast = _prediction_frame(predictor, context, known, model)
            except Exception as exc:
                print(f"Skipping failed backtest prediction for {model}: {exc}")
                continue

            merged = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            merged["model"] = model
            merged["window"] = window + 1
            merged["cutoff"] = cutoff
            merged["horizon_hours"] = (
                (merged["ds"] - pd.Timestamp(cutoff)).dt.total_seconds() / 3600
            ).astype(int)
            merged["error"] = merged["prediction"] - merged["actual"]
            merged["absolute_error"] = merged["error"].abs()
            merged["squared_error"] = merged["error"] ** 2
            rows.append(merged)

    if not rows:
        raise RuntimeError("No rolling holdout predictions were produced")

    return pd.concat(rows, ignore_index=True)


def weighted_quantile_loss(group: pd.DataFrame) -> float:
    denominator = group["actual"].abs().sum()
    if denominator == 0:
        return float("nan")

    losses = []
    for quantile in QUANTILES:
        column = f"q{int(quantile * 100):02d}"
        if column not in group.columns:
            continue
        residual = group["actual"] - group[column]
        pinball = np.maximum(quantile * residual, (quantile - 1.0) * residual)
        losses.append(2.0 * pinball.sum() / denominator)

    return float(np.mean(losses)) if losses else float("nan")


def summarize_backtests(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["model", "target_name"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            n=("error", "size"),
        )
    )
    wql = (
        rows.groupby(["model", "target_name"], group_keys=False)
        .apply(weighted_quantile_loss)
        .rename("WQL")
        .reset_index()
    )
    summary = summary.merge(wql, on=["model", "target_name"], how="left")

    horizons = rows[rows["horizon_hours"].isin(HORIZON_CHECKPOINTS)]
    horizon_summary = (
        horizons.groupby(["model", "target_name", "horizon_hours"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            n=("error", "size"),
        )
    )
    return summary, horizon_summary


def main() -> None:
    print(f"AutoGluon device: {DEVICE}")
    print(f"Chronos-2 model: {MODEL_PATH}")
    print(f"Flow targets: {', '.join(FLOW_TARGETS)}")
    print(
        "Backtest configuration: "
        f"{VALIDATION_WINDOWS} internal validation windows, "
        f"{HOLDOUT_WINDOWS} external 24-hour holdout windows"
    )

    history, targets, covariates = prepare_history()
    train_history, holdout_history = split_train_holdout(history)
    print(
        f"Training through {train_history['ds'].iloc[-1]}; "
        f"external holdout begins {holdout_history['ds'].iloc[0]}"
    )

    predictor = fit_predictor(train_history, targets, covariates)
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv(LEADERBOARD_PATH, index=False)
    print(f"Saved model leaderboard: {LEADERBOARD_PATH}")

    models = leaderboard["model"].tolist()
    rows = rolling_holdout_predictions(
        predictor=predictor,
        history=history,
        train_history=train_history,
        targets=targets,
        covariates=covariates,
        models=models,
    )
    rows.to_csv(BACKTEST_ROWS_PATH, index=False)

    summary, horizon_summary = summarize_backtests(rows)
    summary = summary.sort_values(["target_name", "WQL", "MAE"], na_position="last")
    horizon_summary = horizon_summary.sort_values(["target_name", "horizon_hours", "MAE"])
    summary.to_csv(BACKTEST_SUMMARY_PATH, index=False)
    horizon_summary.to_csv(BACKTEST_HORIZONS_PATH, index=False)

    print(f"Saved raw backtest rows: {BACKTEST_ROWS_PATH}")
    print(f"Saved backtest summary: {BACKTEST_SUMMARY_PATH}")
    print(f"Saved horizon summary: {BACKTEST_HORIZONS_PATH}")
    print("\nExternal holdout results by target:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
