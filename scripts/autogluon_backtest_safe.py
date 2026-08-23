from __future__ import annotations

import os

import numpy as np
import pandas as pd

import autogluon_backtest as base
from chronos_forecast_autogluon import (
    FLOW_TARGETS,
    FLOW_URL,
    SHIFTS_URL,
    WEATHER_URL,
    add_holidays,
    future_long,
    staffing_features,
    to_long,
    validate_flow_targets,
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


INCLUDE_WEATHER = env_flag("AUTOGLUON_INCLUDE_WEATHER", False)


def prepare_history() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """Build a retrospective dataset without using future information to fill gaps.

    Weather is disabled by default because the repository weather.csv is a rolling
    current forecast file, not an archive of forecasts as they were known at each
    historical cutoff. Set AUTOGLUON_INCLUDE_WEATHER=1 only for a diagnostic run.
    """
    flow = pd.read_csv(FLOW_URL)
    flow["ds"] = pd.to_datetime(flow["ds"], errors="coerce")
    flow = flow.dropna(subset=["ds"]).sort_values("ds")
    flow = base.canonicalize_flow_targets(flow)
    targets = validate_flow_targets(flow)

    # Reindex the target series before attaching covariates. This preserves the
    # actual staffing value for an hour even if the flow source happens to have a
    # missing row at that timestamp.
    full_index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    history = flow[["ds", *targets]].set_index("ds").reindex(full_index).reset_index()

    observed_columns: list[str] = []
    for target in targets:
        observed = f"__observed_{target}"
        history[observed] = history[target].notna()
        observed_columns.append(observed)
        history[target] = pd.to_numeric(history[target], errors="coerce").ffill().fillna(0.0)

    staffing = staffing_features(pd.read_csv(SHIFTS_URL))
    staffing_columns = [column for column in staffing.columns if column != "ds"]
    history = history.merge(staffing, on="ds", how="left")
    for column in staffing_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

    weather_columns: list[str] = []
    if INCLUDE_WEATHER:
        weather = pd.read_csv(WEATHER_URL)
        weather["ds"] = pd.to_datetime(weather["ds"], errors="coerce")
        weather_columns = [column for column in weather.columns if column != "ds"]
        history = history.merge(weather, on="ds", how="left")
        for column in weather_columns:
            # Still causal with respect to missing-value filling, but the values
            # themselves may be hindsight-contaminated because weather.csv is not
            # an as-of forecast archive. This mode is diagnostic only.
            history[column] = pd.to_numeric(history[column], errors="coerce").ffill().fillna(0.0)

    history = add_holidays(history)
    holiday_columns = ["is_qc_holiday", "is_jewish_holiday"]
    for column in holiday_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

    covariates = [*staffing_columns, *weather_columns, *holiday_columns]

    missing_counts = {
        target: int((~history[f"__observed_{target}"]).sum())
        for target in targets
    }
    print("Missing target hours before causal forward-fill:")
    for target, count in missing_counts.items():
        print(f"  {target}: {count}")

    return history, targets, covariates, observed_columns


def rolling_holdout_predictions(
    predictor,
    history: pd.DataFrame,
    train_history: pd.DataFrame,
    targets: list[str],
    covariates: list[str],
    models: list[str],
) -> pd.DataFrame:
    first_holdout = len(train_history)
    rows: list[pd.DataFrame] = []

    for window in range(base.HOLDOUT_WINDOWS):
        forecast_start_idx = first_holdout + window * base.PREDICTION_LENGTH
        forecast_end_idx = forecast_start_idx + base.PREDICTION_LENGTH
        if forecast_end_idx > len(history):
            break

        context_frame = history.iloc[:forecast_start_idx].copy()
        future_frame = history.iloc[forecast_start_idx:forecast_end_idx].copy()
        cutoff = context_frame["ds"].iloc[-1]

        context = to_long(context_frame, targets, covariates)
        known = future_long(future_frame[["ds", *covariates]], targets, covariates)

        actual_parts = []
        for target in targets:
            actual_parts.append(
                pd.DataFrame(
                    {
                        "ds": future_frame["ds"].to_numpy(),
                        "target_name": target,
                        "actual": future_frame[target].to_numpy(),
                        "observed": future_frame[f"__observed_{target}"].to_numpy(dtype=bool),
                    }
                )
            )
        actual = pd.concat(actual_parts, ignore_index=True)

        for model in models:
            print(
                f"Backtest window {window + 1}/{base.HOLDOUT_WINDOWS}: "
                f"{model} from cutoff {cutoff}"
            )
            try:
                forecast = base._prediction_frame(predictor, context, known, model)
            except Exception as exc:
                print(f"Skipping failed backtest prediction for {model}: {exc}")
                continue

            merged = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            merged = merged[merged["observed"]].copy()
            if merged.empty:
                print(f"Skipping {model} window {window + 1}: no observed outcomes")
                continue

            merged["model"] = model
            merged["window"] = window + 1
            merged["cutoff"] = cutoff
            merged["horizon_hours"] = (
                (merged["ds"] - pd.Timestamp(cutoff)).dt.total_seconds() / 3600
            ).astype(int)
            merged["error"] = merged["prediction"] - merged["actual"]
            merged["absolute_error"] = merged["error"].abs()
            merged["squared_error"] = merged["error"] ** 2
            rows.append(merged.drop(columns="observed"))

    if not rows:
        raise RuntimeError("No rolling holdout predictions were produced")

    return pd.concat(rows, ignore_index=True)


def summarize_backtests(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["model", "target_name"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            n=("error", "size"),
        )
    )

    wql_rows = []
    for (model, target_name), group in rows.groupby(["model", "target_name"], sort=False):
        wql_rows.append(
            {
                "model": model,
                "target_name": target_name,
                "WQL": base.weighted_quantile_loss(group),
            }
        )
    wql = pd.DataFrame(wql_rows)
    summary = summary.merge(wql, on=["model", "target_name"], how="left")

    horizons = rows[rows["horizon_hours"].isin(base.HORIZON_CHECKPOINTS)]
    horizon_summary = (
        horizons.groupby(["model", "target_name", "horizon_hours"], as_index=False)
        .agg(
            MAE=("absolute_error", "mean"),
            RMSE=("squared_error", lambda values: float(np.sqrt(values.mean()))),
            bias=("error", "mean"),
            n=("error", "size"),
        )
    )

    # Macro-average WQL gives each operational target equal weight regardless of
    # its numeric scale. Count how often a model is best by WQL and by MAE too.
    best_wql = summary.loc[summary.groupby("target_name")["WQL"].idxmin(), ["target_name", "model"]]
    best_mae = summary.loc[summary.groupby("target_name")["MAE"].idxmin(), ["target_name", "model"]]
    overview = (
        summary.groupby("model", as_index=False)
        .agg(
            macro_WQL=("WQL", "mean"),
            mean_target_MAE=("MAE", "mean"),
            mean_abs_bias=("bias", lambda values: float(np.abs(values).mean())),
            targets=("target_name", "nunique"),
        )
    )
    wql_wins = best_wql["model"].value_counts().rename("WQL_wins")
    mae_wins = best_mae["model"].value_counts().rename("MAE_wins")
    overview = overview.merge(wql_wins, left_on="model", right_index=True, how="left")
    overview = overview.merge(mae_wins, left_on="model", right_index=True, how="left")
    overview[["WQL_wins", "MAE_wins"]] = overview[["WQL_wins", "MAE_wins"]].fillna(0).astype(int)
    overview = overview.sort_values(["macro_WQL", "mean_target_MAE"])

    return summary, horizon_summary, overview


def main() -> None:
    mode = "weather-inclusive diagnostic" if INCLUDE_WEATHER else "leakage-safe no-weather"
    print(f"Backtest mode: {mode}")
    if INCLUDE_WEATHER:
        print(
            "WARNING: weather.csv is not an as-of historical forecast archive; "
            "weather-inclusive scores should not be used for model selection."
        )
    print(f"Flow targets: {', '.join(FLOW_TARGETS)}")
    print(
        "Backtest configuration: "
        f"{base.VALIDATION_WINDOWS} internal validation windows, "
        f"{base.HOLDOUT_WINDOWS} external {base.PREDICTION_LENGTH}-hour holdout windows"
    )

    history, targets, covariates, _ = prepare_history()
    train_history, holdout_history = base.split_train_holdout(history)
    print(f"Known covariates ({len(covariates)}): {', '.join(covariates)}")
    print(
        f"Training through {train_history['ds'].iloc[-1]}; "
        f"external holdout begins {holdout_history['ds'].iloc[0]}"
    )

    predictor = base.fit_predictor(train_history, targets, covariates)
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv(base.LEADERBOARD_PATH, index=False)
    print(f"Saved model leaderboard: {base.LEADERBOARD_PATH}")

    models = leaderboard["model"].tolist()
    rows = rolling_holdout_predictions(
        predictor=predictor,
        history=history,
        train_history=train_history,
        targets=targets,
        covariates=covariates,
        models=models,
    )
    rows.to_csv(base.BACKTEST_ROWS_PATH, index=False)

    summary, horizon_summary, overview = summarize_backtests(rows)
    summary = summary.sort_values(["target_name", "WQL", "MAE"], na_position="last")
    horizon_summary = horizon_summary.sort_values(["target_name", "horizon_hours", "MAE"])
    overview_path = os.environ.get("AUTOGLUON_OVERVIEW_PATH", "autogluon_backtest_overview.csv")

    summary.to_csv(base.BACKTEST_SUMMARY_PATH, index=False)
    horizon_summary.to_csv(base.BACKTEST_HORIZONS_PATH, index=False)
    overview.to_csv(overview_path, index=False)

    print(f"Saved raw backtest rows: {base.BACKTEST_ROWS_PATH}")
    print(f"Saved backtest summary: {base.BACKTEST_SUMMARY_PATH}")
    print(f"Saved horizon summary: {base.BACKTEST_HORIZONS_PATH}")
    print(f"Saved model overview: {overview_path}")
    print("\nModel overview across six targets:")
    print(overview.to_string(index=False))
    print("\nExternal holdout results by target:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
