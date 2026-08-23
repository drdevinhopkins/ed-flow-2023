from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import autogluon_backtest as ag_base
import autogluon_backtest_safe as ag_safe
import chronos_native_backtest as native
from chronos_forecast_autogluon import (
    FLOW_TARGETS,
    FLOW_URL,
    SHIFTS_URL,
    validate_flow_targets,
)
from forecast_oncall_impact import build_staffing_features

AG_ROWS_PATH = Path(os.environ.get("STAFFING_AG_ROWS_PATH", "staffing_autogluon_backtest_rows.csv"))
AG_SUMMARY_PATH = Path(os.environ.get("STAFFING_AG_SUMMARY_PATH", "staffing_autogluon_backtest_summary.csv"))
AG_HORIZONS_PATH = Path(os.environ.get("STAFFING_AG_HORIZONS_PATH", "staffing_autogluon_backtest_horizons.csv"))
AG_LEADERBOARD_PATH = Path(os.environ.get("STAFFING_AG_LEADERBOARD_PATH", "staffing_autogluon_leaderboard.csv"))
NATIVE_ROWS_PATH = Path(os.environ.get("STAFFING_NATIVE_ROWS_PATH", "staffing_native_backtest_rows.csv"))
NATIVE_SUMMARY_PATH = Path(os.environ.get("STAFFING_NATIVE_SUMMARY_PATH", "staffing_native_backtest_summary.csv"))
NATIVE_HORIZONS_PATH = Path(os.environ.get("STAFFING_NATIVE_HORIZONS_PATH", "staffing_native_backtest_horizons.csv"))
TARGET_COMPARISON_PATH = Path(os.environ.get("STAFFING_TARGET_COMPARISON_PATH", "staffing_chronos_autogluon_target_comparison.csv"))
MODEL_SUMMARY_PATH = Path(os.environ.get("STAFFING_MODEL_SUMMARY_PATH", "staffing_chronos_autogluon_model_summary.csv"))
RUNTIME_PATH = Path(os.environ.get("STAFFING_RUNTIME_PATH", "staffing_chronos_autogluon_runtime.csv"))


def prepare_staffing_only_history() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a leakage-safe history with the physician-aware staffing covariates."""
    flow = pd.read_csv(FLOW_URL)
    flow["ds"] = pd.to_datetime(flow["ds"], format="mixed", errors="coerce")
    flow = flow.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")
    flow = ag_base.canonicalize_flow_targets(flow)
    targets = validate_flow_targets(flow)

    full_index = pd.date_range(flow["ds"].min(), flow["ds"].max(), freq="h", name="ds")
    history = flow[["ds", *targets]].set_index("ds").reindex(full_index).reset_index()

    for target in targets:
        observed = f"__observed_{target}"
        history[observed] = history[target].notna()
        history[target] = pd.to_numeric(history[target], errors="coerce").ffill().fillna(0.0)

    # Match the staffing representation that improved all six targets in the
    # covariate-ablation backtest: physician identity/role, role counts, and the
    # explicitly scheduled on-call physician.
    staffing = build_staffing_features(pd.read_csv(SHIFTS_URL)).copy()
    staffing["ds"] = pd.to_datetime(staffing["ds"], format="mixed", errors="coerce").dt.floor("h")
    staffing = staffing.dropna(subset=["ds"]).drop_duplicates("ds", keep="last").sort_values("ds")
    covariates = [column for column in staffing.columns if column != "ds"]

    history = history.merge(staffing, on="ds", how="left")
    for column in covariates:
        if column.startswith("physician__"):
            history[column] = history[column].fillna("NotWorking").astype(str)
        elif column == "oncall_physician_id":
            history[column] = history[column].fillna("None").astype(str)
        else:
            history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

    return history, targets, covariates


def build_actual(future_frame: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    parts = []
    for target in targets:
        parts.append(
            pd.DataFrame(
                {
                    "ds": future_frame["ds"].to_numpy(),
                    "target_name": target,
                    "actual": future_frame[target].to_numpy(),
                    "observed": future_frame[f"__observed_{target}"].to_numpy(dtype=bool),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def rolling_autogluon_predictions(
    predictor,
    history: pd.DataFrame,
    train_history: pd.DataFrame,
    targets: list[str],
    covariates: list[str],
    models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_holdout = len(train_history)
    rows: list[pd.DataFrame] = []
    runtimes: list[dict[str, object]] = []

    for window in range(ag_base.HOLDOUT_WINDOWS):
        forecast_start_idx = first_holdout + window * ag_base.PREDICTION_LENGTH
        forecast_end_idx = forecast_start_idx + ag_base.PREDICTION_LENGTH
        if forecast_end_idx > len(history):
            break

        context_frame = history.iloc[:forecast_start_idx].copy()
        future_frame = history.iloc[forecast_start_idx:forecast_end_idx].copy()
        cutoff = context_frame["ds"].iloc[-1]

        context = ag_base.to_long(context_frame, targets, covariates)
        known = ag_base.future_long(future_frame[["ds", *covariates]], targets, covariates)
        actual = build_actual(future_frame, targets)

        for model in models:
            print(
                f"AutoGluon window {window + 1}/{ag_base.HOLDOUT_WINDOWS}: "
                f"{model} from cutoff {cutoff}"
            )
            started = time.perf_counter()
            try:
                forecast = ag_base._prediction_frame(predictor, context, known, model)
            except Exception as exc:
                print(f"Skipping failed prediction for {model}: {exc}")
                continue
            elapsed = time.perf_counter() - started
            runtimes.append(
                {
                    "framework": "AutoGluon",
                    "model": model,
                    "window": window + 1,
                    "cutoff": cutoff,
                    "prediction_seconds": elapsed,
                }
            )

            merged = forecast.merge(actual, on=["ds", "target_name"], how="inner")
            merged = merged[merged["observed"]].copy()
            if merged.empty:
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
        raise RuntimeError("No AutoGluon holdout predictions were produced")

    return pd.concat(rows, ignore_index=True), pd.DataFrame(runtimes)


def select_ag_chronos(summary: pd.DataFrame) -> pd.DataFrame:
    mask = summary["model"].astype(str).str.startswith("Chronos2")
    candidates = summary[mask].copy()
    if candidates.empty:
        raise RuntimeError("AutoGluon summary contains no Chronos2 model")
    return candidates.loc[candidates.groupby("target_name")["MAE"].idxmin()].copy()


def target_comparison(native_summary: pd.DataFrame, ag_summary: pd.DataFrame) -> pd.DataFrame:
    native_focus = native_summary[["target_name", "MAE", "RMSE", "WQL"]].rename(
        columns={"MAE": "native_MAE", "RMSE": "native_RMSE", "WQL": "native_WQL"}
    )
    ag_chronos = select_ag_chronos(ag_summary)[["target_name", "model", "MAE", "RMSE", "WQL"]].rename(
        columns={
            "model": "ag_chronos_model",
            "MAE": "ag_chronos_MAE",
            "RMSE": "ag_chronos_RMSE",
            "WQL": "ag_chronos_WQL",
        }
    )
    best_ag = ag_summary.loc[ag_summary.groupby("target_name")["MAE"].idxmin(), ["target_name", "model", "MAE", "RMSE", "WQL"]].rename(
        columns={
            "model": "best_ag_model",
            "MAE": "best_ag_MAE",
            "RMSE": "best_ag_RMSE",
            "WQL": "best_ag_WQL",
        }
    )

    result = native_focus.merge(ag_chronos, on="target_name", how="inner").merge(best_ag, on="target_name", how="left")
    result["ag_chronos_MAE_improvement_pct_vs_native"] = (
        (result["native_MAE"] - result["ag_chronos_MAE"]) / result["native_MAE"] * 100.0
    )
    result["ag_chronos_WQL_improvement_pct_vs_native"] = (
        (result["native_WQL"] - result["ag_chronos_WQL"]) / result["native_WQL"] * 100.0
    )
    result["best_ag_MAE_improvement_pct_vs_native"] = (
        (result["native_MAE"] - result["best_ag_MAE"]) / result["native_MAE"] * 100.0
    )
    return result.sort_values("target_name")


def model_summary(native_summary: pd.DataFrame, ag_summary: pd.DataFrame) -> pd.DataFrame:
    native_copy = native_summary.copy()
    native_copy["framework"] = "Native"
    ag_copy = ag_summary.copy()
    ag_copy["framework"] = "AutoGluon"
    combined = pd.concat([native_copy, ag_copy], ignore_index=True, sort=False)
    return (
        combined.groupby(["framework", "model"], as_index=False)
        .agg(
            macro_WQL=("WQL", "mean"),
            mean_target_MAE=("MAE", "mean"),
            mean_target_RMSE=("RMSE", "mean"),
            targets=("target_name", "nunique"),
        )
        .sort_values(["macro_WQL", "mean_target_MAE"])
    )


def main() -> None:
    print("Chronos-2 vs AutoGluon benchmark: physician-aware staffing-only covariates")
    print(f"Targets: {', '.join(FLOW_TARGETS)}")
    print(
        f"Configuration: {ag_base.HOLDOUT_WINDOWS} external "
        f"{ag_base.PREDICTION_LENGTH}-hour holdout windows; "
        f"{ag_base.VALIDATION_WINDOWS} AutoGluon validation windows"
    )

    history, targets, covariates = prepare_staffing_only_history()
    train_history, holdout_history = ag_base.split_train_holdout(history)
    categorical = [column for column in covariates if column.startswith("physician__") or column == "oncall_physician_id"]
    numeric = [column for column in covariates if column not in categorical]
    print(
        f"Staffing covariates: {len(covariates)} total "
        f"({len(categorical)} categorical identity/role, {len(numeric)} numeric counts)"
    )
    print(
        f"Context ends {train_history['ds'].iloc[-1]}; "
        f"holdout begins {holdout_history['ds'].iloc[0]}"
    )

    fit_started = time.perf_counter()
    predictor = ag_base.fit_predictor(train_history, targets, covariates)
    fit_seconds = time.perf_counter() - fit_started
    leaderboard = predictor.leaderboard()
    leaderboard.to_csv(AG_LEADERBOARD_PATH, index=False)

    models = leaderboard["model"].tolist()
    ag_rows, ag_runtime = rolling_autogluon_predictions(
        predictor=predictor,
        history=history,
        train_history=train_history,
        targets=targets,
        covariates=covariates,
        models=models,
    )
    ag_summary, ag_horizons, _ = ag_safe.summarize_backtests(ag_rows)

    pipeline = native.load_pipeline()
    native_rows, native_runtime = native.rolling_native_predictions(
        pipeline=pipeline,
        history=history,
        train_history=train_history,
        targets=targets,
        covariates=covariates,
    )
    native_summary, native_horizons = native.summarize(native_rows)

    ag_rows.to_csv(AG_ROWS_PATH, index=False)
    ag_summary.to_csv(AG_SUMMARY_PATH, index=False)
    ag_horizons.to_csv(AG_HORIZONS_PATH, index=False)
    native_rows.to_csv(NATIVE_ROWS_PATH, index=False)
    native_summary.to_csv(NATIVE_SUMMARY_PATH, index=False)
    native_horizons.to_csv(NATIVE_HORIZONS_PATH, index=False)

    comparison = target_comparison(native_summary, ag_summary)
    comparison.to_csv(TARGET_COMPARISON_PATH, index=False)

    models_summary = model_summary(native_summary, ag_summary)
    models_summary.to_csv(MODEL_SUMMARY_PATH, index=False)

    native_runtime = native_runtime.copy()
    native_runtime["framework"] = "Native"
    native_runtime["model"] = "Chronos2Native"
    runtime = pd.concat(
        [
            ag_runtime[["framework", "model", "window", "cutoff", "prediction_seconds"]],
            native_runtime[["framework", "model", "window", "cutoff", "prediction_seconds"]],
        ],
        ignore_index=True,
    )
    runtime["autogluon_fit_seconds"] = np.nan
    if not runtime.empty:
        runtime.loc[runtime.index[0], "autogluon_fit_seconds"] = fit_seconds
    runtime.to_csv(RUNTIME_PATH, index=False)

    print(f"AutoGluon fit time: {fit_seconds:.1f}s")
    print("\nNative vs AutoGluon Chronos-2 by target:")
    print(comparison.to_string(index=False))
    print("\nAll-model macro summary:")
    print(models_summary.to_string(index=False))
    print("\nMean prediction runtime per 24h window:")
    print(
        runtime.groupby(["framework", "model"], as_index=False)["prediction_seconds"]
        .mean()
        .sort_values("prediction_seconds")
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
