#!/usr/bin/env python3
"""Backtest an intraday forecast of final Montreal-calendar-day ED arrivals.

For every hourly report, the experiment predicts how many additional patients will
arrive before the end of the local calendar day.  The final-total forecast is always
constructed as::

    observed_so_far + max(0, predicted_remaining)

This makes the operational invariant explicit: a final forecast can never be lower than
the number of patients already observed.

The runner is deliberately isolated from production.  It compares two transparent
baselines with pooled gradient-boosted completion models, uses expanding time-ordered
folds, and writes only validation artifacts.  Rich calendar features reuse the existing
JGH calendar feature builder.  Optional weather inputs are restricted to observations at
or before each cutoff; realized future weather is never used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


SCRIPT_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

FLOW_URL = (
    "https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/"
    "allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&raw=1"
)
LOCAL_TZ = "America/Montreal"
QUANTILES = (0.1, 0.5, 0.9)

ARRIVAL_COLUMNS = (
    "Inflow_Total",
    "INFLOW_STRETCHER",
    "INFLOW_AMBULATORY",
    "INFLOW_AMBULANCES",
)

# All are contemporaneous ED-state fields, available at the report cutoff. Arrival
# cumulative counters are intentionally excluded; cumulative arrivals are rebuilt from
# hourly Inflow_Total because the source counters have changed reset conventions over
# time.
STATE_CANDIDATES = (
    "FLS",
    "CUM_ADMREQ",
    "CUM_BA1",
    "WAITINGADM",
    "TTStr",
    "TRG_HALLWAY1",
    "TRG_HALLWAY_TBS",
    "reoriented_cum",
    "reoriented_cum_MD",
    "QTRACK1",
    "RESUS",
    "Pod_T",
    "POD_GREEN",
    "POD_GREEN_TBS",
    "POD_YELLOW",
    "POD_YELLOW_TBS",
    "POD_ORANGE",
    "POD_ORANGE_TBS",
    "POD_CONS_MORE2H",
    "POD_IMCONS_MORE4H",
    "POD_XRAY_MORE2H",
    "POD_CT_MORE2H",
    "POST_POD1",
    "VERTSTRET",
    "RAZ_TBS",
    "RAZ_LAZYBOY",
    "RAZ_WAITINGREZ",
    "AMBVERT1",
    "AMBVERTTBS",
    "QTrack_TBS",
    "Garage_TBS",
    "RAZ_CONS_MORE2H",
    "RAZ_IMCONS_MORE4H",
    "RAZ_XRAY_MORE2H",
    "RAZ_CT_MORE2H1",
    "PSYCH1",
    "PSYCH_WAITINGADM",
)

TOTAL_TBS_COMPONENTS = (
    "TRG_HALLWAY_TBS",
    "POD_GREEN_TBS",
    "POD_YELLOW_TBS",
    "POD_ORANGE_TBS",
    "RAZ_TBS",
    "AMBVERTTBS",
    "QTrack_TBS",
    "Garage_TBS",
)
POD_TBS_COMPONENTS = (
    "TRG_HALLWAY_TBS",
    "POD_GREEN_TBS",
    "POD_YELLOW_TBS",
    "POD_ORANGE_TBS",
)
VERTICAL_TBS_COMPONENTS = (
    "RAZ_TBS",
    "AMBVERTTBS",
    "QTrack_TBS",
    "Garage_TBS",
)

WEATHER_COLUMNS = (
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
    "pressure_msl",
    "cloud_cover",
)

PROGRESS_FEATURES = (
    "cutoff_hour",
    "cutoff_hour_sin",
    "cutoff_hour_cos",
    "report_index",
    "reports_remaining",
    "day_progress_fraction",
    "cumulative_arrivals",
    "inflow_last_1h",
    "inflow_last_2h",
    "inflow_last_3h",
    "inflow_last_6h",
    "inflow_last_12h",
    "inflow_acceleration_3h",
    "cumulative_stretcher",
    "cumulative_ambulatory",
    "cumulative_ambulances",
    "stretcher_arrival_share",
    "ambulance_arrival_share",
    "prior_total",
    "prior_trailing_28d",
    "prior_same_weekday_8",
    "expected_fraction",
    "expected_cumulative_from_prior",
    "pace_residual",
)


def parse_local_timestamp(series: pd.Series, local_tz: str = LOCAL_TZ) -> pd.Series:
    """Parse naive local or offset-aware timestamps to naive local wall-clock time."""

    values: list[pd.Timestamp | pd.NaT] = []
    for raw in series:
        try:
            stamp = pd.Timestamp(raw)
        except (TypeError, ValueError):
            values.append(pd.NaT)
            continue
        if pd.isna(stamp):
            values.append(pd.NaT)
            continue
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(local_tz).tz_localize(None)
        values.append(stamp.floor("h"))
    return pd.Series(pd.to_datetime(values), index=series.index, dtype="datetime64[ns]")


def expected_local_hours(day: pd.Timestamp, local_tz: str = LOCAL_TZ) -> list[int]:
    """Return the 23/24/25 local hour labels expected for one Montreal day."""

    start = pd.Timestamp(day).normalize().tz_localize(local_tz)
    end = (pd.Timestamp(day).normalize() + pd.Timedelta(days=1)).tz_localize(local_tz)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return hours.tz_localize(None).hour.tolist()


def _complete_day_flags(frame: pd.DataFrame) -> pd.Series:
    flags: dict[pd.Timestamp, bool] = {}
    for day, group in frame.groupby("day", sort=True):
        expected = sorted(expected_local_hours(pd.Timestamp(day)))
        actual = sorted(group["ds"].dt.hour.astype(int).tolist())
        numeric = pd.to_numeric(group["Inflow_Total"], errors="coerce")
        flags[pd.Timestamp(day)] = actual == expected and numeric.notna().all()
    return frame["day"].map(flags).fillna(False).astype(bool)


def load_hourly_flow(source: str | Path) -> pd.DataFrame:
    """Load hourly flow without regularizing or imputing missing arrival hours."""

    raw = pd.read_csv(source)
    required = {"ds", "Inflow_Total"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Flow source missing columns: {sorted(missing)}")

    # Exact duplicate source timestamps are revisions, not extra hours. Offset-distinct
    # autumn DST timestamps remain separate and become two local rows with hour == 1.
    raw = raw.drop_duplicates("ds", keep="last").copy()
    raw["_source_order"] = np.arange(len(raw), dtype=int)
    raw["ds"] = parse_local_timestamp(raw["ds"])
    raw["Inflow_Total"] = pd.to_numeric(raw["Inflow_Total"], errors="coerce")
    raw = raw.dropna(subset=["ds"]).sort_values(["ds", "_source_order"]).reset_index(drop=True)
    raw["day"] = raw["ds"].dt.normalize()
    raw["is_complete_day"] = _complete_day_flags(raw)
    return raw


def _rolling_prior(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values("day").copy()
    daily["weekday"] = daily["day"].dt.dayofweek
    daily["prior_trailing_28d"] = daily["final_total"].shift(1).rolling(28, min_periods=7).mean()
    daily["prior_same_weekday_8"] = daily.groupby("weekday", sort=False)["final_total"].transform(
        lambda values: values.shift(1).rolling(8, min_periods=3).mean()
    )
    both = daily[["prior_trailing_28d", "prior_same_weekday_8"]]
    daily["prior_total"] = both.mean(axis=1, skipna=True)
    return daily.drop(columns="weekday")


def _basic_calendar_features(days: pd.Series) -> pd.DataFrame:
    stamps = pd.to_datetime(days)
    out = pd.DataFrame({"day": stamps})
    out["calendar_year"] = stamps.dt.year.astype(float)
    out["calendar_month"] = stamps.dt.month.astype(float)
    out["calendar_day_of_week"] = stamps.dt.dayofweek.astype(float)
    out["calendar_is_weekend"] = stamps.dt.dayofweek.ge(5).astype(float)
    day_of_year = stamps.dt.dayofyear.astype(float)
    out["calendar_day_of_year_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    out["calendar_day_of_year_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    out["calendar_linear_trend_days"] = (stamps - stamps.min()).dt.days.astype(float)
    return out


def add_calendar_features(days: pd.Series, mode: str) -> pd.DataFrame:
    """Build basic or existing rich JGH calendar covariates for unique days."""

    base = _basic_calendar_features(days)
    if mode == "basic":
        return base
    if mode != "rich":
        raise ValueError("calendar mode must be 'basic' or 'rich'")

    try:
        from calendar_context_features import add_calendar_context_features
    except ImportError as exc:  # pragma: no cover - depends on optional runtime packages
        raise RuntimeError(
            "Rich calendar context requires the repository requirements, including holidays"
        ) from exc

    rich_input = pd.DataFrame({"ds": pd.to_datetime(days)})
    rich = add_calendar_context_features(rich_input, ts_col="ds").drop(columns="ds")
    rich = rich.rename(columns=lambda column: f"calendar_{column}")
    return pd.concat([base.reset_index(drop=True), rich.reset_index(drop=True)], axis=1)


def _derive_sum(frame: pd.DataFrame, name: str, components: Sequence[str]) -> None:
    if not all(column in frame.columns for column in components):
        return
    numeric = frame.loc[:, components].apply(pd.to_numeric, errors="coerce")
    frame[name] = numeric.sum(axis=1, min_count=len(components))


def add_state_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    _derive_sum(out, "Total_TBS", TOTAL_TBS_COMPONENTS)
    _derive_sum(out, "POD_TBS", POD_TBS_COMPONENTS)
    _derive_sum(out, "Vertical_TBS", VERTICAL_TBS_COMPONENTS)
    _derive_sum(out, "Overflow", ("POST_POD1", "TRG_HALLWAY1"))

    candidates = [
        column
        for column in [*STATE_CANDIDATES, "Total_TBS", "POD_TBS", "Vertical_TBS", "Overflow"]
        if column in out.columns
    ]
    for column in candidates:
        values = pd.to_numeric(out[column], errors="coerce")
        out[f"state_{column}"] = values

    change_columns = [
        column
        for column in (
            "Total_TBS",
            "POD_TBS",
            "Vertical_TBS",
            "Overflow",
            "WAITINGADM",
            "TTStr",
            "TRG_HALLWAY1",
            "TRG_HALLWAY_TBS",
            "RESUS",
        )
        if f"state_{column}" in out.columns
    ]
    for column in change_columns:
        grouped = out.groupby("day", sort=False)[f"state_{column}"]
        out[f"state_{column}_change_1h"] = grouped.diff(1)
        out[f"state_{column}_change_3h"] = grouped.diff(3)
    return out


def build_weather_features(source: str | Path) -> pd.DataFrame:
    """Build cutoff-safe current/trailing weather features from hourly observations."""

    weather = pd.read_csv(source)
    if "ds" not in weather.columns:
        raise ValueError("Weather source must contain ds")
    weather["ds"] = parse_local_timestamp(weather["ds"])
    weather = weather.dropna(subset=["ds"]).sort_values("ds").drop_duplicates("ds", keep="last")

    available = [column for column in WEATHER_COLUMNS if column in weather.columns]
    if not available:
        raise ValueError(f"Weather source has none of the expected columns: {list(WEATHER_COLUMNS)}")
    weather[available] = weather[available].apply(pd.to_numeric, errors="coerce")
    indexed = weather.set_index("ds")

    featured = pd.DataFrame(index=indexed.index)
    for column in available:
        featured[f"weather_current_{column}"] = indexed[column]
        aggregation = "sum" if column in {"precipitation", "rain", "snowfall"} else "mean"
        for hours in (6, 24):
            rolling = indexed[column].rolling(f"{hours}h", min_periods=1)
            featured[f"weather_{column}_{aggregation}_{hours}h"] = getattr(rolling, aggregation)()
    return featured.reset_index().sort_values("ds")


def attach_weather(snapshots: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of merge: no weather timestamp after the cutoff can be attached."""

    left = snapshots.sort_values("ds").copy()
    right = weather.sort_values("ds").copy()
    # Pandas 3 may preserve microsecond resolution for one input and nanosecond
    # resolution for the other. merge_asof requires exact dtype equality even though
    # both are ordinary datetimes, so normalize the unit explicitly.
    left["ds"] = pd.to_datetime(left["ds"], errors="coerce").astype("datetime64[ns]")
    right["ds"] = pd.to_datetime(right["ds"], errors="coerce").astype("datetime64[ns]")
    return pd.merge_asof(left, right, on="ds", direction="backward", tolerance=pd.Timedelta(hours=2))


def build_snapshots(
    flow: pd.DataFrame,
    *,
    calendar_mode: str = "rich",
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one leakage-safe training row for every cutoff in every complete day."""

    complete = flow.loc[flow["is_complete_day"]].copy()
    if complete.empty:
        raise ValueError("No complete Montreal calendar days in flow source")
    complete = complete.sort_values(["day", "ds", "_source_order"]).reset_index(drop=True)

    complete["final_total"] = complete.groupby("day")["Inflow_Total"].transform("sum")
    complete["cumulative_arrivals"] = complete.groupby("day")["Inflow_Total"].cumsum()
    complete["remaining_arrivals"] = complete["final_total"] - complete["cumulative_arrivals"]
    complete["report_index"] = complete.groupby("day").cumcount() + 1
    complete["expected_reports"] = complete["day"].map(
        lambda day: len(expected_local_hours(pd.Timestamp(day)))
    )
    complete["reports_remaining"] = complete["expected_reports"] - complete["report_index"]
    complete["day_progress_fraction"] = complete["report_index"] / complete["expected_reports"]
    complete["cutoff_hour"] = complete["ds"].dt.hour.astype(float)
    radians = 2.0 * np.pi * complete["cutoff_hour"] / 24.0
    complete["cutoff_hour_sin"] = np.sin(radians)
    complete["cutoff_hour_cos"] = np.cos(radians)

    grouped_inflow = complete.groupby("day", sort=False)["Inflow_Total"]
    for hours in (1, 2, 3, 6, 12):
        complete[f"inflow_last_{hours}h"] = grouped_inflow.transform(
            lambda values, window=hours: values.rolling(window, min_periods=1).sum()
        )
    previous_3h = grouped_inflow.transform(
        lambda values: values.shift(3).rolling(3, min_periods=1).sum()
    )
    complete["inflow_acceleration_3h"] = complete["inflow_last_3h"] - previous_3h

    cumulative_names = {
        "INFLOW_STRETCHER": "cumulative_stretcher",
        "INFLOW_AMBULATORY": "cumulative_ambulatory",
        "INFLOW_AMBULANCES": "cumulative_ambulances",
    }
    for source, target in cumulative_names.items():
        if source in complete.columns:
            values = pd.to_numeric(complete[source], errors="coerce")
            complete[target] = values.groupby(complete["day"]).cumsum()
        else:
            complete[target] = np.nan
    denominator = complete["cumulative_arrivals"].replace(0, np.nan)
    complete["stretcher_arrival_share"] = complete["cumulative_stretcher"] / denominator
    complete["ambulance_arrival_share"] = complete["cumulative_ambulances"] / denominator

    daily = complete[["day", "final_total"]].drop_duplicates("day")
    daily = _rolling_prior(daily)
    complete = complete.merge(daily.drop(columns="final_total"), on="day", how="left")

    calendar = add_calendar_features(daily["day"], calendar_mode)
    complete = complete.merge(calendar, on="day", how="left")
    complete = add_state_features(complete)

    if weather is not None:
        complete = attach_weather(complete, weather)
    return complete.sort_values(["day", "ds", "_source_order"]).reset_index(drop=True)


def build_expanding_folds(
    snapshots: pd.DataFrame,
    *,
    n_folds: int,
    test_days: int,
    min_train_days: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    days = np.array(sorted(pd.to_datetime(snapshots["day"].unique())))
    required = min_train_days + n_folds * test_days
    if len(days) < required:
        raise ValueError(f"Need at least {required} complete days; found {len(days)}")

    first_test = len(days) - n_folds * test_days
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    for fold_id in range(n_folds):
        start = first_test + fold_id * test_days
        stop = start + test_days
        train_days = days[:start]
        test_block = days[start:stop]
        train_mask = snapshots["day"].isin(train_days).to_numpy()
        test_mask = snapshots["day"].isin(test_block).to_numpy()
        folds.append((fold_id, train_mask, test_mask))
    return folds


def fit_completion_curve(train: pd.DataFrame) -> pd.DataFrame:
    usable = train.loc[(train["final_total"] > 0) & (train["cumulative_arrivals"] > 0)].copy()
    usable["fraction"] = usable["cumulative_arrivals"] / usable["final_total"]
    usable["completion_factor"] = usable["final_total"] / usable["cumulative_arrivals"]
    grouped = usable.groupby("cutoff_hour")
    curve = grouped.agg(expected_fraction=("fraction", "median"), n=("fraction", "size"))
    curve["factor_p10"] = grouped["completion_factor"].quantile(0.1)
    curve["factor_p50"] = grouped["completion_factor"].quantile(0.5)
    curve["factor_p90"] = grouped["completion_factor"].quantile(0.9)
    curve = curve.reindex(range(24)).interpolate(limit_direction="both")
    curve["expected_fraction"] = curve["expected_fraction"].cummax().clip(1e-4, 1.0)
    return curve


def add_curve_features(frame: pd.DataFrame, curve: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hours = out["cutoff_hour"].astype(int)
    out["expected_fraction"] = hours.map(curve["expected_fraction"])
    out["expected_cumulative_from_prior"] = out["prior_total"] * out["expected_fraction"]
    out["pace_residual"] = out["cumulative_arrivals"] - out["expected_cumulative_from_prior"]
    return out


def _prediction_rows(
    frame: pd.DataFrame,
    *,
    fold_id: int,
    model: str,
    predicted_total: np.ndarray,
    lower_total: np.ndarray,
    upper_total: np.ndarray,
) -> pd.DataFrame:
    observed = frame["cumulative_arrivals"].to_numpy(dtype=float)
    point = np.maximum(np.asarray(predicted_total, dtype=float), observed)
    lower = np.maximum(observed, np.minimum(np.asarray(lower_total, dtype=float), point))
    upper = np.maximum(point, np.asarray(upper_total, dtype=float))
    ordered = np.column_stack([lower, point, upper])
    return pd.DataFrame(
        {
            "fold": fold_id,
            "model": model,
            "day": frame["day"].to_numpy(),
            "ds": frame["ds"].to_numpy(),
            "cutoff_hour": frame["cutoff_hour"].to_numpy(dtype=int),
            "observed_so_far": observed,
            "actual_total": frame["final_total"].to_numpy(dtype=float),
            "actual_remaining": frame["remaining_arrivals"].to_numpy(dtype=float),
            "predicted_total": ordered[:, 1],
            "p10_total": ordered[:, 0],
            "p90_total": ordered[:, 2],
            "predicted_remaining": np.maximum(0.0, ordered[:, 1] - observed),
        }
    )


def predict_completion_curve(
    test: pd.DataFrame, curve: pd.DataFrame, *, fold_id: int
) -> pd.DataFrame:
    hours = test["cutoff_hour"].astype(int)
    observed = test["cumulative_arrivals"].to_numpy(dtype=float)
    prior = test["prior_total"].fillna(test["cumulative_arrivals"]).to_numpy(dtype=float)
    factors = np.column_stack(
        [hours.map(curve[column]).to_numpy(dtype=float) for column in ("factor_p10", "factor_p50", "factor_p90")]
    )
    totals = observed[:, None] * factors
    zero = observed <= 0
    totals[zero, :] = prior[zero, None]
    return _prediction_rows(
        test,
        fold_id=fold_id,
        model="completion_curve",
        predicted_total=totals[:, 1],
        lower_total=totals[:, 0],
        upper_total=totals[:, 2],
    )


def fit_prior_update(train: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for hour in range(24):
        group = train.loc[train["cutoff_hour"].astype(int).eq(hour)].dropna(
            subset=["prior_total", "pace_residual", "final_total"]
        )
        if len(group) < 10:
            rows.append({"cutoff_hour": hour, "beta": 1.0, "residual_p10": 0.0, "residual_p90": 0.0})
            continue
        x = group["pace_residual"].to_numpy(dtype=float)
        y = (group["final_total"] - group["prior_total"]).to_numpy(dtype=float)
        beta = float(np.dot(x, y) / (np.dot(x, x) + 1e-6))
        beta = float(np.clip(beta, 0.0, 5.0))
        fitted = group["prior_total"].to_numpy(dtype=float) + beta * x
        residual = group["final_total"].to_numpy(dtype=float) - fitted
        rows.append(
            {
                "cutoff_hour": hour,
                "beta": beta,
                "residual_p10": float(np.quantile(residual, 0.1)),
                "residual_p90": float(np.quantile(residual, 0.9)),
            }
        )
    return pd.DataFrame(rows).set_index("cutoff_hour")


def predict_prior_update(
    test: pd.DataFrame,
    params: pd.DataFrame,
    curve_prediction: pd.DataFrame,
    *,
    fold_id: int,
) -> pd.DataFrame:
    hours = test["cutoff_hour"].astype(int)
    beta = hours.map(params["beta"]).to_numpy(dtype=float)
    prior = test["prior_total"].to_numpy(dtype=float)
    point = prior + beta * test["pace_residual"].to_numpy(dtype=float)
    lower = point + hours.map(params["residual_p10"]).to_numpy(dtype=float)
    upper = point + hours.map(params["residual_p90"]).to_numpy(dtype=float)

    missing = ~np.isfinite(point)
    if missing.any():
        point[missing] = curve_prediction.loc[missing, "predicted_total"].to_numpy(dtype=float)
        lower[missing] = curve_prediction.loc[missing, "p10_total"].to_numpy(dtype=float)
        upper[missing] = curve_prediction.loc[missing, "p90_total"].to_numpy(dtype=float)
    return _prediction_rows(
        test,
        fold_id=fold_id,
        model="prior_update",
        predicted_total=point,
        lower_total=lower,
        upper_total=upper,
    )


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    progress = [column for column in PROGRESS_FEATURES if column in frame.columns]
    calendar = [column for column in frame.columns if column.startswith("calendar_")]
    state = [column for column in frame.columns if column.startswith("state_")]
    weather = [column for column in frame.columns if column.startswith("weather_")]
    sets = {
        "boosted_progress": progress,
        "boosted_calendar": list(dict.fromkeys([*progress, *calendar])),
        "boosted_state": list(dict.fromkeys([*progress, *state])),
        "boosted_full": list(dict.fromkeys([*progress, *calendar, *state, *weather])),
    }
    if weather:
        sets["boosted_calendar_weather"] = list(dict.fromkeys([*progress, *calendar, *weather]))
    return sets


def _fit_quantile_models(
    train: pd.DataFrame,
    *,
    features: Sequence[str],
    max_iter: int,
    random_state: int,
) -> list[HistGradientBoostingRegressor]:
    x_train = train.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    y_train = train["remaining_arrivals"].to_numpy(dtype=float)
    models: list[HistGradientBoostingRegressor] = []
    for quantile in QUANTILES:
        objective = (
            {"loss": "squared_error"}
            if quantile == 0.5
            else {"loss": "quantile", "quantile": quantile}
        )
        model = HistGradientBoostingRegressor(
            **objective,
            learning_rate=0.05,
            max_iter=max_iter,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )
        model.fit(x_train, y_train)
        models.append(model)
    return models


def _predict_remaining_quantiles(
    models: Sequence[HistGradientBoostingRegressor],
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
) -> np.ndarray:
    x = frame.loc[:, features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return np.column_stack([np.maximum(0.0, model.predict(x)) for model in models])


def fit_quantile_corrections(
    actual_remaining: np.ndarray,
    predicted_remaining: np.ndarray,
    cutoff_hours: np.ndarray,
    *,
    shrinkage_days: float,
) -> pd.DataFrame:
    """Estimate hour-specific residual quantiles with shrinkage to a pooled correction."""

    actual = np.asarray(actual_remaining, dtype=float)
    predicted = np.asarray(predicted_remaining, dtype=float)
    hours = np.asarray(cutoff_hours, dtype=int)
    if predicted.ndim != 2 or predicted.shape[1] != len(QUANTILES):
        raise ValueError(f"predicted_remaining must have shape (n, {len(QUANTILES)})")
    if len(actual) != len(predicted) or len(hours) != len(predicted):
        raise ValueError("calibration arrays must have the same number of rows")
    if shrinkage_days < 0:
        raise ValueError("shrinkage_days must be non-negative")

    def residual_stat(values: np.ndarray, quantile: float) -> float:
        # The point forecast has an explicit mean-bias gate, so correct it with
        # the mean residual. Tail forecasts retain quantile/conformal correction.
        return float(values.mean()) if quantile == 0.5 else float(np.quantile(values, quantile))

    residuals = actual[:, None] - predicted
    pooled = np.array(
        [residual_stat(residuals[:, index], quantile) for index, quantile in enumerate(QUANTILES)]
    )
    rows: list[dict[str, float]] = []
    for hour in sorted(np.unique(hours)):
        mask = hours == hour
        n = int(mask.sum())
        weight = n / (n + shrinkage_days) if n + shrinkage_days else 1.0
        local = np.array(
            [
                residual_stat(residuals[mask, index], quantile)
                for index, quantile in enumerate(QUANTILES)
            ]
        )
        correction = weight * local + (1.0 - weight) * pooled
        # Keep the expected-value point forecast stable: a pooled forward mean
        # residual corrects systematic bias without adding noisy hour-specific
        # offsets. The interval tails retain hour-specific calibration.
        correction[1] = pooled[1]
        rows.append(
            {
                "cutoff_hour": int(hour),
                "n": n,
                **{
                    f"q{int(quantile * 100):02d}_correction": float(correction[index])
                    for index, quantile in enumerate(QUANTILES)
                },
            }
        )
    return pd.DataFrame(rows).set_index("cutoff_hour")


def apply_quantile_corrections(
    predicted_remaining: np.ndarray,
    cutoff_hours: np.ndarray,
    corrections: pd.DataFrame,
) -> np.ndarray:
    predicted = np.asarray(predicted_remaining, dtype=float)
    hours = pd.Series(np.asarray(cutoff_hours, dtype=int))
    adjustment = np.column_stack(
        [
            hours.map(corrections[f"q{int(quantile * 100):02d}_correction"]).fillna(0.0)
            for quantile in QUANTILES
        ]
    )
    corrected = np.maximum(0.0, predicted + adjustment)
    point = corrected[:, 1]
    lower = np.minimum(corrected[:, 0], point)
    upper = np.maximum(corrected[:, 2], point)
    return np.column_stack([lower, point, upper])


def order_interval_around_point(predicted_remaining: np.ndarray) -> np.ndarray:
    """Preserve the expected-value point while enforcing non-crossing bounds."""

    predicted = np.maximum(0.0, np.asarray(predicted_remaining, dtype=float))
    point = predicted[:, 1]
    lower = np.minimum(predicted[:, 0], point)
    upper = np.maximum(predicted[:, 2], point)
    return np.column_stack([lower, point, upper])


def predict_boosted(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    features: Sequence[str],
    model_name: str,
    fold_id: int,
    max_iter: int,
    random_state: int,
    calibration_days: int,
    calibration_shrinkage_days: float,
) -> list[pd.DataFrame]:
    if not features:
        raise ValueError(f"No features available for {model_name}")
    if calibration_days < 7:
        raise ValueError("calibration_days must be at least 7")

    train_days = np.array(sorted(pd.to_datetime(train["day"].unique())))
    effective_calibration_days = min(calibration_days, len(train_days) - 28)
    if effective_calibration_days < 7:
        raise ValueError("Need at least 35 training days for nested calibration")
    calibration_start = train_days[-effective_calibration_days]
    core = train.loc[train["day"].lt(calibration_start)].copy()
    calibration = train.loc[train["day"].ge(calibration_start)].copy()

    # Refit completion-curve features using only the inner training block so the
    # calibration residuals reproduce a genuine future prediction.
    inner_curve = fit_completion_curve(core)
    core = add_curve_features(core, inner_curve)
    calibration = add_curve_features(calibration, inner_curve)
    calibration_models = _fit_quantile_models(
        core,
        features=features,
        max_iter=max_iter,
        random_state=random_state,
    )
    calibration_prediction = _predict_remaining_quantiles(
        calibration_models, calibration, features=features
    )
    corrections = fit_quantile_corrections(
        calibration["remaining_arrivals"].to_numpy(dtype=float),
        calibration_prediction,
        calibration["cutoff_hour"].to_numpy(dtype=int),
        shrinkage_days=calibration_shrinkage_days,
    )

    models = _fit_quantile_models(
        train,
        features=features,
        max_iter=max_iter,
        random_state=random_state,
    )
    raw_unordered = _predict_remaining_quantiles(models, test, features=features)
    raw_remaining = order_interval_around_point(raw_unordered)
    calibrated_remaining = apply_quantile_corrections(
        raw_unordered,
        test["cutoff_hour"].to_numpy(dtype=int),
        corrections,
    )
    observed = test["cumulative_arrivals"].to_numpy(dtype=float)
    outputs: list[pd.DataFrame] = []
    for name, remaining in (
        (model_name, raw_remaining),
        (f"{model_name}_calibrated", calibrated_remaining),
    ):
        totals = observed[:, None] + remaining
        outputs.append(
            _prediction_rows(
                test,
                fold_id=fold_id,
                model=name,
                predicted_total=totals[:, 1],
                lower_total=totals[:, 0],
                upper_total=totals[:, 2],
            )
        )
    return outputs


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    def aggregate(group: pd.DataFrame) -> pd.Series:
        error = group["predicted_total"] - group["actual_total"]
        covered = group["actual_total"].between(group["p10_total"], group["p90_total"])
        return pd.Series(
            {
                "n": len(group),
                "mae": mean_absolute_error(group["actual_total"], group["predicted_total"]),
                "rmse": math.sqrt(mean_squared_error(group["actual_total"], group["predicted_total"])),
                "bias": error.mean(),
                "p80_coverage": covered.mean(),
                "mean_interval_width": (group["p90_total"] - group["p10_total"]).mean(),
            }
        )

    by_hour = predictions.groupby(["model", "cutoff_hour"], sort=True).apply(
        aggregate, include_groups=False
    ).reset_index()
    by_hour.insert(1, "scope", "cutoff_hour")
    overall = predictions.groupby("model", sort=True).apply(aggregate, include_groups=False).reset_index()
    overall.insert(1, "scope", "overall")
    overall.insert(2, "cutoff_hour", -1)
    return pd.concat([overall, by_hour], ignore_index=True)


def build_fixed_ensemble(
    predictions: pd.DataFrame,
    *,
    point_model: str = "boosted_calendar_weather",
    interval_model: str = "boosted_state_calibrated",
    output_model: str = "ensemble_calendar_weather_state",
) -> pd.DataFrame:
    """Blend two prespecified point forecasts and retain calibrated state bounds."""

    keys = ["fold", "day", "ds", "cutoff_hour"]
    point_source = predictions.loc[
        predictions["model"].eq(point_model), [*keys, "predicted_total"]
    ].rename(columns={"predicted_total": "point_source_total"})
    interval_source = predictions.loc[predictions["model"].eq(interval_model)].copy()
    if point_source.empty or interval_source.empty:
        return pd.DataFrame(columns=predictions.columns)
    ensemble = interval_source.merge(point_source, on=keys, how="inner", validate="one_to_one")
    if len(ensemble) != len(interval_source):
        raise ValueError("Fixed ensemble sources are not aligned")

    point = 0.5 * (
        ensemble["predicted_total"].to_numpy(dtype=float)
        + ensemble["point_source_total"].to_numpy(dtype=float)
    )
    observed = ensemble["observed_so_far"].to_numpy(dtype=float)
    point = np.maximum(point, observed)
    ensemble["model"] = output_model
    ensemble["predicted_total"] = point
    ensemble["p10_total"] = np.maximum(
        observed, np.minimum(ensemble["p10_total"].to_numpy(dtype=float), point)
    )
    ensemble["p90_total"] = np.maximum(
        point, ensemble["p90_total"].to_numpy(dtype=float)
    )
    ensemble["predicted_remaining"] = np.maximum(0.0, point - observed)
    return ensemble.loc[:, predictions.columns]


def evaluate_readiness(
    predictions: pd.DataFrame,
    *,
    candidate_model: str | None = None,
    baseline_model: str = "prior_update",
    operational_hours: tuple[int, int] = (11, 18),
) -> dict[str, object]:
    """Evaluate fixed retrospective gates without claiming production readiness."""

    def metrics(frame: pd.DataFrame) -> dict[str, float]:
        error = frame["predicted_total"] - frame["actual_total"]
        return {
            "n": int(len(frame)),
            "mae": float(mean_absolute_error(frame["actual_total"], frame["predicted_total"])),
            "bias": float(error.mean()),
            "p80_coverage": float(
                frame["actual_total"].between(frame["p10_total"], frame["p90_total"]).mean()
            ),
        }

    if candidate_model is None:
        candidate_model = (
            "ensemble_calendar_weather_state"
            if predictions["model"].eq("ensemble_calendar_weather_state").any()
            else "boosted_progress_calibrated"
        )
    candidate = predictions.loc[predictions["model"].eq(candidate_model)].copy()
    baseline = predictions.loc[predictions["model"].eq(baseline_model)].copy()
    if candidate.empty or baseline.empty:
        raise ValueError(f"Readiness models not found: {candidate_model}, {baseline_model}")
    start_hour, end_hour = operational_hours
    candidate_window = candidate.loc[candidate["cutoff_hour"].between(start_hour, end_hour)]
    baseline_window = baseline.loc[baseline["cutoff_hour"].between(start_hour, end_hour)]
    candidate_overall_metrics = metrics(candidate)
    baseline_overall_metrics = metrics(baseline)
    candidate_window_metrics = metrics(candidate_window)
    baseline_window_metrics = metrics(baseline_window)
    hour_bias = (
        candidate_window.assign(
            error=candidate_window["predicted_total"] - candidate_window["actual_total"]
        )
        .groupby("cutoff_hour")["error"]
        .mean()
    )
    max_abs_hour_bias = float(hour_bias.abs().max())
    overall_improvement = 1.0 - (
        candidate_overall_metrics["mae"] / baseline_overall_metrics["mae"]
    )
    window_improvement = 1.0 - (
        candidate_window_metrics["mae"] / baseline_window_metrics["mae"]
    )
    invariant_pass = bool(
        candidate["predicted_total"].ge(candidate["observed_so_far"]).all()
        and candidate["p10_total"].le(candidate["predicted_total"]).all()
        and candidate["predicted_total"].le(candidate["p90_total"]).all()
    )
    gates = {
        "overall_mae_improvement_at_least_5pct": bool(overall_improvement >= 0.05),
        "operational_mae_improvement_at_least_5pct": bool(window_improvement >= 0.05),
        "absolute_overall_bias_at_most_2": bool(abs(candidate_overall_metrics["bias"]) <= 2.0),
        "max_operational_hour_bias_at_most_3": bool(max_abs_hour_bias <= 3.0),
        "overall_p80_coverage_between_75_and_85pct": bool(
            0.75 <= candidate_overall_metrics["p80_coverage"] <= 0.85
        ),
        "operational_p80_coverage_between_75_and_85pct": bool(
            0.75 <= candidate_window_metrics["p80_coverage"] <= 0.85
        ),
        "forecast_and_interval_invariants": invariant_pass,
    }
    return {
        "candidate_model": candidate_model,
        "baseline_model": baseline_model,
        "operational_hours": [start_hour, end_hour],
        "candidate_overall": candidate_overall_metrics,
        "baseline_overall": baseline_overall_metrics,
        "candidate_operational": candidate_window_metrics,
        "baseline_operational": baseline_window_metrics,
        "overall_mae_improvement_fraction": float(overall_improvement),
        "operational_mae_improvement_fraction": float(window_improvement),
        "max_absolute_operational_hour_bias": max_abs_hour_bias,
        "retrospective_gates": gates,
        "retrospective_ready": bool(all(gates.values())),
        "production_ready": False,
        "production_blockers": [
            "At least 28 complete prospective shadow days are required (56 preferred).",
            "Versioned fit/predict artifacts, freshness monitoring, fallback, and runbook are required.",
            "An explicit go/no-go review is required before publishing forecasts.",
        ],
    }


def run_backtest(
    snapshots: pd.DataFrame,
    *,
    cutoff_hours: Iterable[int],
    n_folds: int,
    test_days: int,
    min_train_days: int,
    max_iter: int,
    random_state: int,
    calibration_days: int = 56,
    calibration_shrinkage_days: float = 28.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoff_hours = sorted(set(int(hour) for hour in cutoff_hours))
    invalid = [hour for hour in cutoff_hours if not 0 <= hour <= 23]
    if invalid:
        raise ValueError(f"cutoff hours must be in 0..23: {invalid}")

    eligible = snapshots.loc[snapshots["cutoff_hour"].astype(int).isin(cutoff_hours)].copy()
    folds = build_expanding_folds(
        eligible, n_folds=n_folds, test_days=test_days, min_train_days=min_train_days
    )
    outputs: list[pd.DataFrame] = []
    feature_records: list[dict[str, object]] = []

    for fold_id, train_mask, test_mask in folds:
        train = eligible.loc[train_mask].copy().reset_index(drop=True)
        test = eligible.loc[test_mask].copy().reset_index(drop=True)
        curve = fit_completion_curve(train)
        train = add_curve_features(train, curve)
        test = add_curve_features(test, curve)

        curve_prediction = predict_completion_curve(test, curve, fold_id=fold_id)
        outputs.append(curve_prediction)
        update = fit_prior_update(train)
        outputs.append(predict_prior_update(test, update, curve_prediction, fold_id=fold_id))

        for model_name, features in feature_sets(train).items():
            model_outputs = predict_boosted(
                    train,
                    test,
                    features=features,
                    model_name=model_name,
                    fold_id=fold_id,
                    max_iter=max_iter,
                    random_state=random_state + fold_id,
                    calibration_days=calibration_days,
                    calibration_shrinkage_days=calibration_shrinkage_days,
                )
            outputs.extend(model_outputs)
            feature_records.extend(
                {"model": output["model"].iat[0], "feature": feature, "fold": fold_id}
                for output in model_outputs
                for feature in features
            )

    predictions = pd.concat(outputs, ignore_index=True)
    ensemble = build_fixed_ensemble(predictions)
    if not ensemble.empty:
        predictions = pd.concat([predictions, ensemble], ignore_index=True)
    summary = summarize_predictions(predictions)
    features = pd.DataFrame(feature_records).drop_duplicates().sort_values(["model", "feature", "fold"])
    return predictions, summary, features


def parse_cutoff_hours(raw: str) -> list[int]:
    if "-" in raw and "," not in raw:
        start, end = (int(value.strip()) for value in raw.split("-", maxsplit=1))
        return list(range(start, end + 1))
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-csv", default=FLOW_URL)
    parser.add_argument("--weather-csv")
    parser.add_argument("--calendar-context", choices=("basic", "rich"), default="rich")
    parser.add_argument("--cutoff-hours", default="6-22")
    parser.add_argument("--n-folds", type=int, default=6)
    parser.add_argument("--test-days", type=int, default=28)
    parser.add_argument("--min-train-days", type=int, default=365)
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--calibration-days", type=int, default=56)
    parser.add_argument("--calibration-shrinkage-days", type=float, default=28.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("validation/intraday-day-completion")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    flow = load_hourly_flow(args.flow_csv)
    weather = build_weather_features(args.weather_csv) if args.weather_csv else None
    snapshots = build_snapshots(flow, calendar_mode=args.calendar_context, weather=weather)

    complete_days = int(snapshots["day"].nunique())
    source_days = int(flow["day"].nunique())
    source_start = pd.Timestamp(flow["ds"].min())
    source_end = pd.Timestamp(flow["ds"].max())
    print(
        f"Loaded {source_days} source days ({source_start} to {source_end}); "
        f"{complete_days} complete Montreal days are eligible",
        flush=True,
    )
    predictions, summary, features = run_backtest(
        snapshots,
        cutoff_hours=parse_cutoff_hours(args.cutoff_hours),
        n_folds=args.n_folds,
        test_days=args.test_days,
        min_train_days=args.min_train_days,
        max_iter=args.max_iter,
        random_state=args.random_state,
        calibration_days=args.calibration_days,
        calibration_shrinkage_days=args.calibration_shrinkage_days,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    features.to_csv(args.output_dir / "feature_sets.csv", index=False)
    readiness = evaluate_readiness(predictions)
    (args.output_dir / "readiness.json").write_text(json.dumps(readiness, indent=2) + "\n")
    best_by_hour = (
        summary.loc[summary["scope"].eq("cutoff_hour")]
        .sort_values(["cutoff_hour", "mae", "model"])
        .groupby("cutoff_hour", as_index=False)
        .first()
    )
    best_by_hour.to_csv(args.output_dir / "best_by_hour.csv", index=False)
    config = {
        "flow_csv": str(args.flow_csv),
        "weather_csv": str(args.weather_csv) if args.weather_csv else None,
        "calendar_context": args.calendar_context,
        "cutoff_hours": parse_cutoff_hours(args.cutoff_hours),
        "n_folds": args.n_folds,
        "test_days": args.test_days,
        "min_train_days": args.min_train_days,
        "max_iter": args.max_iter,
        "calibration_days": args.calibration_days,
        "calibration_shrinkage_days": args.calibration_shrinkage_days,
        "source_days": source_days,
        "complete_days": complete_days,
        "source_start": source_start.isoformat(),
        "source_end": source_end.isoformat(),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    overall = summary.loc[summary["scope"].eq("overall")].sort_values("mae")
    print("\nOverall results across selected cutoff hours:", flush=True)
    print(overall.to_string(index=False), flush=True)
    print(f"\nWrote validation artifacts to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
