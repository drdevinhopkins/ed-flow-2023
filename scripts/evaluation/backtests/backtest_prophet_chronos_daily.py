#!/usr/bin/env python3
"""Leakage-safe Prophet covariates and fixed blending for daily ED arrivals.

Experimental only. Controls exclude weather: retrospective observed weather is not
a substitute for archived forecast-time weather. No production routes are changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "daily_visits"
COMPONENTS = ["prophet_trend", "prophet_weekly", "prophet_yearly", "prophet_calendar"]
REGRESSORS = ["is_qc_holiday", "is_major_jewish_holiday",
              "is_major_jewish_holiday_eve", "is_rebound_after_long_closure"]
ANCHOR = pd.Timestamp("2000-01-03")


def calendar(frame):
    from backtest_daily_holiday_targeted import scenario_features
    return scenario_features(frame, "calendar_closure")


class ProphetVintages:
    """One fit per origin; every prediction date is strictly after its fit cutoff.

    Missing targets are omitted, never imputed. Unlike the Chronos context, Prophet
    can use verified observations separated by gaps. Historical covariates use
    weekly forecast vintages rather than fitted values from the final model.
    """
    def __init__(self, daily, train_days=1095, min_train=365, model_factory=None):
        self.daily = daily[["ds", TARGET]].copy()
        self.train_days = train_days
        self.min_train = min_train
        self.model_factory = model_factory
        self.cache = {}
        self.audit = []

    def predict(self, origin, horizon=7):
        origin = pd.Timestamp(origin)
        key = (origin, horizon)
        if key in self.cache:
            return self.cache[key].copy()
        train = self.daily.loc[
            self.daily.ds.between(origin - pd.Timedelta(days=self.train_days - 1), origin)
            & self.daily[TARGET].notna()
        ].copy()
        if len(train) < self.min_train:
            raise ValueError(f"Prophet origin {origin.date()}: only {len(train)} verified training days")
        yearly = (train.ds.max() - train.ds.min()).days >= 730
        if self.model_factory is None:
            from prophet import Prophet
            factory = Prophet
        else:
            factory = self.model_factory
        model = factory(yearly_seasonality=yearly, weekly_seasonality=True,
                        daily_seasonality=False, seasonality_mode="additive",
                        changepoint_prior_scale=0.05, uncertainty_samples=0)
        train = calendar(train).rename(columns={TARGET: "y"})
        for column in REGRESSORS:
            model.add_regressor(column, prior_scale=0.1, standardize=False)
        model.fit(train[["ds", "y", *REGRESSORS]], seed=42)
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        prediction = model.predict(calendar(pd.DataFrame({"ds": dates})))
        out = pd.DataFrame({"ds": dates, "prophet_origin": origin})
        for name, source in [("yhat", "yhat"), ("trend", "trend"),
                             ("weekly", "weekly"), ("yearly", "yearly"),
                             ("calendar", "extra_regressors_additive")]:
            out[f"prophet_{name}"] = prediction[source].to_numpy() if source in prediction else 0.0
        if not np.isfinite(out[["prophet_yhat", *COMPONENTS]].to_numpy()).all():
            raise ValueError("Non-finite Prophet features")
        self.audit.append(dict(origin=origin, train_start=train.ds.min(), train_end=train.ds.max(),
                               training_days=len(train), yearly_enabled=yearly, horizon=horizon))
        self.cache[key] = out
        return out.copy()

    def historical(self, dates):
        """Generate D+1..D+7 forecasts from the preceding fixed weekly origin."""
        dates = pd.DatetimeIndex(dates)
        origins = ANCHOR + pd.to_timedelta(((dates - ANCHOR).days - 1) // 7 * 7, unit="D")
        parts = [self.predict(origin) for origin in origins.unique()]
        out = pd.DataFrame({"ds": dates}).merge(pd.concat(parts), on="ds", validate="one_to_one")
        if len(out) != len(dates) or not (out.prophet_origin < out.ds).all():
            raise ValueError("Historical Prophet features must be strictly out of sample")
        return out


def prepare_daily(daily):
    out = daily[["ds", TARGET]].copy()
    out["ds"] = pd.to_datetime(out.ds).dt.normalize()
    if out.ds.duplicated().any() or out.ds.isna().any():
        raise ValueError("Daily dates must be unique and valid")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="raise")
    valid = out[TARGET].dropna()
    if not np.isfinite(valid).all() or valid.lt(0).any():
        raise ValueError("Daily targets must be finite nonnegative counts or missing")
    return out.set_index("ds").sort_index().asfreq("D").reset_index()


def history_at(daily, cutoff, context_days, min_history):
    available = daily.loc[daily.ds.le(cutoff)].copy()
    invalid = np.flatnonzero(available[TARGET].isna().to_numpy())
    start = int(invalid[-1] + 1) if len(invalid) else 0
    history = available.iloc[start:].tail(context_days)
    if len(history) < min_history or history.empty or history.ds.iloc[-1] != cutoff:
        raise ValueError("Insufficient contiguous verified Chronos history")
    return history.reset_index(drop=True)


def select_cutoffs(daily, horizon, context_days, min_history, min_train, train_days, num_cutoffs):
    """Date-only season-balanced selection; no selection on target magnitude/error."""
    rows = []
    train_counts = daily.set_index("ds")[TARGET].notna().rolling(train_days, min_periods=1).sum()
    all_origins = ANCHOR + pd.to_timedelta(((daily.ds - ANCHOR).dt.days - 1) // 7 * 7, unit="D")
    feature_ready = pd.Series(train_counts.reindex(all_origins).fillna(0).to_numpy() >= min_train,
                              index=daily.ds)
    targets = daily.set_index("ds")[TARGET]
    for cutoff in daily.ds:
        dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=horizon)
        actual = targets.reindex(dates)
        if actual.isna().any():
            continue
        try:
            history = history_at(daily, cutoff, context_days, min_history)
        except ValueError:
            continue
        # Trim before Prophet's first eligible weekly vintage, identically for all scenarios.
        eligible = feature_ready.reindex(history.ds).to_numpy()
        invalid = np.flatnonzero(~np.asarray(eligible))
        start = int(invalid[-1] + 1) if len(invalid) else 0
        history = history.iloc[start:]
        if len(history) < min_history:
            continue
        rows.append(dict(cutoff=cutoff, history_start=history.ds.min(), history_days=len(history),
                         season=["winter", "spring", "summer", "fall"][(cutoff.month % 12) // 3]))
    if not rows:
        raise ValueError("No eligible cutoffs; inspect missing days and training requirements")
    eligible = pd.DataFrame(rows)
    # Round-robin seasons, spread evenly within each season; overlapping horizons excluded.
    pools = []
    for _, group in eligible.groupby("season", sort=True):
        indices = list(dict.fromkeys(np.linspace(0, len(group)-1, max(1, int(np.ceil(num_cutoffs/4)))).astype(int)))
        indices += [i for i in range(len(group)) if i not in indices]
        pools.append(group.iloc[indices])
    selected = []
    while len(selected) < num_cutoffs and any(len(p) for p in pools):
        for i, pool in enumerate(pools):
            if pool.empty or len(selected) >= num_cutoffs:
                continue
            row = pool.iloc[0]
            pools[i] = pool.iloc[1:]
            if all(abs((row.cutoff - old["cutoff"]).days) >= horizon for old in selected):
                selected.append(row.to_dict())
    if not selected:
        raise ValueError("No non-overlapping cutoffs")
    return pd.DataFrame(selected).sort_values("cutoff").reset_index(drop=True)


def chronos_predict(pipeline, history, future, columns, horizon):
    history = history[["ds", TARGET, *columns]].copy()
    history["id"] = "jgh_daily"
    kwargs = dict(prediction_length=horizon, id_column="id", timestamp_column="ds",
                  target=[TARGET], quantile_levels=[0.5], context_length=len(history))
    if columns:
        future = future[["ds", *columns]].copy()
        future["id"] = "jgh_daily"
        kwargs["future_df"] = future
    result = pipeline.predict_df(history, **kwargs)
    if result.ds.duplicated().any() or len(result) != horizon:
        raise ValueError("Unexpected Chronos forecast dates")
    aligned = result.set_index("ds").reindex(future.ds)["predictions"].to_numpy()
    if not np.isfinite(aligned).all():
        raise ValueError("Incomplete or non-finite Chronos predictions")
    return aligned


def summarize(detail, groups):
    table = detail.groupby([*groups, "scenario"], as_index=False).agg(
        n=("error", "size"), mae=("abs_error", "mean"), mse=("squared_error", "mean"),
        bias=("error", "mean"), error_sum=("abs_error", "sum"), actual_sum=("actual", "sum"))
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("error_sum") / table.pop("actual_sum").replace(0, np.nan)
    table["comparison"] = "all"  # permits an overall join without a special case
    keys = groups or ["comparison"]
    for control in ["baseline", "calendar_closure"]:
        ref = table.loc[table.scenario.eq(control), [*keys, "mae"]].rename(columns={"mae": f"{control}_mae"})
        table = table.merge(ref, on=keys, validate="many_to_one")
        table[f"improvement_vs_{control}_pct"] = 100 * (1 - table.mae / table[f"{control}_mae"].replace(0, np.nan))
    return table.drop(columns="comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-url", help="Hourly source; defaults to shared production input")
    parser.add_argument("--daily-csv", type=Path, help="Verified ds,daily_visits snapshot; never raw cumulative totals")
    parser.add_argument("--num-cutoffs", type=int, default=24)
    parser.add_argument("--horizon-days", type=int, default=7)
    parser.add_argument("--context-days", type=int, default=365)
    parser.add_argument("--min-history-days", type=int, default=120)
    parser.add_argument("--prophet-train-days", type=int, default=1095)
    parser.add_argument("--prophet-min-train", type=int, default=365)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-prophet"))
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not (1 <= args.horizon_days <= 7 and args.num_cutoffs > 0
            and 28 <= args.min_history_days <= args.context_days <= 8192
            and 30 <= args.prophet_min_train <= args.prophet_train_days):
        parser.error("Invalid horizon, cutoff count, or training/context limits")
    if args.daily_csv:
        daily = prepare_daily(pd.read_csv(args.daily_csv))
    else:
        from backtest_holiday_features import FLOW_URL, load_daily_visits
        daily = prepare_daily(load_daily_visits(args.flow_url or FLOW_URL))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "verified_daily.csv", index=False)
    cutoffs = select_cutoffs(daily, args.horizon_days, args.context_days, args.min_history_days,
                             args.prophet_min_train, args.prophet_train_days, args.num_cutoffs)
    cutoffs.to_csv(output / "cutoffs.csv", index=False)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(target_sha256=hashlib.sha256(daily.to_csv(index=False).encode()).hexdigest(),
                    comparison="calendar controls only; weather excluded", blend_prophet_weight=0.25,
                    actual_cutoffs=len(cutoffs), status="prepared")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(cutoffs.to_string(index=False), flush=True)
    if args.prepare_only:
        return
    import torch
    from chronos import Chronos2Pipeline
    torch.set_num_threads(4)
    pipeline = Chronos2Pipeline.from_pretrained(args.model_id, device_map="cuda" if torch.cuda.is_available() else "cpu")
    metadata["package_versions"] = {name: importlib.metadata.version(name) for name in ["prophet", "chronos-forecasting", "torch", "pandas"]}
    metadata["model_revision"] = getattr(pipeline.model.config, "_commit_hash", None)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    vintages = ProphetVintages(daily, args.prophet_train_days, args.prophet_min_train)
    details = []
    for row in cutoffs.itertuples(index=False):
        print(f"Forecasting {row.cutoff.date()}", flush=True)
        history = daily.loc[daily.ds.between(row.history_start, row.cutoff)].copy()
        past = vintages.historical(history.ds)
        future = vintages.predict(row.cutoff, args.horizon_days)
        history = calendar(history).merge(past, on="ds", validate="one_to_one")
        future = calendar(future)
        cal_columns = [c for c in calendar(daily.head(1)).columns if c not in {"ds", TARGET}]
        forecasts = {
            "baseline": chronos_predict(pipeline, history, future, [], args.horizon_days),
            "calendar_closure": chronos_predict(pipeline, history, future, cal_columns, args.horizon_days),
            "prophet_components": chronos_predict(pipeline, history, future, [*cal_columns, *COMPONENTS], args.horizon_days),
            "prophet_yhat": chronos_predict(pipeline, history, future, [*cal_columns, "prophet_yhat"], args.horizon_days),
            "prophet_only": future.prophet_yhat.to_numpy(),
        }
        forecasts["blend_25pct_prophet"] = 0.75 * forecasts["calendar_closure"] + 0.25 * forecasts["prophet_only"]
        actual = daily.set_index("ds")[TARGET].reindex(future.ds).to_numpy()
        for scenario, values in forecasts.items():
            frame = pd.DataFrame(dict(ds=future.ds, actual=actual, prediction=values))
            frame["cutoff"] = row.cutoff
            frame["scenario"] = scenario
            frame["history_days"] = row.history_days
            frame["context_group"] = "120plus_days" if row.history_days >= 120 else "28to119_days"
            frame["horizon_day"] = np.arange(1, args.horizon_days + 1)
            frame["season"] = frame.ds.dt.month.map(lambda m: ["winter", "spring", "summer", "fall"][(m % 12)//3])
            frame["holiday"] = future[["is_qc_holiday", "is_major_jewish_holiday", "is_ramq_holiday"]].max(axis=1).to_numpy()
            frame["error"] = frame.prediction - frame.actual
            frame["abs_error"] = frame.error.abs()
            frame["squared_error"] = frame.error ** 2
            details.append(frame)
        detail = pd.concat(details, ignore_index=True)
        detail.to_csv(output / "detail.csv", index=False)
        pd.DataFrame(vintages.audit).to_csv(output / "prophet_vintages.csv", index=False)
        past.assign(cutoff=row.cutoff).to_csv(output / f"features_{row.cutoff.date()}.csv", index=False)
        future.to_csv(output / f"future_{row.cutoff.date()}.csv", index=False)
    for filename, groups in [("summary", []), ("by_season", ["season"]),
                              ("by_horizon", ["horizon_day"]), ("by_holiday", ["holiday"]),
                              ("by_context", ["context_group"]), ("by_cutoff", ["cutoff"])]:
        table = summarize(detail, groups)
        table.to_csv(output / f"{filename}.csv", index=False)
        if filename == "summary":
            print(table.to_string(index=False), flush=True)
    metadata["status"] = "complete"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
