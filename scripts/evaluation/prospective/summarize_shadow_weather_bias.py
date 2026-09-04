#!/usr/bin/env python3
"""Summarize signed-error bias for prospective shadow-weather routes.

This is diagnostic-only. It quantifies whether a weather route introduces systematic
under- or over-prediction even when MAE improves. Rows are restricted to forecasts where
weather routing was actually active.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAX_ABS_BIAS_WORSENING_FRACTION_OF_BASELINE_MAE = 0.10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detail_csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/prospective-weather/latest-score"),
    )
    return parser.parse_args()


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    required = {
        "target_name",
        "horizon_band",
        "weather_route_active",
        "baseline_error",
        "weather_error",
        "baseline_absolute_error",
    }
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Detail file missing required columns: {sorted(missing)}")

    active = detail.loc[detail["weather_route_active"].astype(bool)].copy()
    if active.empty:
        return pd.DataFrame()

    group = ["target_name", "horizon_band"]
    out = active.groupby(group, as_index=False).agg(
        n=("weather_error", "size"),
        baseline_bias=("baseline_error", "mean"),
        weather_bias=("weather_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
    )
    out["baseline_abs_bias"] = out["baseline_bias"].abs()
    out["weather_abs_bias"] = out["weather_bias"].abs()
    out["absolute_bias_worsening"] = out["weather_abs_bias"] - out["baseline_abs_bias"]
    tolerance = (
        out["baseline_mae"] * MAX_ABS_BIAS_WORSENING_FRACTION_OF_BASELINE_MAE
    )
    out["bias_worsening_tolerance"] = tolerance
    out["bias_not_materially_worse"] = out["absolute_bias_worsening"] <= tolerance
    out["bias_direction"] = np.select(
        [out["weather_bias"] > 0, out["weather_bias"] < 0],
        ["overpredict", "underpredict"],
        default="neutral",
    )
    return out.sort_values(group)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.read_csv(args.detail_csv)
    summary = summarize(detail)
    path = args.output_dir / "weather-route-bias.csv"
    summary.to_csv(path, index=False)
    if summary.empty:
        print("No matured active weather-route rows; bias summary is empty.")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
