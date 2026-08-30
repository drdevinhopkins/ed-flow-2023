#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from hourly_long_term_features import (
    annual_target_memory_frame,
    build_long_term_feature_frame,
    scenario_columns,
)


def synthetic_flow() -> pd.DataFrame:
    ds = pd.date_range("2021-01-01", "2026-03-15", freq="h")
    year_component = (ds.year - 2021).astype(float) * 10.0
    seasonal = np.sin(2 * np.pi * (ds.dayofyear.to_numpy() - 1) / 365.25)
    values = year_component + seasonal
    return pd.DataFrame({"ds": ds, "Total_TBS": values})


def test_annual_lag_matches_same_calendar_hour() -> None:
    flow = synthetic_flow()
    ts = pd.Series(pd.to_datetime(["2025-02-10 13:00:00"]))
    frame = annual_target_memory_frame(flow, ts, ["Total_TBS"])
    lookup = flow.set_index("ds")["Total_TBS"]
    assert np.isclose(
        frame.loc[0, "Total_TBS__lag_1y"],
        lookup.loc[pd.Timestamp("2024-02-10 13:00:00")],
    )
    assert np.isclose(
        frame.loc[0, "Total_TBS__lag_2y"],
        lookup.loc[pd.Timestamp("2023-02-10 13:00:00")],
    )
    assert frame.loc[0, "Total_TBS__annual_growth_recent"] > 9.0


def test_future_secular_features_are_frozen_at_cutoff() -> None:
    flow = synthetic_flow()
    cutoff = pd.Timestamp("2025-12-01 00:00:00")
    timestamps = pd.Series(
        pd.date_range(cutoff - pd.Timedelta(hours=3), periods=28, freq="h")
    )
    frame = build_long_term_feature_frame(
        flow, timestamps, ["Total_TBS"], cutoff=cutoff
    )
    future = frame.loc[frame["ds"] > cutoff]
    cols = [
        c
        for c in future.columns
        if "__level_" in c
        or "__growth_90d_yoy" in c
        or "__growth_365d_yoy" in c
    ]
    assert cols
    for column in cols:
        assert future[column].nunique(dropna=False) == 1


def test_future_features_do_not_change_if_post_cutoff_actuals_change() -> None:
    flow = synthetic_flow()
    cutoff = pd.Timestamp("2025-12-01 00:00:00")
    timestamps = pd.Series(pd.date_range(cutoff + pd.Timedelta(hours=1), periods=24, freq="h"))
    first = build_long_term_feature_frame(flow, timestamps, ["Total_TBS"], cutoff=cutoff)

    altered = flow.copy()
    altered.loc[altered["ds"] > cutoff, "Total_TBS"] += 10000.0
    second = build_long_term_feature_frame(altered, timestamps, ["Total_TBS"], cutoff=cutoff)

    cols = scenario_columns("secular_growth", ["Total_TBS"])
    np.testing.assert_allclose(first[cols].to_numpy(), second[cols].to_numpy())


def test_scenario_columns_separate_seasonality_and_growth() -> None:
    targets = ["Total_TBS"]
    annual = scenario_columns("annual_memory", targets)
    growth = scenario_columns("secular_growth", targets)
    combined = scenario_columns("annual_plus_growth", targets)
    assert "Total_TBS__lag_1y" in annual
    assert "Total_TBS__growth_365d_yoy" in growth
    assert set(annual).issubset(combined)
    assert set(growth).issubset(combined)


if __name__ == "__main__":
    test_annual_lag_matches_same_calendar_hour()
    test_future_secular_features_are_frozen_at_cutoff()
    test_future_features_do_not_change_if_post_cutoff_actuals_change()
    test_scenario_columns_separate_seasonality_and_growth()
    print("hourly long-term feature tests passed")
