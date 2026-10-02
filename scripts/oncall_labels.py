"""Activation labels are observations: missing hours remain unknown."""
from __future__ import annotations

import pandas as pd


def merge_activation_labels(hourly: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.rename(columns={"oncall-used-for-busy": "oncall_active"}).copy()
    labels["ds"] = pd.to_datetime(labels["ds"], errors="coerce").dt.floor("h")
    labels = labels.dropna(subset=["ds"])[["ds", "oncall_active"]]
    labels["oncall_active"] = pd.to_numeric(labels["oncall_active"], errors="coerce")
    if not labels["oncall_active"].dropna().isin([0, 1]).all():
        raise ValueError("Activation labels must be explicit 0 or 1.")
    if labels.groupby("ds")["oncall_active"].nunique().gt(1).any():
        raise ValueError("Conflicting activation labels for the same hour.")
    labels = labels.drop_duplicates("ds", keep="last")
    return hourly.merge(labels, on="ds", how="left", validate="many_to_one")


def add_activation_targets(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Require every clock hour in the outcome window to have a label."""
    out = df.copy()
    active = out.set_index("ds")["oncall_active"]
    for horizon in horizons:
        future = pd.concat([
            pd.Series(active.reindex(out["ds"] + pd.Timedelta(hours=step)).to_numpy(),
                      index=out.index)
            for step in range(1, horizon + 1)
        ], axis=1)
        out[f"oncall_within_{horizon}h"] = future.max(axis=1, skipna=False)
    return out
