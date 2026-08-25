#!/usr/bin/env python3
"""Aggregate parallel feature-ablation results for candidate operational targets."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CANDIDATE_TARGETS = (
    "Hourly_Inflow_Total",
    "Hourly_Inflow_Stretcher",
    "Hourly_Inflow_Ambulances",
    "AdmissionRequests_New",
    "Workup_Delay_Burden",
)
EXPECTED_BANDS = ("h01_04", "h05_08", "h09_12", "h13_24")

def metrics(detail, group_cols):
    table = detail.groupby(group_cols, as_index=False).agg(
        n=("abs_error","size"), mae=("abs_error","mean"), mse=("squared_error","mean"),
        mean_error=("error","mean"), abs_error_sum=("abs_error","sum"), abs_actual_sum=("abs_actual","sum"))
    table["rmse"] = np.sqrt(table.pop("mse"))
    table["wape"] = table.pop("abs_error_sum") / table.pop("abs_actual_sum").replace(0, np.nan)
    return table

def add_baseline(table, keys):
    baseline = table.loc[table.scenario.eq("baseline"), [*keys,"mae"]].rename(columns={"mae":"baseline_mae"})
    out = table.merge(baseline, on=keys, how="left")
    out["mae_improvement"] = out["baseline_mae"] - out["mae"]
    out["mae_improvement_pct"] = out["mae_improvement"] / out["baseline_mae"].replace(0,np.nan) * 100
    out["beats_baseline"] = out["mae_improvement"].gt(0)
    return out

def winners(table, keys):
    idx = table.groupby(keys, observed=True)["mae"].idxmin()
    return table.loc[idx].sort_values(keys).reset_index(drop=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--expected-cutoffs",type=int,default=8); a=p.parse_args()
    paths=sorted(a.input_dir.rglob("detail-*.csv"))
    if len(paths)!=a.expected_cutoffs: raise RuntimeError(f"Expected {a.expected_cutoffs} detail files, found {len(paths)}")
    detail=pd.concat((pd.read_csv(x) for x in paths),ignore_index=True)
    if set(detail.target_name)!=set(CANDIDATE_TARGETS): raise RuntimeError(f"Unexpected targets: {sorted(detail.target_name.unique())}")
    if detail.cutoff.nunique()!=a.expected_cutoffs: raise RuntimeError("Unexpected cutoff count")
    if set(detail.horizon_band)!=set(EXPECTED_BANDS): raise RuntimeError("Unexpected horizon bands")
    overall=add_baseline(metrics(detail,["target_name","scenario","family"]),["target_name"]).sort_values(["target_name","mae"])
    by_band=add_baseline(metrics(detail,["target_name","horizon_band","scenario","family"]),["target_name","horizon_band"]).sort_values(["target_name","horizon_band","mae"])
    by_hour=add_baseline(metrics(detail,["target_name","horizon_hour","scenario","family"]),["target_name","horizon_hour"]).sort_values(["target_name","horizon_hour","mae"])
    band_winners=winners(by_band,["target_name","horizon_band"])
    safe=by_band.loc[~by_band.scenario.isin({"baseline","weather_raw","weather_raw_plus_snow"}) & by_band.beats_baseline].copy()
    safe_winners=winners(safe,["target_name","horizon_band"]) if not safe.empty else safe
    a.output_dir.mkdir(parents=True,exist_ok=True)
    detail.to_csv(a.output_dir/"detail.csv",index=False); overall.to_csv(a.output_dir/"summary.csv",index=False); by_band.to_csv(a.output_dir/"by_horizon_band.csv",index=False); by_hour.to_csv(a.output_dir/"by_horizon_hour.csv",index=False); winners(overall,["target_name"]).to_csv(a.output_dir/"winners_by_target.csv",index=False); band_winners.to_csv(a.output_dir/"winners_by_target_horizon_band.csv",index=False); safe_winners.to_csv(a.output_dir/"safe_nonweather_winners_by_target_horizon_band.csv",index=False)
    print(band_winners[["target_name","horizon_band","scenario","mae","baseline_mae","mae_improvement_pct","beats_baseline"]].to_string(index=False))
if __name__=="__main__": main()
