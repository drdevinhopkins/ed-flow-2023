#!/usr/bin/env python3
"""Pre-registered discovery/confirmation check for short-horizon weather policies.

Diagnostic only. The first DISCOVERY_DATES complete issue dates are used to rank policies.
The selected discovery winner is then evaluated only on later complete issue dates. This
prevents searching many policy subsets and treating the same dates as confirmatory evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DISCOVERY_DATES = 28
CONFIRMATION_DATES = 28
MIN_CONFIRMATION_WIN_RATE = 0.55


def _policy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in frame.groupby("policy", dropna=False):
        improvement = pd.to_numeric(group["mae_improvement_pct_vs_baseline"], errors="coerce").dropna()
        vs_current = pd.to_numeric(group["mae_improvement_pct_vs_current_weather_policy"], errors="coerce").dropna()
        if improvement.empty:
            continue
        rows.append({
            "policy": policy,
            "n_issue_dates": int(group["forecast_issue_date"].nunique()),
            "mean_improvement_pct_vs_baseline": float(improvement.mean()),
            "median_improvement_pct_vs_baseline": float(improvement.median()),
            "issue_date_win_rate_vs_baseline": float((improvement > 0).mean()),
            "worst_issue_date_improvement_pct": float(improvement.min()),
            "mean_improvement_pct_vs_current_weather_policy": float(vs_current.mean()) if not vs_current.empty else np.nan,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["mean_improvement_pct_vs_baseline", "issue_date_win_rate_vs_baseline", "worst_issue_date_improvement_pct", "policy"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )


def summarize(by_date: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = by_date.loc[by_date["issue_date_complete"].astype(bool)].copy()
    complete["forecast_issue_date"] = complete["forecast_issue_date"].astype(str)
    dates = sorted(complete["forecast_issue_date"].unique())

    discovery_dates = dates[:DISCOVERY_DATES]
    confirmation_dates = dates[DISCOVERY_DATES:DISCOVERY_DATES + CONFIRMATION_DATES]
    discovery = complete.loc[complete["forecast_issue_date"].isin(discovery_dates)].copy()
    confirmation = complete.loc[complete["forecast_issue_date"].isin(confirmation_dates)].copy()

    discovery_summary = _policy_summary(discovery)
    if discovery_summary.empty:
        status = pd.DataFrame([{
            "complete_issue_dates": len(dates),
            "discovery_issue_dates": len(discovery_dates),
            "confirmation_issue_dates": len(confirmation_dates),
            "selected_policy": "",
            "confirmation_status": "collecting_discovery",
            "confirmation_evidence_ready": False,
        }])
        return discovery_summary, status

    # Baseline is a comparator, never a candidate routing policy.
    candidates = discovery_summary.loc[~discovery_summary["policy"].eq("all_baseline")].copy()
    selected = candidates.iloc[0] if not candidates.empty else discovery_summary.iloc[0]
    selected_policy = str(selected["policy"])

    selected_confirmation = confirmation.loc[confirmation["policy"].eq(selected_policy)].copy()
    confirmation_summary = _policy_summary(selected_confirmation)
    confirm_ready = len(confirmation_dates) >= CONFIRMATION_DATES

    if confirmation_summary.empty:
        mean_vs_baseline = np.nan
        median_vs_baseline = np.nan
        win_rate = np.nan
        mean_vs_current = np.nan
        confirmation_pass = False
    else:
        row = confirmation_summary.iloc[0]
        mean_vs_baseline = float(row["mean_improvement_pct_vs_baseline"])
        median_vs_baseline = float(row["median_improvement_pct_vs_baseline"])
        win_rate = float(row["issue_date_win_rate_vs_baseline"])
        mean_vs_current = float(row["mean_improvement_pct_vs_current_weather_policy"])
        confirmation_pass = (
            mean_vs_baseline > 0
            and median_vs_baseline > 0
            and win_rate >= MIN_CONFIRMATION_WIN_RATE
            and (selected_policy == "current_weather_policy" or (pd.notna(mean_vs_current) and mean_vs_current > 0))
        )

    if len(discovery_dates) < DISCOVERY_DATES:
        label = "collecting_discovery"
    elif not confirm_ready:
        label = "collecting_confirmation"
    elif confirmation_pass:
        label = "confirmed_candidate"
    else:
        label = "confirmation_failed"

    status = pd.DataFrame([{
        "complete_issue_dates": len(dates),
        "discovery_issue_dates": len(discovery_dates),
        "confirmation_issue_dates": len(confirmation_dates),
        "selected_policy": selected_policy,
        "discovery_mean_improvement_pct_vs_baseline": float(selected["mean_improvement_pct_vs_baseline"]),
        "confirmation_mean_improvement_pct_vs_baseline": mean_vs_baseline,
        "confirmation_median_improvement_pct_vs_baseline": median_vs_baseline,
        "confirmation_issue_date_win_rate": win_rate,
        "confirmation_mean_improvement_pct_vs_current_weather_policy": mean_vs_current,
        "confirmation_evidence_ready": confirm_ready,
        "confirmation_pass": bool(confirmation_pass),
        "confirmation_status": label,
        "discovery_dates_remaining": max(0, DISCOVERY_DATES - len(discovery_dates)),
        "confirmation_dates_remaining": max(0, CONFIRMATION_DATES - len(confirmation_dates)),
    }])
    return discovery_summary, status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy_by_date_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.policy_by_date_csv)
    required = {
        "forecast_issue_date", "issue_date_complete", "policy",
        "mae_improvement_pct_vs_baseline", "mae_improvement_pct_vs_current_weather_policy",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing policy-grid columns: {sorted(missing)}")

    discovery, status = summarize(frame)
    if not discovery.empty:
        discovery.to_csv(args.output_dir / "weather-policy-discovery-ranking.csv", index=False)
    status.to_csv(args.output_dir / "weather-policy-confirmation-status.csv", index=False)
    print("Weather policy discovery/confirmation status:")
    print(status.to_string(index=False))


if __name__ == "__main__":
    main()
