#!/usr/bin/env python3
"""Summarize how much prospective weather evidence remains before evaluation readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN_PROSPECTIVE_SPAN_DAYS = 56
MIN_ISSUE_DATES = 28
MIN_ACTIVE_ROWS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.summary_csv)
    required = {"target_name", "horizon_band", "n", "n_issue_dates", "prospective_span_days", "promotion_status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Summary missing required columns: {sorted(missing)}")

    result = frame[[
        "target_name", "horizon_band", "n", "n_issue_dates", "prospective_span_days",
        "directional_criteria_met", "promotion_evidence_ready", "promotion_status"
    ]].copy()
    result["min_active_rows"] = MIN_ACTIVE_ROWS
    result["min_issue_dates"] = MIN_ISSUE_DATES
    result["min_span_days"] = MIN_PROSPECTIVE_SPAN_DAYS
    result["active_rows_remaining"] = (MIN_ACTIVE_ROWS - result["n"]).clip(lower=0)
    result["issue_dates_remaining"] = (MIN_ISSUE_DATES - result["n_issue_dates"]).clip(lower=0)
    result["span_days_remaining"] = (MIN_PROSPECTIVE_SPAN_DAYS - result["prospective_span_days"]).clip(lower=0)
    result["readiness_fraction_rows"] = (result["n"] / MIN_ACTIVE_ROWS).clip(upper=1)
    result["readiness_fraction_dates"] = (result["n_issue_dates"] / MIN_ISSUE_DATES).clip(upper=1)
    result["readiness_fraction_span"] = (result["prospective_span_days"] / MIN_PROSPECTIVE_SPAN_DAYS).clip(upper=1)
    result["overall_readiness_fraction"] = result[[
        "readiness_fraction_rows", "readiness_fraction_dates", "readiness_fraction_span"
    ]].min(axis=1)

    output = args.output_dir / "weather-route-evidence-gaps.csv"
    result.sort_values(["target_name", "horizon_band"]).to_csv(output, index=False)
    print("Prospective weather evidence remaining before evaluation readiness:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
