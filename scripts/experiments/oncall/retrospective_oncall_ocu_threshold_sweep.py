from __future__ import annotations

"""Threshold sweep for day-level ocU validation.

Known ocU-positive days support a capture-rate calculation only.  Days without an ocU
marker remain unknown, so ``all_scored_days_flagged`` is alert burden/descriptive
frequency, not a false-positive count and must not be used to claim specificity/PPV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep pressure thresholds against ocU-positive days.")
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--min-threshold", type=int, default=80)
    parser.add_argument("--max-threshold", type=int, default=99)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    daily = pd.read_csv(inp / "oncall_ocu_postcutoff_daily_validation.csv")
    state = pd.read_csv(inp / "oncall_schedule_state_daily.csv")
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    state["date"] = pd.to_datetime(state["date"]).dt.date
    merged = daily.merge(
        state[
            [
                "date",
                "reserve_reassigned_regular_shift_proxy",
                "b2_absent_on_mon_thu",
                "schedule_supply_state",
            ]
        ],
        on="date",
        how="left",
    )

    positive = merged["day_label_status"].eq("ocu_positive_used_day")
    reassigned = merged["reserve_reassigned_regular_shift_proxy"].fillna(False).astype(bool)
    positive_reserve_not_reassigned = positive & ~reassigned
    positive_reassigned = positive & reassigned
    b2_absent = merged["b2_absent_on_mon_thu"].fillna(False).astype(bool)

    rows: list[dict[str, object]] = []
    scores = pd.to_numeric(merged["max_need_pressure_percentile"], errors="coerce")
    for threshold in range(args.min_threshold, args.max_threshold + 1):
        flagged = scores >= threshold
        rows.append(
            {
                "threshold_percentile": threshold,
                "known_ocu_positive_total": int(positive.sum()),
                "known_ocu_positive_captured": int((flagged & positive).sum()),
                "known_ocu_positive_capture_rate": float(flagged[positive].mean()) if positive.any() else np.nan,
                "ocu_positive_reserve_not_reassigned_total": int(positive_reserve_not_reassigned.sum()),
                "ocu_positive_reserve_not_reassigned_captured": int((flagged & positive_reserve_not_reassigned).sum()),
                "ocu_positive_reserve_not_reassigned_capture_rate": float(flagged[positive_reserve_not_reassigned].mean()) if positive_reserve_not_reassigned.any() else np.nan,
                "ocu_positive_reserve_reassigned_total": int(positive_reassigned.sum()),
                "ocu_positive_reserve_reassigned_captured": int((flagged & positive_reassigned).sum()),
                "ocu_positive_reserve_reassigned_capture_rate": float(flagged[positive_reassigned].mean()) if positive_reassigned.any() else np.nan,
                "ocu_positive_b2_absent_total": int((positive & b2_absent).sum()),
                "ocu_positive_b2_absent_captured": int((flagged & positive & b2_absent).sum()),
                "all_scored_days": int(len(merged)),
                "all_scored_days_flagged_descriptive_only": int(flagged.sum()),
                "all_scored_day_flag_fraction_descriptive_only": float(flagged.mean()),
                "flagged_days_per_week_descriptive_only": float(flagged.mean() * 7.0),
                "specificity_estimable": False,
                "ppv_estimable": False,
            }
        )

    result = pd.DataFrame(rows)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "oncall_ocu_threshold_sweep.csv", index=False)

    pareto = result[
        result["known_ocu_positive_captured"].diff().fillna(1).ne(0)
        | result["all_scored_days_flagged_descriptive_only"].diff().fillna(1).ne(0)
    ]
    print(result.to_string(index=False))
    print("\nReminder: unlabeled days are unknown, not negatives; alert burden is descriptive only.")


if __name__ == "__main__":
    main()
