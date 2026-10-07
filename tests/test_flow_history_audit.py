import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evaluation"))
from audit_flow_history import audit_history, audit_scoring


def source(start="2026-01-01", days=3):
    return pd.DataFrame({"ds": pd.date_range(start, periods=24 * days, freq="h"),
                         "Inflow_Total": 1.0})


def totals(hourly):
    return hourly.groupby(hourly.ds.dt.normalize())["Inflow_Total"].sum().rename(
        "Daily_Inflow_Total").reset_index()


class HistoryAuditTests(unittest.TestCase):
    def audit(self, hourly, daily=None, now="2026-01-04T12:00:00-05:00"):
        return audit_history(hourly, totals(hourly) if daily is None else daily,
                             now=pd.Timestamp(now))

    def test_single_missing_hour_is_not_a_complete_day(self):
        hourly = source().drop(index=30)
        summary, quality, gaps = self.audit(hourly)
        self.assertEqual(summary["published_ordinary_incomplete_totals"], 1)
        self.assertEqual(summary["contiguous_audit_eligible_previous_days"], 1)
        self.assertEqual(gaps.ds.tolist(), [pd.Timestamp("2026-01-02 06:00")])
        row = quality.iloc[1]
        self.assertTrue(row.daily_sum_matches)
        self.assertFalse(row.audit_eligible)

    def test_whole_missing_day_is_visible_and_breaks_context(self):
        hourly = source()
        daily = totals(hourly)
        hourly = hourly.loc[hourly.ds.dt.day.ne(2)]
        summary, quality, gaps = self.audit(hourly, daily)
        self.assertEqual(len(gaps), 24)
        self.assertEqual(summary["gap_episodes"], 1)
        self.assertEqual(quality.iloc[1].observed_rows, 0)
        self.assertEqual(summary["published_sum_mismatches"], 1)

    def test_duplicate_does_not_cover_missing_slot(self):
        hourly = source().drop(index=30)
        hourly = pd.concat([hourly, hourly.iloc[[29]]], ignore_index=True)
        summary, quality, _ = self.audit(hourly)
        self.assertEqual(summary["duplicate_hourly_rows"], 2)
        self.assertEqual(quality.iloc[1].observed_rows, 24)
        self.assertEqual(quality.iloc[1].unique_hours, 23)
        self.assertFalse(quality.iloc[1].audit_eligible)

    def test_nonfinite_or_negative_inflow_is_invalid(self):
        hourly = source()
        hourly.loc[25, "Inflow_Total"] = np.inf
        hourly.loc[26, "Inflow_Total"] = -1
        summary, quality, _ = self.audit(hourly)
        self.assertEqual(summary["invalid_inflow_rows"], 2)
        self.assertFalse(quality.iloc[1].audit_eligible)

    def test_dst_requires_verification_without_guessing_arrivals(self):
        hourly = source("2026-03-07").drop(index=27)
        summary, quality, gaps = self.audit(hourly, now="2026-03-10T12:00:00-04:00")
        self.assertEqual(summary["missing_slots_on_dst_dates"], 1)
        self.assertEqual(quality.iloc[1].quality_status, "dst_requires_verification")
        self.assertFalse(quality.iloc[1].audit_eligible)
        self.assertEqual(len(gaps), 1)

    def test_current_day_and_leading_partial_are_separate(self):
        hourly = source().iloc[1:60]
        daily = totals(hourly)
        daily = daily.iloc[:-1]
        summary, quality, _ = self.audit(hourly, daily, now="2026-01-03T12:00:00-05:00")
        self.assertEqual(quality.iloc[0].quality_status, "leading_partial")
        self.assertEqual(quality.iloc[-1].quality_status, "current_day_partial")
        self.assertEqual(summary["ordinary_incomplete_previous_days"], 0)
        self.assertEqual(summary["contiguous_audit_eligible_previous_days"], 1)

    def test_first_day_internal_gap_is_not_a_leading_boundary(self):
        hourly = source().drop(index=6)
        summary, quality, _ = self.audit(hourly)
        self.assertEqual(quality.iloc[0].quality_status, "incomplete_or_invalid")
        self.assertEqual(summary["ordinary_incomplete_previous_days"], 1)
        self.assertEqual(summary["ordinary_incomplete_days_in_1095_day_context"], 1)
        self.assertEqual(summary["published_ordinary_incomplete_totals"], 1)

    def test_first_day_invalid_inflow_is_not_a_leading_boundary(self):
        hourly = source()
        hourly.loc[6, "Inflow_Total"] = np.nan
        summary, quality, _ = self.audit(hourly)
        self.assertEqual(quality.iloc[0].quality_status, "incomplete_or_invalid")
        self.assertEqual(summary["ordinary_incomplete_previous_days"], 1)
        self.assertEqual(summary["published_ordinary_incomplete_totals"], 1)

    def test_daily_mismatch_or_no_hourly_source_is_reported(self):
        hourly = source()
        daily = totals(hourly)
        daily.loc[1, "Daily_Inflow_Total"] = 100
        daily.loc[len(daily)] = [pd.Timestamp("2025-12-31"), 25]
        summary, quality, _ = self.audit(hourly, daily)
        self.assertEqual(summary["published_sum_mismatches"], 2)
        self.assertEqual(summary["daily_dates_without_hourly_source"], 1)
        self.assertFalse(quality.iloc[2].audit_eligible)

    def test_scoring_flags_partial_actual_and_cutoff_safe_baseline(self):
        hourly = source().drop(index=30)
        daily = totals(hourly)
        _, quality, _ = self.audit(hourly)
        detail = pd.DataFrame({"ds": pd.to_datetime(["2026-01-02", "2026-01-09"]),
                               "data_cutoff": pd.to_datetime(["2026-01-01", "2026-01-03"]),
                               "actual": [23, 30]})
        summary, rows = audit_scoring(detail, quality, daily)
        self.assertEqual(summary["scored_actual_rows_requiring_review"], 2)
        self.assertEqual(rows.iloc[1].baseline_dates_requiring_review, "2026-01-02")
        # First row's fallback uses Jan 1 only; the future Jan 2 gap cannot leak in.
        self.assertFalse(rows.iloc[0].baseline_requires_review)
        self.assertFalse(rows.iloc[1].scored_actual_matches_daily)

    def test_scoring_flags_revised_actual_values(self):
        hourly = source()
        daily = totals(hourly)
        _, quality, _ = self.audit(hourly)
        detail = pd.DataFrame({"ds": [pd.Timestamp("2026-01-03")],
                               "data_cutoff": [pd.Timestamp("2026-01-02")],
                               "actual": [22]})
        summary, rows = audit_scoring(detail, quality, daily)
        self.assertEqual(summary["scored_actual_mismatches"], 1)
        self.assertFalse(rows.iloc[0].actual_requires_review)

    def test_invalid_timestamp_format_fails_explicitly(self):
        hourly = source()
        hourly["ds"] = hourly.ds.astype(str)
        hourly.loc[1, "ds"] = "not a timestamp"
        with self.assertRaisesRegex(ValueError, "invalid timestamps"):
            self.audit(hourly, totals(source()))
        hourly = source()
        hourly["ds"] = hourly.ds.dt.tz_localize("America/Montreal")
        with self.assertRaisesRegex(ValueError, "naive Montreal"):
            self.audit(hourly, totals(source()))


if __name__ == "__main__":
    unittest.main()
