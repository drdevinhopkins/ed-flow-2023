from __future__ import annotations

"""Helpers for enforcing trustworthy historical on-call label coverage.

The source hourly label CSV contains rows through 2026-05-15 06:00, but row presence is
not sufficient evidence that the values are complete.  The newer schedule marker
``ocU`` ("OC busy used") begins on 2026-05-01 and confirms on-call use on May 1--4,
while the hourly CSV records every hour on those dates as zero.  Therefore the May
portion of the hourly file contains known false negatives.

For retrospective activation-dependent validation we conservatively treat:

- through 2026-04-30 23:00: hourly labels are the trusted high-resolution source;
- from 2026-05-01 onward: hourly activation state is unknown unless separately
  backfilled; ``ocU`` may be used only as a day-level positive marker.

This boundary is a data-quality guard, not a claim that every pre-May label is perfect.
"""

import pandas as pd

from forecast_oncall_probability import ONCALL_LABELS_PATH, TS_COL


# First day on which the schedule contains the newer ocU bookkeeping marker.  Direct
# overlap audit found ocU positives on 2026-05-01..04 while the hourly label CSV says
# zero throughout those dates, so May 1 is the first known-untrustworthy hourly day.
TRUSTED_HOURLY_LABEL_END = pd.Timestamp("2026-04-30 23:00:00")


def explicit_label_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Physical timestamp bounds represented by rows in the hourly label CSV."""
    labels = pd.read_csv(ONCALL_LABELS_PATH, usecols=[TS_COL])
    ts = pd.to_datetime(labels[TS_COL], errors="coerce").dropna()
    if ts.empty:
        raise ValueError(f"No valid timestamps in {ONCALL_LABELS_PATH}")
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max())


def trusted_hourly_label_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the conservative interval usable for exact-hour validation."""
    start, explicit_end = explicit_label_bounds()
    end = min(explicit_end, TRUSTED_HOURLY_LABEL_END)
    if end < start:
        raise ValueError(f"Trusted hourly label interval is empty: {start}..{end}")
    return start, end


def truncate_to_explicit_label_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows inside the trusted exact-hour label interval.

    Kept under the existing function name so earlier experiment callers inherit the
    stricter guard automatically.  This must be called before add_horizon_targets().
    """
    start, end = trusted_hourly_label_bounds()
    ts = pd.to_datetime(df[TS_COL], errors="coerce")
    out = df[ts.between(start, end, inclusive="both")].copy()
    if out.empty:
        raise ValueError(
            f"No operational rows overlap trusted on-call label coverage {start}..{end}"
        )
    out.attrs["oncall_label_coverage_start"] = start
    out.attrs["oncall_label_coverage_end"] = end
    out.attrs["oncall_label_explicit_file_end"] = explicit_label_bounds()[1]
    return out.reset_index(drop=True)
