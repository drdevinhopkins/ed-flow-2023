from __future__ import annotations

"""Helpers for enforcing verified historical on-call label coverage.

The production loader left-merges the label table and fills missing labels with zero.
That is convenient inside the known-complete label interval, but it is unsafe for
retrospective validation after the label file itself ends: absent future rows are not
verified negatives.

All retrospective activation-dependent analyses therefore truncate the operational
state to the final explicit timestamp in the source label file *before* horizon targets
are constructed. The last H hours then naturally become NA for an H-hour target.
"""

import pandas as pd

from forecast_oncall_probability import ONCALL_LABELS_PATH, TS_COL


def explicit_label_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    labels = pd.read_csv(ONCALL_LABELS_PATH, usecols=[TS_COL])
    ts = pd.to_datetime(labels[TS_COL], errors="coerce").dropna()
    if ts.empty:
        raise ValueError(f"No valid timestamps in {ONCALL_LABELS_PATH}")
    return pd.Timestamp(ts.min()), pd.Timestamp(ts.max())


def truncate_to_explicit_label_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows whose on-call activation state is explicitly represented.

    This must be called before add_horizon_targets().
    """
    start, end = explicit_label_bounds()
    ts = pd.to_datetime(df[TS_COL], errors="coerce")
    out = df[ts.between(start, end, inclusive="both")].copy()
    if out.empty:
        raise ValueError(
            f"No operational rows overlap explicit on-call label coverage {start}..{end}"
        )
    out.attrs["oncall_label_coverage_start"] = start
    out.attrs["oncall_label_coverage_end"] = end
    return out.reset_index(drop=True)
