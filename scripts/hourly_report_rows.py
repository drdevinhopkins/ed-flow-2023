"""Resolve duplicate labels within a single hourly report before merging revisions."""
from __future__ import annotations

import pandas as pd


def collapse_report_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the maximum of each value for rows sharing a report timestamp.

    Apply to ONE source report, not concatenated historical revisions: ordinary
    corrections must still replace older values, including downward corrections.
    Maxima may come from different source rows, as requested for the DST report.
    """
    if "ds" not in frame:
        raise ValueError("Report is missing ds")
    out = frame.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="raise")
    if out["ds"].isna().any():
        raise ValueError("Report contains missing timestamps")
    return out.groupby("ds", as_index=False, sort=True).max()[frame.columns]


def merge_report_history(history: pd.DataFrame, report: pd.DataFrame) -> pd.DataFrame:
    """Collapse a fresh report, then replace matching historical timestamps."""
    current = collapse_report_rows(report)
    old = history.copy()
    old["ds"] = pd.to_datetime(old["ds"], errors="raise")
    return (pd.concat([old, current], ignore_index=True)
            .drop_duplicates("ds", keep="last")
            .sort_values("ds").reset_index(drop=True))
