#!/usr/bin/env python3
"""Run the routed history-window experiment on immutable forecast origins.

Research-only wrapper around ``backtest_hourly_history_windows_routed``.  The
underlying backtest historically selected origins relative to the latest row in
the live source, which meant otherwise identical reruns could shift by an hour
or more as the dataset advanced.  This wrapper pins the eight Stage-2 origins so
history-window and long-term-memory experiments can be compared apples-to-apples.
"""

from __future__ import annotations

import pandas as pd

import backtest_hourly_history_windows_routed as routed


# Eight 24-hour forecast origins, spaced exactly six weeks (1,008 hours) apart.
# These are the Stage-2 origins used by the long-term-memory/growth experiment.
FIXED_CUTOFFS = pd.to_datetime(
    [
        "2025-11-02 08:00:00",
        "2025-12-14 08:00:00",
        "2026-01-25 08:00:00",
        "2026-03-08 08:00:00",
        "2026-04-19 08:00:00",
        "2026-05-31 08:00:00",
        "2026-07-12 08:00:00",
        "2026-08-23 08:00:00",
    ]
).tolist()


def fixed_cutoffs(flow, *, horizon, num_cutoffs, spacing_hours, max_history_hours):
    """Return the immutable validation origins after invariant/availability checks."""
    if num_cutoffs != len(FIXED_CUTOFFS):
        raise ValueError(
            f"Fixed validation requires num_cutoffs={len(FIXED_CUTOFFS)}, "
            f"got {num_cutoffs}"
        )
    if spacing_hours != 1008:
        raise ValueError(
            f"Fixed validation requires spacing_hours=1008, got {spacing_hours}"
        )

    cutoffs = pd.DatetimeIndex(FIXED_CUTOFFS)
    if not cutoffs.is_monotonic_increasing or not cutoffs.is_unique:
        raise ValueError("Fixed validation cutoffs must be unique and increasing")
    observed_spacing = cutoffs.to_series().diff().dropna()
    expected_spacing = pd.Timedelta(hours=spacing_hours)
    if not observed_spacing.eq(expected_spacing).all():
        raise ValueError(
            "Fixed validation cutoffs no longer match the required 1,008-hour spacing"
        )

    ds = pd.to_datetime(flow["ds"])
    earliest = ds.min()
    latest = ds.max()
    for cutoff in FIXED_CUTOFFS:
        if cutoff - pd.Timedelta(hours=max_history_hours) < earliest:
            raise ValueError(f"Insufficient pre-cutoff history for {cutoff}")
        if cutoff + pd.Timedelta(hours=horizon) > latest:
            raise ValueError(f"Missing post-cutoff actuals for {cutoff}")
    return FIXED_CUTOFFS.copy()


def main() -> None:
    # Patch only the research helper used by routed.main(); production code and
    # production workflows do not import this wrapper.
    routed.history_bt.select_cutoffs = fixed_cutoffs
    routed.main()


if __name__ == "__main__":
    main()
