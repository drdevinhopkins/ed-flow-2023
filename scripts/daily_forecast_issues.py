"""Scheduled daily forecast issue labels, independent of model or publishing code."""
from __future__ import annotations

import pandas as pd

LOCAL_TZ = 'America/Montreal'
ISSUE_SLOTS = ('morning', 'evening', 'manual')


def issue_metadata(generated_at, *, cutoff, slot='manual'):
    """Label actual issue time; never backdate an evening catch-up as a morning issue."""
    stamp = pd.Timestamp(generated_at)
    if stamp.tzinfo is None:
        raise ValueError('Forecast issue timestamp must be timezone-aware')
    if slot not in ISSUE_SLOTS:
        raise ValueError(f'Unknown forecast issue slot: {slot}')
    local = stamp.tz_convert(LOCAL_TZ)
    day = local.tz_localize(None).normalize()
    if slot != 'manual':
        if pd.Timestamp(cutoff).normalize() != day - pd.Timedelta(days=1):
            raise ValueError('Scheduled forecast requires yesterday as its data cutoff')
        expected = 'morning' if 6 <= local.hour < 18 else 'evening' if local.hour >= 18 else None
        if slot != expected:
            raise ValueError(f'Issue time {local} is outside the {slot} refresh window')
    return {'forecast_issue_slot': slot, 'forecast_issue_day_local': day.date().isoformat()}
