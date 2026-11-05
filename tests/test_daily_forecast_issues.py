import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from daily_forecast_issues import issue_metadata


@pytest.mark.parametrize('stamp,slot,cutoff', [
    ('2026-05-01T10:00Z', 'morning', '2026-04-30'),
    ('2026-05-01T22:00Z', 'evening', '2026-04-30'),
    ('2026-11-01T11:00Z', 'morning', '2026-10-31'),
    ('2026-11-01T23:00Z', 'evening', '2026-10-31'),
])
def test_refresh_windows_follow_montreal_clock_across_dst(stamp, slot, cutoff):
    labels = issue_metadata(stamp, cutoff=cutoff, slot=slot)
    assert labels['forecast_issue_slot'] == slot
    assert labels['forecast_issue_day_local'] == (pd.Timestamp(cutoff) + pd.Timedelta(days=1)).date().isoformat()


@pytest.mark.parametrize('stamp,slot,cutoff', [
    ('2026-11-01T10:59Z', 'morning', '2026-10-31'),
    ('2026-11-01T22:59Z', 'evening', '2026-10-31'),
    ('2026-11-01T23:00Z', 'morning', '2026-10-31'),
    ('2026-11-01T23:00Z', 'evening', '2026-10-30'),
])
def test_scheduled_labels_cannot_misrepresent_timing_or_stale_cutoff(stamp, slot, cutoff):
    with pytest.raises(ValueError):
        issue_metadata(stamp, cutoff=cutoff, slot=slot)
