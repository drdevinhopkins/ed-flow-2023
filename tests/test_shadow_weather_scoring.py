import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluation.prospective.score_shadow_weather_forecast import _summarize


def test_weather_summary_does_not_count_unchanged_routes_as_losses():
    detail = pd.DataFrame(
        {
            "target_name": ["Total_TBS"] * 4,
            "horizon_band": ["1-4h"] * 4,
            "actual": [10.0, 10.0, 10.0, 10.0],
            "baseline_absolute_error": [4.0, 3.0, 0.0, 0.0],
            "weather_absolute_error": [2.0, 1.0, 0.0, 0.0],
            "paired_absolute_error_delta": [2.0, 2.0, 0.0, 0.0],
            "weather_wins": [True, True, False, False],
            "baseline_error": [4.0, 3.0, 0.0, 0.0],
            "weather_error": [2.0, 1.0, 0.0, 0.0],
            "baseline_squared_error": [16.0, 9.0, 0.0, 0.0],
            "weather_squared_error": [4.0, 1.0, 0.0, 0.0],
            "weather_route_active": [True, True, False, False],
        }
    )

    # All-pair win rate is diluted by unchanged routes.
    all_summary = _summarize(detail)
    assert all_summary.loc[0, "weather_win_rate"] == 0.5

    # Promotion scoring must use only rows where a weather route was actually active.
    active_summary = _summarize(detail.loc[detail["weather_route_active"]])
    assert active_summary.loc[0, "weather_win_rate"] == 1.0
    assert active_summary.loc[0, "prospective_pass_directional"]
