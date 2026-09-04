#!/usr/bin/env python3
"""Compatibility entrypoint for the intraday day-completion backtest."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from intraday_day_completion_model import *  # noqa: F401,F403,E402
from intraday_day_completion_model import _prediction_rows, main  # noqa: E402,F401


if __name__ == "__main__":
    main()
