from __future__ import annotations

"""Run the ocU replay using the conservative pre-May exact-hour label boundary.

The base ocU replay originally used the physical end of the hourly CSV (May 15).  An
overlap audit subsequently found known false-negative hourly labels on May 1--4,
where schedule ``ocU`` rows independently confirm on-call use.  This wrapper swaps in
the trusted hourly-label bound (through April 30) without duplicating the replay.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = Path(__file__).resolve().parents[2]
for path in (HERE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import retrospective_oncall_ocu_replay as base  # noqa: E402
from retrospective_oncall_label_coverage import trusted_hourly_label_bounds  # noqa: E402


base.explicit_label_bounds = trusted_hourly_label_bounds


if __name__ == "__main__":
    base.main()
