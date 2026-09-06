from __future__ import annotations

"""Run a retrospective on-call experiment inside explicit label coverage.

Usage:
    python retrospective_oncall_label_safe_runner.py <module_name> [module args...]

This wrapper intentionally does not change the production loader. It replaces the
experiment module's imported load_dataset() with a guarded version that truncates to
the final explicit row in hourly_oncall_used_for_busy_since_2022.csv before any
future activation targets are constructed.
"""

import importlib
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from retrospective_oncall_label_coverage import (  # noqa: E402
    explicit_label_bounds,
    truncate_to_explicit_label_coverage,
)


def patch_module(module) -> None:
    # Most experiment modules import load_dataset directly.
    if hasattr(module, "load_dataset"):
        original = module.load_dataset

        def guarded_load_dataset():
            return truncate_to_explicit_label_coverage(original())

        module.load_dataset = guarded_load_dataset

    # The leakage-safe congestion wrapper delegates its main() to a base module.
    delegated = getattr(module, "base", None)
    if delegated is not None and hasattr(delegated, "load_dataset"):
        original = delegated.load_dataset

        def guarded_base_load_dataset():
            return truncate_to_explicit_label_coverage(original())

        delegated.load_dataset = guarded_base_load_dataset


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: retrospective_oncall_label_safe_runner.py <module> [args...]")
    module_name = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    module = importlib.import_module(module_name)
    patch_module(module)
    start, end = explicit_label_bounds()
    print(f"Explicit on-call label coverage: {start} through {end}")
    module.main()


if __name__ == "__main__":
    main()
