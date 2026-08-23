from __future__ import annotations

import os

import autogluon_backtest as ag_base
from chronos_forecast_autogluon import BATCH_SIZE if False else None
from chronos_forecast_autogluon import DEVICE, MODEL_PATH


def chronos_only_hyperparameters() -> dict[str, dict]:
    return {
        "Chronos2": {
            "model_path": MODEL_PATH,
            "device": DEVICE,
            "batch_size": int(os.environ.get("AUTOGLUON_BATCH_SIZE", "8")),
        }
    }


def main() -> None:
    # Reuse the exact staffing-only preparation/scoring code while restricting
    # AutoGluon to its Chronos-2 wrapper. This isolates framework/representation
    # differences from model-selection and ensemble effects.
    ag_base.model_hyperparameters = chronos_only_hyperparameters

    import chronos_autogluon_staffing_comparison as comparison

    comparison.main()


if __name__ == "__main__":
    main()
