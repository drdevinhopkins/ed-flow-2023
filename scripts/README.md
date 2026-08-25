# Scripts directory

This directory contains operational entrypoints, shared forecasting helpers, evaluation runners, and research experiments.

## Layout

```text
scripts/
├── evaluation/
│   ├── backtests/       # standalone retrospective validation runners
│   └── ablations/       # controlled feature/source comparisons
├── experiments/
│   └── autogluon/       # alternative-model prototypes not used by production
├── legacy/              # historical scripts retained for reference
└── *.py                 # operational entrypoints and compatibility-sensitive shared modules
```

## Rules

1. **Operational paths stay stable.** Scripts invoked by server automation, systemd, Dropbox watchers, or production GitHub Actions should not be moved without updating and validating every caller.
2. **Standalone evaluation belongs under `evaluation/`.** New backtests and ablations should not be added to the scripts root.
3. **Experiments belong under `experiments/`.** Experimental model stacks must not overwrite production forecast artifacts by default.
4. **Reusable code should become package code.** Feature engineering, routing, data loading, metrics, and model helpers should gradually move out of executable scripts into a proper `src/ed_flow/` package.
5. **Production should not depend on evaluation modules.** Some current production forecasts still import legacy `backtest_*` modules. Those files remain at the scripts root temporarily to preserve behavior; extract their reusable functions into `src/ed_flow/` before relocating them.

## Running relocated evaluation scripts

Relocated runners may still import compatibility modules from the scripts root. Run them with the scripts directory on `PYTHONPATH`, for example:

```bash
PYTHONPATH=scripts python scripts/evaluation/backtests/backtest_daily_weather_features_dense.py --help
```

The corresponding GitHub Actions workflows set `PYTHONPATH=scripts` explicitly.

## Migration target

The next cleanup phase should introduce a package such as:

```text
src/ed_flow/
├── data/
├── features/
├── forecasting/
├── routing/
└── evaluation/
```

Production scripts can then become thin CLI entrypoints, after which the remaining compatibility-sensitive `backtest_*` modules can be renamed or relocated safely.
