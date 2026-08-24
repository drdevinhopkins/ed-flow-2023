# ED Flow 2023

A Python-based emergency-department flow monitoring, forecasting, staffing-analysis, anomaly-detection, and operational decision-support pipeline.

The repository combines historical and current ED flow data with ShiftAdmin staffing, weather, calendar/holiday context, and on-call information. GitHub Actions and local automation publish derived datasets to Dropbox for downstream analytics and Power BI reporting.

For current implementation status and near-term priorities, see [`docs/project-status.md`](docs/project-status.md). The master feature-engineering roadmap is tracked in [issue #24](https://github.com/drdevinhopkins/ed-flow-2023/issues/24).

## Modeling architecture

The project deliberately treats three forecasting problems separately:

1. **Daily ED arrivals / inflow** — native Chronos-2 with calendar/holiday and validated weather features.
2. **Hourly operational flow** — forecasts crowding/flow targets over the next 24 hours.
3. **On-call decision support** — estimates probability of on-call activation and modeled flow impact under activation scenarios.

Scheduled physician staffing is intentionally excluded from the default daily-arrivals model because staffing is operational supply rather than a driver of patient arrival volume. Staffing is, however, important for hourly flow and on-call modeling.

## What the project does

- Downloads and parses the ED hourly PDF report into tabular data.
- Maintains current and historical ED flow datasets.
- Fetches and expands ShiftAdmin schedules into hourly staffing features.
- Maintains calendar/holiday and weather covariates.
- Generates the existing/legacy hourly Chronos-2 forecast products.
- Generates an **independent additive hourly `forecast-v2.csv`** using validated target × horizon feature routing.
- Produces daily-arrival forecasts with engineered holiday and weather features.
- Estimates 4h/6h/8h on-call activation probabilities.
- Produces physician-aware modeled on-call impact scenarios.
- Calculates operational KPIs and anomaly ranges.
- Publishes forecast, staffing, alert, and decision-support datasets to Dropbox for Power BI.

## Repository layout

```text
.
├── .github/workflows/          Production, validation, and backtest workflows
├── alert_examples/             Example alert payloads
├── automation/                 Local operations and systemd runbooks
├── data/                       Checked-in sample and historical datasets
├── docs/
│   └── project-status.md       Current forecasting status and roadmap summary
├── models/                     Generated model artifacts (ignored by Git)
├── notebooks/                  Exploratory and legacy notebooks
├── scripts/
│   ├── get_current.py          Parse the current hourly PDF and update data
│   ├── shiftadmin.py           Fetch and transform staffing schedules
│   ├── chronos_forecast.py     Existing hourly Chronos-2 production forecast
│   ├── hourly_forecast_v2.py   Independent target/horizon-routed hourly v2 forecast
│   ├── hourly_feature_routing.py  Validated target × horizon routing policy
│   ├── chronos_forecast_v2.py  Historical on-call-aware Chronos experiment
│   ├── forecast_oncall_probability.py  4h/6h/8h on-call activation probability
│   ├── forecast_oncall_impact.py       Physician-aware on-call scenario forecasts
│   ├── validate_forecast_inputs.py     Freshness/completeness gates
│   ├── calculated_kpis.py      Forecast and operational KPIs
│   ├── anomaly_detection.py    Anomaly-range generation
│   ├── update_weather.py       Weather update pipeline
│   ├── alerts.py               Build and upload alert outputs
│   └── run_ed_flow_update.sh   Existing local end-to-end update sequence
├── tests/                      Feature/routing regression tests
├── main.py                     Minimal project scaffold entry point
├── pyproject.toml              uv/project dependency definition
└── *requirements.txt           Workflow-specific dependency sets
```

> **Naming note:** `scripts/hourly_forecast_v2.py` is the current independent hourly v2 producer. `scripts/chronos_forecast_v2.py` is an older on-call-related experiment retained for historical compatibility.

## Existing hourly pipeline

The existing hourly pipeline remains intentionally unchanged by the v2 work.

The local wrapper runs the core sequence:

```text
get_current.py
  → shiftadmin.py
  → chronos_forecast.py
  → forecast_oncall_impact.py
  → forecast_oncall_probability.py
  → calculated_kpis.py
  → alerts.py
```

The Dropbox watcher in `scripts/watch_dropbox_pdf.py` can trigger the wrapper when `/hourlyreport.pdf` changes.

Representative existing forecast products include:

- `chronos_forecast.csv`
- `ED_Hourly_Forecasts_Anomalies_v1.0.csv`
- `forecast_variable_effects.csv`
- `forecast_variable_effects_hourly.csv`

The additive v2 workflow does **not** replace, patch, or overwrite these files.

## Independent hourly forecast v2

The v2 forecast was introduced after a common-base native Chronos-2 feature ablation across six operational targets:

- `Total_TBS`
- `POD_TBS`
- `Vertical_TBS`
- `TTStr`
- `Overflow`
- `WAITINGADM`

The final validation showed that there is **no universal best feature bundle**. The best representation depends on both target and forecast horizon, so v2 uses an explicit target × horizon routing table.

### Default safe routing

Weather-winning routes remain disabled by default until enough exact forecast-time weather snapshots have accumulated for prospective validation.

| Target | h1–4 | h5–8 | h9–12 | h13–24 |
| --- | --- | --- | --- | --- |
| `Overflow` | baseline | baseline | baseline | baseline |
| `POD_TBS` | staffing structure/effects | baseline | current staffing | current staffing |
| `TTStr` | current staffing | current staffing | current staffing | calendar demand |
| `Total_TBS` | calendar demand | current staffing | current staffing | current staffing |
| `Vertical_TBS` | calendar demand | current staffing | current staffing | current staffing |
| `WAITINGADM` | staffing structure/effects | staffing structure/effects | staffing structure/effects | baseline |

The exact implementation lives in `scripts/hourly_feature_routing.py`.

Retrospective weather routes can be enabled with:

```bash
CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=1
```

They are **off by default** because historical hourly weather validation used revised/realized weather rather than archived exact forecast-time inputs.

### `forecast-v2.csv`

`scripts/hourly_forecast_v2.py` writes and uploads only:

```text
forecast-v2.csv
```

The file contains a continuous 48-hour window for each of the six targets:

- latest **24 historical observed hours**
- next **24 routed forecast hours**

That produces **288 rows total** (`6 targets × 48 hours`).

The Power BI-oriented schema keeps historical truth and future forecasts in separate columns:

- `ds`
- `target_name`
- `actual` — historical rows only
- `forecast` — future rows only
- `forecast_lower`
- `forecast_upper`
- `row_type` — `observed` or `forecast`
- `anomaly_yhat`
- `anomaly_yhat_lower`
- `anomaly_yhat_upper`
- `actual_anomaly`
- `actual_colour`
- `forecast_anomaly`
- `forecast_colour`
- `horizon_hour`
- `horizon_band`
- `scenario`
- `feature_family`
- `forecast_origin`
- `generated_at_utc`
- `routing_version`
- `weather_routing_enabled`

### Anomaly colours

Historical actual colours are **directional for ED congestion**:

- actual above the upper anomaly bound → red `#D13438`
- actual above expected but within the upper bound → amber `#FFB900`
- actual at/below expected, including below the lower anomaly bound → green `#107C10`

An unusually low actual still retains `actual_anomaly = yes` for statistical tracking; it is simply displayed as green because unusually low congestion is operationally favorable.

Forecast anomaly colours retain the existing two-sided convention.

## Daily arrivals

The daily Chronos-2 arrivals workflow includes engineered calendar and weather context.

Calendar features include, among others:

- Quebec/federal/institutional holidays
- major Jewish holidays and eves
- closure structure
- long-weekend shoulders
- pre/post-holiday and rebound effects

The validated daily weather representation adds raw weather plus engineered snow/post-snow context. In retrospective validation it improved MAE by roughly 6% overall, with larger gains on weather-event and post-major-snow days.

## On-call decision support

### Activation probability

`scripts/forecast_oncall_probability.py` estimates:

- `P(on-call activation within 4h)`
- `P(on-call activation within 6h)`
- `P(on-call activation within 8h)`

The CatBoost models use current ED state, recent trajectory, staffing structure, physician identity, scheduled on-call physician identity, and calendar/weather context.

Initial runtime AUROC was approximately 0.91 / 0.88 / 0.85 for 4h / 6h / 8h. Independent calibration validation and label-completeness checks remain open work.

### Modeled impact

`scripts/forecast_oncall_impact.py` compares:

- no on-call activation
- activation for 4 hours
- activation for 6 hours
- activation for 8 hours

Outputs include Total/POD/Vertical TBS, overflow, and stretcher occupancy. Stretcher occupancy is derived as:

```text
TTStr / 53 * 100
```

These are **model-based counterfactual forecasts, not causal treatment-effect estimates**. Matched-state/causal validation remains required before interpreting scenario deltas as the benefit of activating on-call.

## Automated workflows

Important workflows include:

| Workflow | Trigger / purpose |
| --- | --- |
| `hourly-update.yml` | Existing hourly refresh: current ED data, weather/METAR, staffing, legacy Chronos forecast, KPIs, alerts |
| `hourly-forecast-v2.yml` | Runs the independent v2 forecast after a successful `Hourly update`; also supports manual/PR validation |
| `hourly-target-routing-ci.yml` | Routing regression tests and guards that legacy execution files remain unchanged |
| `hourly-final-feature-ablation.yml` | Common-base hourly feature-family validation |
| `hourly-weather-feature-backtest.yml` | Hourly weather representation backtesting |
| `daily-visits-forecast.yml` | Native Chronos-2 daily-arrival forecast |
| `calendar-context-backtest.yml` | Calendar-context validation |
| `holiday-feature-backtest.yml` | Holiday-feature validation |
| `update-weather.yml` | Scheduled weather refresh |
| `daily-training.yml` | Legacy/daily model training and anomaly-related work |
| `chronos-forecast.yml` | Direct legacy Chronos forecast dispatch |

Configure the relevant GitHub Actions secrets as appropriate, including:

```dotenv
DROPBOX_APP_KEY=...
DROPBOX_APP_SECRET=...
DROPBOX_REFRESH_TOKEN=...
SHIFTADMIN_USER=...
SHIFTADMIN_PASS=...
HF_TOKEN=...
DETA_PROJECT_KEY=...
GITHUB_TOKEN=...
```

Not every workflow requires every secret.

## Requirements and installation

Project metadata currently targets Python 3.14, while several GitHub Actions forecasting workflows use Python 3.12. Keep this difference in mind when reproducing workflow environments locally.

Using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Chronos workflows use `chronos-requirements.txt`; weather and legacy model workflows have narrower requirement files where appropriate.

## Running forecasts manually

From a configured repository checkout:

```bash
source .venv/bin/activate
python scripts/validate_forecast_inputs.py
python scripts/chronos_forecast.py
```

To run the independent v2 forecast:

```bash
python scripts/validate_forecast_inputs.py
CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=0 python scripts/hourly_forecast_v2.py
```

The v2 script requires Dropbox credentials because its successful production path uploads `forecast-v2.csv`.

## Generated outputs

Representative outputs include:

- `current.csv`, `current.xlsx`
- `allData.csv`, `allData.xlsx`
- `allDataWithCalculatedColumns.csv`, `allDataWithCalculatedColumns.xlsx`
- `all_shifts.csv`, `hourly_shifts.csv`
- `chronos_forecast.csv`
- `ED_Hourly_Forecasts_Anomalies_v1.0.csv`
- **`forecast-v2.csv`**
- `forecast_variable_effects.csv`, `forecast_variable_effects_hourly.csv`
- `anomaly_detection_ranges.csv`
- `daily_inflow_forecast.csv`
- `alerts.csv`, `alerts.xlsx`
- `weather.csv`
- `oncall_impact_forecast.csv`, `oncall_impact_summary.csv`
- `oncall_need_probability.csv`, `oncall_need_probability_validation.csv`
- generated model artifacts under `models/`

Most generated data/model outputs are ignored by Git and are published through the operational data pipeline instead.

## Power BI reporting

Power BI consumes selected Dropbox data products from this repository.

The recommended new hourly-flow report should use `forecast-v2.csv` directly:

- put `ds` on the time axis
- use `actual` as the historical series
- use `forecast` as the future series
- use `forecast_lower` / `forecast_upper` for uncertainty bands where useful
- use `actual_colour` / `forecast_colour` for conditional formatting
- filter or facet by `target_name`

Because `actual` and `forecast` are separate columns, the observed-to-forecast transition can be plotted without reshaping the CSV.

The legacy Power BI reports can continue using the existing forecast CSVs because v2 does not replace them.

## Development and validation

The repository now includes regression tests for calendar, weather, daily forecasting, staffing, and target/horizon routing under `tests/`.

Useful checks include:

```bash
python -m compileall -q main.py scripts
pytest -q tests
```

The v2 GitHub Actions path also performs runtime/data-contract validation, including:

- exactly 24 observed and 24 forecast hours per target
- separate `actual` and `forecast` columns
- complete anomaly intervals/colour annotations
- contiguous observed-to-forecast handoff
- target/horizon routing checks
- weather-routing guard
- assertions that v2 does not create or modify the legacy forecast CSVs
- synthetic directional anomaly-colour regression cases

## Local operations

`automation/ed_flow_automation_check_runbook.md` documents the production-style local setup, including the Dropbox watcher and systemd services/timers.

The runbook references `/home/dhopkins/apps/ed-flow-2023`; update paths and service definitions when deploying elsewhere.

## Important limitations

- Some scripts still execute substantial work at startup rather than exposing fully reusable library/CLI interfaces.
- Several data sources are remote Dropbox URLs with schema assumptions embedded in scripts.
- Generated outputs are generally not versioned in Git.
- Python versions differ between local project metadata and GitHub Actions.
- Retrospective hourly weather validation is optimistic relative to true forecast-time conditions; weather-winning hourly routes remain disabled until prospective snapshot validation is sufficient.
- On-call probability calibration needs independent validation.
- On-call impact scenarios are predictive/model-based rather than causal estimates.

## Tracking

- [`docs/project-status.md`](docs/project-status.md) — repository-level status summary
- [#24](https://github.com/drdevinhopkins/ed-flow-2023/issues/24) — master feature-engineering roadmap
- [#23](https://github.com/drdevinhopkins/ed-flow-2023/issues/23) — physician staffing feature engineering (completed)
- [#18](https://github.com/drdevinhopkins/ed-flow-2023/issues/18) — forecast input gates / initial covariate ablation (completed)
- [#22](https://github.com/drdevinhopkins/ed-flow-2023/issues/22) — on-call activation probability
- [#7](https://github.com/drdevinhopkins/ed-flow-2023/issues/7) — on-call counterfactual impact

## License and ownership

No license file is currently present. Add a license before distributing or reusing the project outside its intended environment.
