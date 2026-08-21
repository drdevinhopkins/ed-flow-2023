# ED Flow 2023

A Python-based data pipeline for emergency-department flow monitoring, forecasting, staffing analysis, anomaly detection, and operational alerts.

The repository combines historical ED flow data with current reports, ShiftAdmin staffing data, weather, holidays, and on-call information. Scheduled GitHub Actions and local automation jobs generate forecasts and publish derived datasets to Dropbox.

## What the project does

- Downloads and parses the ED hourly PDF report into tabular data.
- Maintains current and historical ED flow datasets.
- Fetches and expands ShiftAdmin schedules into hourly staffing features.
- Generates hourly forecasts with Amazon Chronos-2.
- Generates NeuralProphet forecasts for inflow and total treatment-bay/stretcher (TBS) demand.
- Calculates operational KPIs and produces charts and alert artifacts.
- Detects values above forecast ranges and uploads alerts.
- Retrieves weather data from Open-Meteo and joins it to forecasting data.
- Supports on-call impact and on-call-need forecasting experiments.

## Repository layout

```text
.
├── .github/workflows/       Scheduled and repository-dispatch workflows
├── alert_examples/          Example alert payloads
├── automation/              Local operations and systemd runbooks
├── data/                    Checked-in sample and historical datasets
├── docs/                    Analysis notes and forecasting documentation
├── models/                  Generated model artifacts (ignored by Git)
├── notebooks/               Exploratory and legacy notebooks
├── scripts/
│   ├── get_current.py       Parse the current hourly PDF and update data
│   ├── shiftadmin.py        Fetch and transform staffing schedules
│   ├── chronos_forecast.py  Chronos-2 forecasts and variable-effect analysis
│   ├── chronos_forecast_v2.py  On-call-aware Chronos experiment
│   ├── total_tbs_np.py      Train NeuralProphet inflow/TBS models
│   ├── calculated_kpis.py   Calculate forecast and operational KPIs
│   ├── anomaly_detection.py  Generate anomaly alerts
│   ├── update_weather.py    Update historical weather data
│   ├── alerts.py             Build and upload alert outputs
│   └── run_ed_flow_update.sh Run the local end-to-end update sequence
├── main.py                  Minimal project scaffold entry point
├── pyproject.toml            uv/project dependency definition
└── *requirements.txt         Workflow-specific dependency sets
```

## Data flow

The primary update sequence is:

1. `get_current.py` downloads and parses `hourlyreport.pdf`, updates `current.*` and `allData.*`, calculates aggregate fields, and uploads the results to Dropbox.
2. `shiftadmin.py` fetches a rolling window of schedules, writes `all_shifts.csv` and `hourly_shifts.csv`, and uploads them.
3. `chronos_forecast.py` combines flow, staffing, holiday, and weather data to produce forecasts and variable-effect comparisons.
4. `calculated_kpis.py` creates Prophet-based comparisons and KPI artifacts.
5. `alerts.py` compares current observations with anomaly ranges and writes `alerts.csv`/`alerts.xlsx`.

The Dropbox watcher in `scripts/watch_dropbox_pdf.py` can trigger that wrapper when `/hourlyreport.pdf` changes.

The local wrapper runs the core sequence:

```text
get_current.py → shiftadmin.py → chronos_forecast.py → calculated_kpis.py → alerts.py
```

## Requirements

- Python 3.14 for the project configuration (`.python-version` and `pyproject.toml`).
- A working C/C++ build environment may be needed by some scientific Python packages.
- Dropbox API credentials for scripts that read or upload project data.
- ShiftAdmin credentials for `scripts/shiftadmin.py`.
- A Hugging Face token for Chronos model access when required by the model/runtime.
- Sufficient memory and disk space for PyTorch and forecasting model caches.

The GitHub Actions currently use Python 3.12, while the project metadata specifies Python 3.14. Align these versions before relying on one environment as the canonical deployment target.

## Installation

Using `uv`:

```bash
uv sync
```

Or create a virtual environment and install the relevant dependency set with `pip`:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Additional workflows use narrower requirement files:

- `chronos-requirements.txt` — Chronos forecasting workflow.
- `weather-requirements.txt` — Open-Meteo weather update.
- `nprequirements.txt` — NeuralProphet experiment.
- `requirements-jgh.txt` — older/experimental JGH data workflow.

The Chronos GitHub workflow installs CPU PyTorch separately:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r chronos-requirements.txt
```

## Configuration

Create a local `.env` file or export the variables in the execution environment. `.env` is ignored by Git and must never be committed.

Common variables:

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

Not every script needs every variable:

| Variable | Used by |
| --- | --- |
| `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_REFRESH_TOKEN` | Most data update, forecast, and upload scripts |
| `SHIFTADMIN_USER`, `SHIFTADMIN_PASS` | `scripts/shiftadmin.py` |
| `HF_TOKEN` | Chronos workflows/model access when required |
| `DETA_PROJECT_KEY` | Daily training workflow's anomaly step |
| `GITHUB_TOKEN` | Optional private model-release downloads in `total_tbs_np_forecast.py` |

Several scripts also read source files from Dropbox URLs embedded in the code. Those remote files must exist and have the expected columns before the scripts can run.

## Running the pipeline

From the repository root:

```bash
source .venv/bin/activate
python scripts/get_current.py
python scripts/shiftadmin.py
python scripts/chronos_forecast.py
python scripts/calculated_kpis.py
python scripts/alerts.py
```

Or use the local wrapper after updating its repository path and virtual-environment assumptions for your machine:

```bash
bash scripts/run_ed_flow_update.sh
```

Forecasting scripts may download model weights and can take substantially longer than the data preparation steps. `chronos_forecast.py` is configured for CPU inference on the current branch; `chronos_forecast_v2.py` still requests CUDA and requires compatible GPU support.

## Automated workflows

| Workflow | Trigger | Main action |
| --- | --- | --- |
| `hourly-update.yml` | GitHub `repository_dispatch` event `trigger_get_current_data_action` | Refresh current data, weather/METAR, staffing, Chronos forecasts, KPIs, and alerts |
| `chronos-forecast.yml` | GitHub `repository_dispatch` event `trigger_chronos_forecast` | Run `scripts/chronos_forecast.py` |
| `daily-training.yml` | Daily cron at `06:00 UTC` | Train NeuralProphet models, run anomaly detection, and upload model release assets |
| `update-weather.yml` | Every four hours | Run `scripts/update_weather.py` and upload `weather.csv` |

The repository-dispatch workflows require an external caller to send the matching event. Configure the following GitHub Actions secrets as appropriate: `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_REFRESH_TOKEN`, `SHIFTADMIN_USER`, `SHIFTADMIN_PASS`, `HF_TOKEN`, and `DETA_PROJECT_KEY`.

The daily workflow publishes model assets to the `v1` GitHub release. Bump `MODEL_VERSION` in the workflow when intentionally creating a new model cache/release version.

## Generated outputs

Most generated CSV, Excel, PDF, model, and cache files are ignored by Git. Representative outputs include:

- `current.csv`, `current.xlsx`
- `allData.csv`, `allData.xlsx`
- `allDataWithCalculatedColumns.csv`, `allDataWithCalculatedColumns.xlsx`
- `all_shifts.csv`, `hourly_shifts.csv`
- `chronos_forecast.csv`
- `ED_Hourly_Forecasts_Anomalies_v1.0.csv`
- `forecast_variable_effects.csv`, `forecast_variable_effects_hourly.csv`
- `anomaly_detection_ranges.csv`, `daily_inflow_forecast.csv`
- `alerts.csv`, `alerts.xlsx`
- `weather.csv`
- `models/total_tbs-<version>.np`, `models/inflow_total_np-<version>.np`
- `oncall_need_forecast.csv` for the on-call-need experiment

## Local operations

`automation/ed_flow_automation_check_runbook.md` documents the production-style local setup, including:

- A persistent Dropbox watcher service.
- A four-hour weather timer.
- A daily anomaly-detection timer.
- systemd status and journal commands for troubleshooting.

The runbook currently references `/home/dhopkins/apps/ed-flow-2023`; update that path and the service definitions when deploying elsewhere.

## Development and validation

There are currently no checked-in unit tests, and the linting/testing GitHub workflow is commented out. Before changing a script, at minimum validate Python syntax:

```bash
python -m compileall -q main.py scripts
```

For data or model changes, run the affected script in a configured environment and inspect its generated files and logs. Most scripts perform network I/O and write/upload artifacts at module execution time, so importing them for isolated unit testing is not currently supported.

## Important limitations

- Several scripts execute work at import/startup time rather than exposing reusable CLI functions.
- Remote Dropbox URLs and expected schemas are embedded in scripts, which makes reproducibility and local testing dependent on external data.
- Generated outputs are ignored, so pipeline results are not normally versioned in Git.
- Workflow steps marked `continue-on-error: true` can allow a job to proceed after an intermediate failure; inspect logs and output-file checks rather than relying only on a green workflow status.
- Python versions differ between local project metadata and GitHub Actions and should be standardized.

## License and ownership

No license file is currently present in the repository. Add a license before distributing or reusing the project outside its intended environment.
