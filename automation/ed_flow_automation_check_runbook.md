# ED Flow Automation Check Runbook

This runbook covers the three automations on `jgh000533svaps` for the ED flow project:

1. Dropbox-triggered hourly update watcher
2. Weather update every 4 hours
3. Daily anomaly detection at 6:00 AM local Eastern time

Project directory:

```bash
/home/dhopkins/apps/ed-flow-2023
```

Python environment used by the automations:

```bash
/home/dhopkins/apps/ed-flow-2023/.venv/bin/python
```

---

## 0. Quick all-in-one status check

Run this first:

```bash
echo "=== SERVICES ==="
systemctl list-units --type=service --all --no-pager | grep -Ei 'ed-flow|dropbox|weather|anomaly|update'

echo
echo "=== TIMERS ==="
systemctl list-timers --all --no-pager | grep -Ei 'ed-flow|dropbox|weather|anomaly|update'

echo
echo "=== WEATHER TIMER ==="
systemctl status ed-flow-weather.timer --no-pager -l

echo
echo "=== ANOMALY TIMER ==="
systemctl status ed-flow-anomaly.timer --no-pager -l

echo
echo "=== DROPBOX WATCHER ==="
systemctl status ed-flow-dropbox-watcher.service --no-pager -l
```

Expected high-level result:

- `ed-flow-dropbox-watcher.service` should be `active (running)`.
- `ed-flow-weather.timer` should be `active (waiting)`.
- `ed-flow-anomaly.timer` should be `active (waiting)`.
- Weather and anomaly services are usually `inactive (dead)` between runs because they are `oneshot` jobs.

---

## 1. Dropbox-triggered hourly update watcher

### Purpose

Watches Dropbox for changes to:

```text
/hourlyreport.pdf
```

When that PDF changes, it runs:

```bash
/home/dhopkins/apps/ed-flow-2023/scripts/run_ed_flow_update.sh
```

The wrapper is the canonical local hourly pipeline on `jgh000533svaps`.

Current sequence:

```text
get_current.py
  → update_metar.py
  → shiftadmin.py
  → validate_forecast_inputs.py
  → chronos_forecast.py
  → forecast_oncall_impact.py
  → forecast_oncall_probability.py
  → calculated_kpis.py
  → alerts.py
  → hourly_forecast_v2.py  (additive/non-blocking)
```

Important behavior:

- The wrapper loads `.env` so direct/manual invocation has the same credentials as the Dropbox watcher.
- `CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=0` is exported explicitly. Hourly weather-winning v2 routes stay disabled until prospective forecast-time validation is sufficient.
- `validate_forecast_inputs.py` is a hard gate. If ED, staffing, or weather inputs are stale/incomplete, the established forecast pipeline stops rather than producing a misleading forecast.
- `hourly_forecast_v2.py` runs last as an additive step. If v2 fails, the established legacy forecast, on-call, KPI, and alert outputs have already completed and the wrapper records a warning rather than failing the whole workflow.
- The separate `ed-flow-weather.timer` remains responsible for refreshing `weather.csv`; the hourly wrapper does not duplicate `update_weather.py`.
- `update_metar.py` does run on every hourly PDF-triggered refresh so METAR history is kept current.

### Expected outputs

The established pipeline continues to publish its existing outputs, including:

```text
chronos_forecast.csv
ED_Hourly_Forecasts_Anomalies_v1.0.csv
forecast_variable_effects.csv
forecast_variable_effects_hourly.csv
oncall_impact_forecast.csv
oncall_impact_summary.csv
oncall_need_probability.csv
oncall_need_probability_validation.csv
alerts.csv
alerts.xlsx
```

The additive v2 step publishes:

```text
forecast-v2.csv
```

### Check service status

```bash
systemctl status ed-flow-dropbox-watcher.service --no-pager -l
```

Expected:

```text
Active: active (running)
```

### Check recent logs

```bash
sudo journalctl -u ed-flow-dropbox-watcher.service --since "24 hours ago" --no-pager
```

Filtered version:

```bash
sudo journalctl -u ed-flow-dropbox-watcher.service --since "24 hours ago" --no-pager \
  | grep -Ei 'started|running|dropbox|cursor|longpoll|change|changed|hourlyreport|pdf|workflow|START:|DONE:|WARNING:|validation|METAR|forecast v2|forecast-v2|success|completed|failed|traceback|error|exception'
```

### Signs it is working

You should see messages like:

```text
Dropbox reported changes. Checking changed files.
Detected updated target PDF: /hourlyreport.pdf
Running workflow: /home/dhopkins/apps/ed-flow-2023/scripts/run_ed_flow_update.sh
=== START: python scripts/get_current.py ===
=== DONE: python scripts/get_current.py ===
...
=== START: python scripts/validate_forecast_inputs.py ===
[PASS] ED freshness: ...
[PASS] ED hourly continuity: ...
[PASS] Staffing horizon coverage: ...
[PASS] Weather horizon coverage: ...
=== DONE: python scripts/validate_forecast_inputs.py ===
...
=== START (additive): python scripts/hourly_forecast_v2.py ===
Wrote and uploaded forecast-v2.csv (288 rows)
=== DONE: python scripts/hourly_forecast_v2.py ===
Workflow completed successfully.
```

If only v2 fails, you may instead see:

```text
=== WARNING: additive step failed (...): python scripts/hourly_forecast_v2.py ===
Workflow completed successfully.
```

That is intentional: v2 is currently isolated from the established production outputs.

It is also normal to see:

```text
No Dropbox changes during longpoll window.
Dropbox changed, but hourlyreport.pdf did not change.
```

Those mean the watcher is alive and correctly ignoring irrelevant Dropbox activity.

### Check what command the service runs

```bash
systemctl cat ed-flow-dropbox-watcher.service
```

### Restart the watcher if needed

```bash
sudo systemctl restart ed-flow-dropbox-watcher.service
systemctl status ed-flow-dropbox-watcher.service --no-pager -l
```

### Follow logs live

```bash
sudo journalctl -u ed-flow-dropbox-watcher.service -f
```

---

## 2. Weather update every 4 hours

### Purpose

Runs:

```bash
/home/dhopkins/apps/ed-flow-2023/scripts/update_weather.py
```

every 4 hours and uploads:

```text
weather.csv
```

The hourly wrapper does not rerun this script. Instead, `validate_forecast_inputs.py` verifies that the existing `weather.csv` covers all 24 forecast hours before either production forecast is allowed to use it.

### Check timer schedule

```bash
systemctl list-timers ed-flow-weather.timer --no-pager
```

Expected:

- A recent `LAST` run
- A future `NEXT` run
- Roughly 4 hours between runs

### Check timer status

```bash
systemctl status ed-flow-weather.timer --no-pager -l
```

Expected:

```text
Active: active (waiting)
```

### Check latest service run

```bash
systemctl status ed-flow-weather.service --no-pager -l
```

Expected after a successful run:

```text
code=exited, status=0/SUCCESS
```

### Check recent logs

```bash
sudo journalctl -u ed-flow-weather.service --since "7 days ago" --no-pager
```

Filtered version:

```bash
sudo journalctl -u ed-flow-weather.service --since "7 days ago" --no-pager \
  | grep -Ei 'uploaded|weather|success|failed|traceback|error|exception|ModuleNotFoundError'
```

### Signs it is working

You should see output like:

```text
Coordinates: ...
Elevation: ...
Timezone: ...
uploaded as b'weather.csv'
```

### Test-run manually

```bash
sudo systemctl start ed-flow-weather.service
sudo journalctl -u ed-flow-weather.service -n 100 --no-pager
```

### Restart/reload timer if needed

```bash
sudo systemctl daemon-reload
sudo systemctl restart ed-flow-weather.timer
systemctl list-timers ed-flow-weather.timer --no-pager
```

---

## 3. Daily anomaly detection at 6:00 AM

### Purpose

Runs:

```bash
/home/dhopkins/apps/ed-flow-2023/scripts/anomaly_detection.py
```

every morning at 6:00 AM local server time.

### Check server timezone

```bash
timedatectl
```

Expected timezone should be Eastern, for example:

```text
Time zone: America/Toronto
```

or:

```text
Time zone: America/New_York
```

Note: in summer, 6:00 AM Eastern local time is 6:00 AM EDT, not EST.

### Check timer schedule

```bash
systemctl list-timers ed-flow-anomaly.timer --no-pager
```

Expected:

- `NEXT` should show the next 6:00 AM run
- `LAST` should show the previous morning’s run once it has run at least once

### Check timer status

```bash
systemctl status ed-flow-anomaly.timer --no-pager -l
```

Expected:

```text
Active: active (waiting)
```

### Check latest service run

```bash
systemctl status ed-flow-anomaly.service --no-pager -l
```

Expected after a successful run:

```text
code=exited, status=0/SUCCESS
```

### Check recent logs

```bash
sudo journalctl -u ed-flow-anomaly.service --since "7 days ago" --no-pager
```

Filtered version:

```bash
sudo journalctl -u ed-flow-anomaly.service --since "7 days ago" --no-pager \
  | grep -Ei 'uploaded|anomaly|alert|success|failed|traceback|error|exception|ModuleNotFoundError'
```

### Test-run manually

```bash
sudo systemctl start ed-flow-anomaly.service
sudo journalctl -u ed-flow-anomaly.service -n 100 --no-pager
```

### Restart/reload timer if needed

```bash
sudo systemctl daemon-reload
sudo systemctl restart ed-flow-anomaly.timer
systemctl list-timers ed-flow-anomaly.timer --no-pager
```

---

## 4. Check all ED flow logs together

### Last 24 hours

```bash
for u in ed-flow-dropbox-watcher.service ed-flow-weather.service ed-flow-anomaly.service; do
  echo
  echo "===== $u ====="
  sudo journalctl -u "$u" --since "24 hours ago" --no-pager \
    | grep -Ei 'uploaded|detected|workflow|START:|DONE:|WARNING:|validation|forecast-v2|completed|success|failed|traceback|error|exception|ModuleNotFoundError|No Dropbox changes|hourlyreport' \
    || true
done
```

### Last 7 days

```bash
for u in ed-flow-dropbox-watcher.service ed-flow-weather.service ed-flow-anomaly.service; do
  echo
  echo "===== $u ====="
  sudo journalctl -u "$u" --since "7 days ago" --no-pager \
    | grep -Ei 'uploaded|detected|workflow|START:|DONE:|WARNING:|validation|forecast-v2|completed|success|failed|traceback|error|exception|ModuleNotFoundError|hourlyreport' \
    || true
done
```

---

## 5. Common interpretations

### `inactive (dead)` for weather/anomaly service

Usually normal. These are `oneshot` services. They run, exit, and remain inactive until the next timer trigger.

What matters is:

```text
status=0/SUCCESS
```

### `active (running)` for Dropbox watcher

This is expected. It is a persistent longpoll watcher.

### `No Dropbox changes during longpoll window`

Normal. The watcher is alive and waiting.

### `Dropbox changed, but hourlyreport.pdf did not change`

Normal. Dropbox had some activity, but not the target PDF.

### `Forecast input validation failed`

Do not bypass the gate just to obtain a forecast. Identify which input failed:

- ED freshness / continuity → inspect `get_current.py` and the source PDF
- staffing horizon coverage → inspect ShiftAdmin refresh/output
- weather horizon coverage → inspect `ed-flow-weather.timer` and `weather.csv`

The gate exists to prevent stale or incomplete inputs from silently generating an operational forecast.

### `WARNING: additive step failed ... hourly_forecast_v2.py`

The established production pipeline completed, but `forecast-v2.csv` was not refreshed. Check the preceding v2 traceback/error. The next successful wrapper run will retry v2 automatically.

### `ModuleNotFoundError`

The service is using a Python environment that is missing a package. Check:

```bash
/home/dhopkins/apps/ed-flow-2023/.venv/bin/python -m pip show PACKAGE_NAME
```

Install into the same venv:

```bash
cd /home/dhopkins/apps/ed-flow-2023
/home/dhopkins/apps/ed-flow-2023/.venv/bin/python -m pip install PACKAGE_NAME
```

### `status=1/FAILURE`

Check the full logs:

```bash
sudo journalctl -u SERVICE_NAME -n 200 --no-pager
```

Example:

```bash
sudo journalctl -u ed-flow-anomaly.service -n 200 --no-pager
```

---

## 6. Useful service files

View the current systemd definitions:

```bash
systemctl cat ed-flow-dropbox-watcher.service
systemctl cat ed-flow-weather.service
systemctl cat ed-flow-weather.timer
systemctl cat ed-flow-anomaly.service
systemctl cat ed-flow-anomaly.timer
```

Edit a service or timer:

```bash
sudo systemctl edit --full SERVICE_OR_TIMER_NAME
```

After edits:

```bash
sudo systemctl daemon-reload
sudo systemctl restart SERVICE_OR_TIMER_NAME
```

---

## 7. Manual script checks

Run from the project directory:

```bash
cd /home/dhopkins/apps/ed-flow-2023
```

### Full Dropbox-triggered workflow

```bash
/home/dhopkins/apps/ed-flow-2023/scripts/run_ed_flow_update.sh
```

Because the wrapper loads `.env`, this is the preferred manual end-to-end test.

### Input gate only

```bash
source .venv/bin/activate
python scripts/validate_forecast_inputs.py
```

### Forecast v2 only

```bash
source .venv/bin/activate
set -a
source .env
set +a
CHRONOS_HOURLY_ENABLE_WEATHER_ROUTING=0 python scripts/hourly_forecast_v2.py
```

Expected successful final message:

```text
Wrote and uploaded forecast-v2.csv (288 rows)
```

### METAR refresh only

```bash
source .venv/bin/activate
python scripts/update_metar.py
```

### Weather script

```bash
/home/dhopkins/apps/ed-flow-2023/.venv/bin/python scripts/update_weather.py
```

### Anomaly detection script

```bash
/home/dhopkins/apps/ed-flow-2023/.venv/bin/python scripts/anomaly_detection.py
```

---

## 8. Expected automation summary

| Automation | Unit | Schedule/Trigger | Healthy state |
|---|---|---|---|
| Dropbox hourly report watcher | `ed-flow-dropbox-watcher.service` | Dropbox longpoll detects `/hourlyreport.pdf` changes; runs legacy + on-call + additive v2 forecasts | `active (running)` |
| Weather update | `ed-flow-weather.timer` → `ed-flow-weather.service` | Every 4 hours | Timer `active (waiting)`, service last run `0/SUCCESS` |
| Anomaly detection | `ed-flow-anomaly.timer` → `ed-flow-anomaly.service` | Daily at 6:00 AM local Eastern time | Timer `active (waiting)`, service last run `0/SUCCESS` |
