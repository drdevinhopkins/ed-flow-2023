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
  | grep -Ei 'started|running|dropbox|cursor|longpoll|change|changed|hourlyreport|pdf|workflow|success|completed|failed|traceback|error|exception'
```

### Signs it is working

You should see messages like:

```text
Dropbox reported changes. Checking changed files.
Detected updated target PDF: /hourlyreport.pdf
Running workflow: /home/dhopkins/apps/ed-flow-2023/scripts/run_ed_flow_update.sh
Workflow completed successfully.
```

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
    | grep -Ei 'uploaded|detected|workflow|completed|success|failed|traceback|error|exception|ModuleNotFoundError|No Dropbox changes|hourlyreport' \
    || true
done
```

### Last 7 days

```bash
for u in ed-flow-dropbox-watcher.service ed-flow-weather.service ed-flow-anomaly.service; do
  echo
  echo "===== $u ====="
  sudo journalctl -u "$u" --since "7 days ago" --no-pager \
    | grep -Ei 'uploaded|detected|workflow|completed|success|failed|traceback|error|exception|ModuleNotFoundError|hourlyreport' \
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

### Dropbox-triggered workflow script

```bash
/home/dhopkins/apps/ed-flow-2023/scripts/run_ed_flow_update.sh
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
| Dropbox hourly report watcher | `ed-flow-dropbox-watcher.service` | Dropbox longpoll detects `/hourlyreport.pdf` changes | `active (running)` |
| Weather update | `ed-flow-weather.timer` → `ed-flow-weather.service` | Every 4 hours | Timer `active (waiting)`, service last run `0/SUCCESS` |
| Anomaly detection | `ed-flow-anomaly.timer` → `ed-flow-anomaly.service` | Daily at 6:00 AM local Eastern time | Timer `active (waiting)`, service last run `0/SUCCESS` |
