# Anomaly refresh recovery

The hourly input gate requires anomaly intervals for every hour in the next
24 hours. A failed gate must not be bypassed or repaired by forward-filling ranges.

Run `python scripts/anomaly_detection.py --hourly-only` from the repository root
in the server's configured Python environment. It reads the existing `.env`
credentials and publishes to the existing Dropbox destinations. Omit
`--hourly-only` to also refresh the existing daily inflow forecast.

Alternatively, run **Manual anomaly refresh** from GitHub Actions once this
workflow is on main. It only refreshes hourly anomaly references, then runs the
full forecast input validator. After it succeeds, rerun **Hourly update** to
validate current inputs and trigger the normal downstream forecasts.

The generator requires fresh ED data and 48 hours of complete intervals at
publication time, retaining headroom for the daily refresh cadence. It still
models 72 hours beyond training history. Required target failures stop publication;
other failed columns are reported. Upload errors propagate, and the uploaded
bytes are downloaded from Dropbox and checked against the generated file.

On October 25, 2026, GitHub's Daily Training workflow was observed to be manually
disabled (last run June 27). Do not assume it is the active daily refresh or
re-enable it without reviewing the hospital server's schedule. The manual
workflow adds no recurring schedule. Inspect the server's anomaly job logs and
timer separately to establish why the daily publication stopped.
