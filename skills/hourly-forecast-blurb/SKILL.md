# Hourly ED Flow Forecast Blurb

## When to use this skill

Use this skill when the user asks for a short operational interpretation of the current JGH ED hourly flow forecast, including phrases such as:

- `flow blurb`
- `hourly flow blurb`
- `summarize the forecast`
- `what does the ED flow forecast look like?`
- `how are we looking tonight?`
- `how are we looking overnight?`
- `give me the operational forecast`

The goal is a concise, clinically useful situational-awareness summary, not a technical model report.

## Repository and input

Work from the repository root, normally:

```bash
/home/dhopkins/apps/ed-flow-2023
```

Preferred forecast input:

```text
forecast-v2.1.csv
```

The current generator is:

```text
scripts/hourly_forecast_v2_1.py
```

If multiple forecast files exist, use the newest successfully generated valid `forecast-v2.1.csv` and report/retain its `forecast_origin` and `generated_at_utc` internally.

Do not silently use an older forecast if the latest one is missing, stale, or incomplete.

## Targets

Interpret the following targets:

| Target | Operational meaning |
|---|---|
| `Total_TBS` | Total patients to be seen across the ED |
| `POD_TBS` | Patients to be seen in POD |
| `Vertical_TBS` | Patients to be seen in Vertical |
| `TTStr` | Total stretcher occupancy / stretcher burden |
| `Overflow` | Overflow burden |
| `WAITINGADM` | Patients waiting for admission |
| `TRG_HALLWAY1` | Triage hallway occupancy |
| `TRG_HALLWAY_TBS` | Patients to be seen in the triage hallway |

Use human-readable names in the final prose.

## Validate before interpreting

The selected file should include at least:

```text
ds
target_name
actual
forecast
forecast_lower
forecast_upper
row_type
horizon_hour
forecast_origin
generated_at_utc
forecast_anomaly
scenario
baseline_forecast
feature_effect
explanation_family
explanation_direction
explanation_text
weather_routing_enabled
```

Before writing a blurb, confirm:

- future `row_type == "forecast"` rows exist;
- all expected targets are represented;
- the 24-hour horizon is sufficiently complete for the requested interpretation;
- there are no duplicate future `ds` + `target_name` rows;
- required forecast values are present;
- timestamps are plausibly current.

Present times in `America/Montreal` local time.

If validation fails, do not fabricate a summary. State clearly why a current blurb cannot be produced.

## Compute first

Do not give the raw CSV directly to the prose LLM and ask it to discover trends.

Use Python/pandas or equivalent deterministic code to calculate the facts first.

### Current state

For each target, use the latest observed value at or before `forecast_origin`.

Record:

- latest observed value;
- timestamp;
- approximate change over the previous 2–4 hours when enough history exists.

### Forecast windows

Calculate these windows:

```text
1–4 hours
5–8 hours
9–12 hours
13–24 hours
```

For each target/window calculate:

- first forecast;
- final forecast;
- minimum;
- maximum;
- time of maximum;
- change from the latest observed value;
- broad direction: rising, falling, or stable.

Do not turn tiny numerical movements into meaningful trends. As a practical default, changes under about one patient are usually operationally stable unless context makes them important.

For high-valued metrics such as stretcher burden, consider relative as well as absolute change before using strong wording.

### Important peaks

Prioritize narrative attention approximately in this order:

1. `Total_TBS`
2. `TTStr`
3. `WAITINGADM`
4. `POD_TBS`
5. `Vertical_TBS`
6. `TRG_HALLWAY_TBS`
7. `TRG_HALLWAY1`
8. `Overflow`

This is not a rigid ranking. A large or anomalous movement in a lower-ranked metric may be more important.

Identify the most relevant peaks over the next 12 and 24 hours, including local clock time.

### Forecast anomalies

Inspect `forecast_anomaly`.

When future rows are flagged, identify:

- target;
- time/window;
- approximate magnitude and direction when determinable.

Describe these carefully, for example:

> unusually high for the model's historical expectation

Do not equate a model anomaly with a guaranteed crisis.

### Uncertainty

`forecast_lower` and `forecast_upper` are the model's 0.2 and 0.8 quantiles.

Do not call them a 95% confidence interval.

Normally omit uncertainty from the routine blurb unless it changes the operational interpretation. When useful, phrase it as a model percentile range, e.g.:

> The central forecast is about 18, with a model 20th–80th percentile range of roughly 14–22.

## Explainability

For future rows:

```text
feature_effect = forecast - baseline_forecast
```

This is an **associational routed-scenario contrast versus the history-only baseline, not a causal effect**.

Inspect:

```text
scenario
explanation_family
feature_effect
feature_effect_pct
explanation_direction
explanation_text
```

Only surface feature effects that are large enough to matter operationally.

Prefer absolute patient differences to percentages when the baseline is small.

Good examples:

> Staffing context is pulling the 9–12 hour POD forecast about 2 patients lower than the history-only baseline.

> Calendar context adds roughly 3 patients to Total TBS over the next few hours versus the history-only baseline.

Bad examples:

> Staffing causes two fewer POD patients.

> Dr. X is making the department faster.

Do not name individual physicians in the routine blurb even if physician-specific engineered features contribute to `staffing_structure_effects`. Use `staffing context` or `staffing-structure context` unless the user explicitly requests physician-level analysis.

If `weather_routing_enabled` is false, do not describe weather as a model driver.

Aggregate repeated adjacent feature effects into one statement instead of listing each hour.

## Cross-metric interpretation

Prefer coherent operational interpretation over a metric dump.

Useful patterns:

- Total TBS falling + `WAITINGADM` rising → front-end pressure is easing but boarding pressure is building.
- Stable Total TBS + rising `TTStr` → overall demand is stable but the ED is becoming more stretcher-heavy.
- Falling POD + rising Vertical → burden is shifting from POD toward Vertical.
- Rising triage hallway metrics + stable overall TBS → front-end congestion is worsening despite stable overall demand.
- Falling Total TBS + persistently high `WAITINGADM` → demand is easing but downstream congestion remains.

If signals disagree, preserve the nuance rather than forcing one overall direction.

## Create a compact payload

After deterministic calculation, create a compact structured object for the prose model rather than sending the full CSV.

Example shape only:

```json
{
  "forecast_origin_local": "2026-08-26 19:00",
  "current": {
    "Total TBS": 18,
    "POD TBS": 5,
    "Vertical TBS": 4,
    "stretcher burden": 41,
    "waiting admission": 12,
    "triage hallway TBS": 2
  },
  "next_4h": {
    "Total TBS": {
      "start": 18,
      "end": 23,
      "peak": 25,
      "peak_time": "22:00",
      "trend": "rising"
    }
  },
  "next_8h": {},
  "next_12h": {},
  "next_24h": {},
  "anomalies": [],
  "important_feature_effects": []
}
```

All numbers above are illustrative only. Never reuse example values in a real blurb.

## Local LLM

Use the locally hosted OpenAI-compatible LLM for prose when available.

Prefer an explicitly configured endpoint:

```bash
ED_FLOW_LLM_BASE_URL
```

If unset, a common local endpoint on `jgh000533svaps` is:

```text
http://127.0.0.1:8082/v1
```

Discover the served model instead of hard-coding the model name:

```bash
curl -s "${ED_FLOW_LLM_BASE_URL:-http://127.0.0.1:8082/v1}/models"
```

Use a low temperature, approximately `0.2`, for reproducible operational prose.

Do not send patient-identifiable information. This workflow should contain aggregate operational data only.

If the local LLM is unavailable but deterministic analysis succeeded, write the blurb directly from the structured summary.

## Prose-model system prompt

Use the following or an equivalent system prompt:

```text
You are writing a concise hourly operational forecast for the Jewish General Hospital Emergency Department.

You will receive a structured summary generated deterministically from an hourly forecasting model. Do not recalculate or invent numbers. Use only the supplied facts.

Explain:
- the current ED flow state;
- whether pressure is likely to rise, fall, or remain stable over the next 4–12 hours;
- the timing and magnitude of the most important expected peaks;
- which operational areas are driving the pattern;
- any genuinely important forecast anomalies;
- one or two meaningful model-context effects, if supplied.

Prioritize Total TBS, stretcher burden, waiting-for-admission burden, POD/Vertical distribution, and triage hallway pressure.

Write for an emergency physician or ED command-centre reader, not a data scientist.

Style:
- 2 short paragraphs maximum;
- usually 70–130 words;
- lead with the operational takeaway;
- use Montreal local clock times;
- round patient counts sensibly, usually to whole patients;
- use about, roughly, or around when appropriate;
- avoid alarmist language;
- avoid generic filler;
- do not list every metric;
- do not make staffing recommendations unless explicitly requested;
- do not make causal claims from feature effects;
- never invent thresholds;
- never say confidence interval for the supplied 20th–80th percentile bounds;
- do not mention Chronos, CSV columns, routing versions, Python, or implementation details unless asked.

Feature effects compare a routed contextual forecast with a history-only model baseline. They are associational scenario contrasts, not causal effects.

Return only the final blurb. No heading, bullets, preamble, or explanation.
```

Send the compact deterministic JSON summary as the user message.

## Verify the generated prose

The deterministic summary is the source of truth.

Before returning or publishing the blurb, verify every:

- patient count;
- local clock time;
- direction of change;
- peak;
- anomaly claim;
- baseline comparison.

Correct or regenerate the prose if it:

- invents a number;
- swaps POD and Vertical;
- makes a causal claim from a feature effect;
- attributes effects to weather when weather routing is disabled;
- names an individual physician without being asked;
- calls the 20th–80th percentile bounds a 95% confidence interval;
- recommends calling in staff or changing assignments without being asked;
- exaggerates a small movement into a surge or crisis.

## Preferred output structure

A routine blurb should usually follow:

```text
[Overall direction + key near-term pressure.] [Most important peak and timing.] [Where the burden is concentrated.]

[What happens later / whether pressure eases or persists.] [Optional important anomaly or model-context effect.]
```

If the forecast is stable, say so plainly. Do not manufacture drama.

Example style only:

```text
ED pressure is forecast to build through the evening, with Total TBS rising from the high teens to the mid-20s and peaking around 22:00. The increase is expected to be concentrated mainly in POD and Vertical, while stretcher burden remains elevated and the number waiting for admission changes relatively little.

Pressure should begin easing overnight, although the triage hallway remains busier than its current level for several hours. Staffing context is modestly lowering the POD forecast versus the history-only baseline; this is a model association rather than a causal staffing effect.
```

Never reuse these example values in a real forecast.

## Output

Unless the user asks for technical detail, return only the final blurb.

Do not include raw CSV, JSON, code, or a 24-hour table in the routine response.

If the user asks why the forecast looks this way, then provide the supporting feature effects, uncertainty, routing, and relevant numeric comparisons separately.

## Core rule

**Compute first, narrate second.**

LLM prose is only a presentation layer. Every factual statement must be grounded in deterministic analysis of the current forecast output.
