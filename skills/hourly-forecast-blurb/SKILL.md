---
name: hourly-forecast-blurb
description: Use when asked for a JGH ED flow blurb or operational interpretation of the current hourly forecast. Validate current-hour inputs, compute deterministic operational facts, assess on-call need, and produce a concise clinically useful handoff without reconstructing canonical metrics.
version: 1.0.0
author: ed-flow-2023
platforms: [linux]
metadata:
  hermes:
    tags: [ed-flow, forecasting, operations, jgh]
    category: ed-flow
    requires_toolsets: [terminal]
---

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

## Repository and inputs

Work from the repository root, normally:

```bash
/home/dhopkins/apps/ed-flow-2023
```

Inspect the newest operational inputs, especially:

```text
current.csv
forecast-v2.1.csv
oncall_need_probability.csv
oncall_impact_summary.csv
forecast_variable_effects_hourly.csv
blurb_reference_stats.json
hourly_forecast_blurbs.csv
```

The current forecast generator is `scripts/hourly_forecast_v2_1.py`.

## Readiness gate

Determine the expected data hour from the current `America/Montreal` clock hour. Before composing anything, require all inputs needed for the blurb to be consistently refreshed for that hour.

At minimum:

- `current.csv` contains the expected hour;
- `forecast-v2.1.csv` has `forecast_origin` equal to that hour;
- on-call need and impact files are refreshed for that same hour when expected;
- explainability/staffing inputs used in prose are not stale relative to that hour;
- the hour's `blurb_id` is not already present in `hourly_forecast_blurbs.csv`.

Do not mix hours. Do not fall back to the previous hour merely to produce a message. If readiness fails, do not generate/publish a blurb.

## Canonical targets — critical

For metrics already represented as targets in `forecast-v2.1.csv`, the target rows are the source of truth for both observed/current values and forecasts:

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

**Never manually reconstruct `Total_TBS`, `POD_TBS`, `Vertical_TBS`, or `Overflow` from the wide `current.csv` row.** Do not sum raw zone columns to reproduce a metric that the pipeline already supplies. Do not add occupancy fields such as `QTRACK1`, `RESUS`, `POD_T`, `VERTSTRET`, `AMBVERT1`, or similarly named fields into TBS totals.

This rule exists because the raw report contains many adjacent, similarly named columns and manual parsing can silently shift columns or mix occupancy with TBS. If a canonical target looks surprising, verify the target row; do not replace it with an ad-hoc calculation.

Use `current.csv` for the readiness gate and for operational facts that are not already canonical forecast targets.

## Validate before interpreting

Before writing a blurb, confirm future forecast rows exist, expected targets are represented, required values are present, timestamps are current, and there are no duplicate future `ds + target_name` rows. Present all times in `America/Montreal`.

If validation fails, do not fabricate a summary.

## Compute first

Do not give the raw CSV directly to a prose LLM and ask it to discover trends. Calculate the facts deterministically first.

For each canonical target, use the observed row at `forecast_origin` as the current value. Calculate the relevant near-term direction, maxima/minima, peak time, and change from current over useful windows (normally 1–4, 5–8, 9–12, and 13–24 hours).

Do not turn tiny movements into meaningful trends. Changes under about one patient are usually operationally stable unless context makes them important.

## Current crowding

Express `TTStr` as stretcher occupancy percentage using nominal capacity 53:

```text
stretcher occupancy % = TTStr / 53 * 100
```

Round sensibly for prose.

Interpret `Overflow` operationally rather than as an abstract count:

- about 0–16: generally within the comfortably usable first two overflow rooms;
- around the mid-teens: roughly two overflow rooms in use;
- just above 16: roughly the first two overflow rooms plus minor spillover into prepod/additional overflow space;
- above ~16: increasingly dependent on staffing/opening rooms 3–5, with greater risk of prepod accumulation;
- ~30–40: most/all nominal overflow-room capacity is being used;
- >40: beyond practical overflow capacity.

When useful, translate the forecast into that physical meaning. For example, prefer:

> Overflow in the mid-teens would mean roughly two overflow rooms in use, with possibly a small spillover into prepod.

rather than simply:

> Overflow stays in the mid-teens.

Avoid the phrase `overcapacity rooms`; call them `overflow rooms`.

## Boarders / WAITINGADM

Boarders contribute to stretcher occupancy and overflow but are primarily the responsibility of admitting services rather than the ED flow team. Do not make boarder count a routine focus or ED action item. Mention it mainly when needed to explain persistent occupancy/overflow despite improving ED workload, or when exceptionally important.

## Midnight handoff

When midnight is within the forecast horizon, especially in afternoon/evening blurbs, include forecast `Total_TBS` at midnight when it helps the two night physicians anticipate workload.

Use `blurb_reference_stats.json` to classify midnight Total TBS against the rolling two-year actual midnight distribution. Prefer `by_prior_evening_day` matching the current evening's weekday; fall back to weekday/weekend, then overall. Translate this to `light`, `typical`, `heavier-than-usual`, or `very heavy`. Never expose percentile jargon in the routine blurb.

## Vertical versus POD

During afternoon/evening, compare canonical `Vertical_TBS` and `POD_TBS` current and near-term forecasts. Vertical is normally substantially busier than POD, so do not recommend redeployment merely because Vertical > POD.

Use `blurb_reference_stats.json` `evening_vertical_vs_pod` to decide whether the imbalance is unusually severe and persistent. If POD itself is under unusual pressure or forecast to worsen substantially, do not strip POD coverage unless another suitable overlap physician is available. When clearly actionable and staffing permits, use A2 on weekdays and Y5 on weekends when naming the orange evening POD shift; L1 overlap can also be suggested when appropriate.

## On-call

Assess on-call internally using both calibrated need probability and modeled impact. The outward conclusion should remain simple: `USE`, `CONSIDER`, `NOT INDICATED`, or `NO CLEAR RECOMMENDATION`.

Do not recommend on-call simply because the ED is crowded.

When on-call is not indicated now **and** the available 4/6/8-hour calibrated need remains low with no meaningful modeled benefit from activation, explicitly give the useful forward-looking reassurance:

> On-call is not currently needed and, based on how the day is shaping up, is unlikely to be required this evening.

Use `tonight`/`later today` instead when that better matches the clock. Do not make this forward-looking claim if longer-horizon need is meaningfully elevated, mixed, stale, unavailable, or the impact model suggests possible benefit. In those cases, stop at `On-call is not currently needed` or use the appropriate stronger recommendation.

## Explainability and physician team strength

For future rows, `feature_effect = forecast - baseline_forecast` is an associational routed-scenario contrast, not a causal effect. Only surface effects that are operationally meaningful and directionally consistent.

If physician/staffing effects are meaningfully large and consistent, the blurb may say the team looks a bit stronger than usual, roughly neutral, or a bit weaker than usual for flow. Do not name or rank individual physicians and do not imply causality. Omit this if small, inconsistent, stale, or uncertain.

If weather routing is disabled, do not describe weather as a model driver.

## Core narrative

Prefer a quick handoff between ED flow physicians. Use:

```text
NOW → WHERE WE ARE HEADING / PEAK → MIDNIGHT HANDOFF WHEN RELEVANT → ACTION
```

Lead with direction: worsening, stable, improving, or likely past the peak. Use only the few numbers that tell the operational story. Usually write 2–4 sentences and stay under roughly 90 words.

Prefer familiar terms: TBS, POD, vertical, prepod, overflow, stretcher occupancy, night docs, on-call. Avoid forecasting/data-science jargon.

Do not dump metrics. If multiple metrics tell the same story, choose the most operationally useful ones.

## Teams delivery metadata

Every successful hourly blurb is retained, but only selected rows should be sent to Teams.

Routine data hours are 07:00, 11:00, 15:00, and 19:00 America/Montreal. These always get `send_recommended=true` and `send_reason=ROUTINE`, unless they also contain a newly active/escalated on-call recommendation, in which case use `ROUTINE_ONCALL`.

Outside routine hours, recommend sending only when on-call newly becomes `CONSIDER` or `USE`, or escalates from `CONSIDER` to `USE`. Do not repeat an unchanged active recommendation every hour. Such event-driven sends use `ONCALL_ALERT`; otherwise use `NONE` and `send_recommended=false`.

`blurb_id` identifies the data hour as `YYYYMMDD-HH00` in America/Montreal.

## Publication

Do not write `hourly_forecast_blurbs.csv` directly. Publication should go through the repository append worker/request mechanism. The request payload must preserve the verified data hour, exact blurb, on-call recommendation/rationale, send metadata, and source/readiness status. If the append workflow fails, report the failure rather than claiming the row was appended.

## Verify the prose

Before publishing, verify every patient count, percentage, local time, direction, peak, operational translation, on-call statement, and staffing/explainability claim against the deterministic inputs.

Reject or correct prose that:

- reconstructs a canonical metric from raw columns;
- invents or misreads a number;
- swaps POD and Vertical;
- mixes occupancy fields with TBS fields;
- makes a causal claim from feature effects;
- attributes effects to disabled/stale inputs;
- names an individual physician;
- recommends staffing changes without adequate operational evidence;
- exaggerates a small movement;
- makes an evening on-call reassurance without supporting longer-horizon data.

## Core rule

**Compute first, narrate second. Canonical metrics stay canonical.**

LLM prose is only a presentation layer. Every factual statement must be grounded in deterministic analysis of the verified current-hour inputs.
