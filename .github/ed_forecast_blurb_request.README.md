This file documents the assistant-authored ED forecast blurb trigger.

The actionable request lives in `.github/ed_forecast_blurb_request.json` on the dedicated `ed-forecast-blurb-trigger` branch. Updating that JSON triggers GitHub Actions to append the exact row to Dropbox. The worker validates the CSV schema, de-duplicates on `blurb_id` (with `generated_at_local` as a fallback), and performs a revision-aware Dropbox update.

Delivery metadata is written with every new row so Fabric / Power Automate do not need to reinterpret the forecast:

- `blurb_id`: stable unique ID, normally `YYYYMMDD-HHMM` in America/Montreal time.
- `send_recommended`: boolean gate for Teams delivery.
- `send_reason`: one of `ROUTINE`, `ONCALL_ALERT`, `ROUTINE_ONCALL`, or `NONE`.

Routine Teams sends are 07:45, 11:45, 15:45, and 19:45 local time. Outside those times, send an event-driven on-call alert when the recommendation newly becomes `CONSIDER` or `USE`, or escalates from `CONSIDER` to `USE`. Do not repeat the same active on-call recommendation every hour. If a routine blurb also contains a new/escalated on-call recommendation, use `ROUTINE_ONCALL` so Power Automate sends only one message.

The append worker can migrate the legacy six-column Dropbox CSV in place on the first new append; historical rows are preserved and their delivery metadata is left blank because prior send state is unknown.
