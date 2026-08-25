This file documents the assistant-authored ED forecast blurb trigger.

The actionable request lives in `.github/ed_forecast_blurb_request.json` on the dedicated `ed-forecast-blurb-trigger` branch. Updating that JSON triggers GitHub Actions to append the exact row to Dropbox. The worker validates the CSV schema and de-duplicates on `generated_at_local` before a revision-aware Dropbox update.
