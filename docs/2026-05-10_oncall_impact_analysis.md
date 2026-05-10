# On-Call Impact Analysis & Counterfactual Forecasting
**Date:** 2026-05-10

## Overview
This project integrates high-fidelity "Busy On-Call" labels into the ED flow forecasting pipeline to enable "What-If" analysis. The goal is to quantify the impact of calling in an on-call physician on the Total Bed-Stay (TBS) metrics in the Emergency Department.

## 1. On-Call Labeling Logic
The labels were derived from the `on-call-used-project` experiment, which identifies true "busy call-ins" by:
- Matching physician billing timestamps with on-call schedules.
- Excluding billing activity that occurs during regular scheduled shifts.
- Filtering out "spillover" (isolated one-hour hits after a shift) and "singletons" (single-patient episodes).
- Inferring gaps between billing hits to create continuous "busy blocks."

The resulting dataset provides a binary label: `oncall_busy` (1 if the physician was called in for volume, 0 otherwise).

## 2. Forecasting Implementation (`chronos_forecast_v2.py`)
We transitioned from a purely descriptive flow forecast to a counterfactual model using the **Chronos-2** pipeline.

### Key Technical Changes:
- **Feature Integration**: Added `oncall_busy` as a covariate (feature) rather than a target.
- **Dtype Alignment**: Ensured `oncall_busy` is cast to `float64` across both historical and future dataframes to prevent pipeline crashes.
- **Hardware Acceleration**: Switched `device_map` to `"cuda"` for GPU-accelerated inference.
- **Relative Pathing**: Updated data loading to use relative paths for better portability.

## 3. Counterfactual Scenarios
The script runs two parallel forecasts for the next 24 hours to isolate the effect of staffing:

| Scenario | `oncall_busy` Assumption | Goal |
| :--- | :--- | :--- |
| **Baseline** | `0` for all 24 hours | Predict TBS if no extra help is called in. |
| **Intervention** | `1` for first 8 hours, then `0` | Predict TBS if on-call help is deployed immediately. |

## 4. Output & PowerBI Integration
The script produces `oncall_impact_forecast.csv`, which includes:
- `tbs_no_oncall`: Predicted TBS (Baseline).
- `tbs_with_oncall`: Predicted TBS (Intervention).
- `tbs_reduction`: The delta (`no_oncall` - `with_oncall`), providing a direct measure of the "TBS saved" by the on-call physician.

## 5. Future Directions
- **Target Forecasting**: Exploring the ability to forecast the `oncall_busy` column itself to predict *when* a call-in is likely to be needed.
- **Scenario Optimization**: Testing different intervention windows (e.g., 4h vs 8h) to find the optimal staffing response.
