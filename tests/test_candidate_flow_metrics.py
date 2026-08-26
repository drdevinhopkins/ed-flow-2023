import pandas as pd

from candidate_flow_metrics import (
    CANDIDATE_TARGETS,
    WORKUP_DELAY_COMPONENTS,
    add_candidate_metrics,
    reset_aware_increment,
)


def test_reset_aware_increment_handles_counter_reset_and_gap():
    ds = pd.Series(pd.to_datetime([
        "2026-01-01 00:00",
        "2026-01-01 01:00",
        "2026-01-01 02:00",
        "2026-01-01 03:00",
        "2026-01-01 05:00",
    ]))
    counter = pd.Series([10, 12, 15, 1, 4], dtype="float64")

    result = reset_aware_increment(counter, ds)

    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 2
    assert result.iloc[2] == 3
    assert result.iloc[3] == 1
    assert pd.isna(result.iloc[4])


def test_candidate_metric_derivations_are_explicit_burdens_not_unique_patients():
    frame = pd.DataFrame({
        "ds": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 01:00"]),
        "Inflow_Total": [7, 9],
        "INFLOW_STRETCHER": [3, 4],
        "INFLOW_AMBULANCES": [1, 2],
        "CUM_ADMREQ": [5, 7],
    })
    for idx, column in enumerate(WORKUP_DELAY_COMPONENTS, start=1):
        frame[column] = [idx, idx]

    result = add_candidate_metrics(frame)

    assert set(CANDIDATE_TARGETS).issubset(result.columns)
    assert pd.isna(result.loc[0, "AdmissionRequests_New"])
    assert result.loc[1, "AdmissionRequests_New"] == 2
    assert result.loc[0, "Workup_Delay_Burden"] == sum(range(1, 9))
