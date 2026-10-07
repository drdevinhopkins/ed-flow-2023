"""Daily target correctness at the producer and mixed-version consumer boundary."""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import daily_arrival_quality as quality

NOW = pd.Timestamp("2026-02-10T18:00:00Z")


def hourly(start="2026-02-01", days=10):
    ds = pd.date_range(start, periods=days * 24, freq="h")
    # Only observed hours on today's date belong in the snapshot.
    ds = ds[ds <= NOW.tz_convert("America/Montreal").tz_localize(None)]
    return pd.DataFrame({"ds": ds, "Inflow_Total": 2.0})


def legacy_daily(source):
    return source.groupby(source.ds.dt.normalize()).Inflow_Total.sum(min_count=1).rename(
        "Daily_Inflow_Total").reset_index()


def test_missing_hours_and_entire_days_are_blank_without_imputation():
    source = hourly()
    source = source.loc[~source.ds.eq(pd.Timestamp("2026-02-03T15:00"))
                        & ~source.ds.dt.normalize().eq(pd.Timestamp("2026-02-05"))]
    daily, metadata = quality.build_daily_inflow_outputs(source, now=NOW)
    assert daily.columns.tolist() == ["ds", "Daily_Inflow_Total"]
    lookup = daily.set_index("ds").Daily_Inflow_Total
    assert len(daily) == 9  # current Montreal day excluded
    assert pd.isna(lookup["2026-02-03"]) and pd.isna(lookup["2026-02-05"])
    assert lookup.dropna().eq(48).all()
    meta = metadata.set_index("ds")
    assert meta.loc["2026-02-03", "observed_inflow_sum"] == 46
    assert meta.loc["2026-02-03", "missing_clock_hours"] == "15"
    assert meta.loc["2026-02-05", "observed_rows"] == 0
    assert pd.isna(meta.loc["2026-02-05", "observed_inflow_sum"])
    assert not meta.loc["2026-02-10", "audit_eligible"]
    # CSV keeps the established schema and round-trips unverified targets as blank.
    reread = pd.read_csv(io.StringIO(daily.to_csv(index=False)))
    assert reread.Daily_Inflow_Total.isna().sum() == 2


@pytest.mark.parametrize("problem", ["duplicate", "duplicate_replacing_gap", "negative", "nan", "inf"])
def test_row_counts_alone_do_not_prove_completeness(problem):
    source = hourly()
    index = source.index[source.ds.eq(pd.Timestamp("2026-02-03T15:00"))][0]
    if problem.startswith("duplicate"):
        row = source.loc[[index]].copy()
        if problem == "duplicate_replacing_gap":
            source = source.loc[~source.ds.eq(pd.Timestamp("2026-02-03T16:00"))]
        source = pd.concat([source, row], ignore_index=True)
    else:
        source.loc[index, "Inflow_Total"] = {"negative": -1, "nan": np.nan, "inf": np.inf}[problem]
    daily, meta = quality.build_daily_inflow_outputs(source, now=NOW)
    assert daily.set_index("ds").loc["2026-02-03"].isna().all()
    assert not meta.set_index("ds").loc["2026-02-03", "audit_eligible"]


@pytest.mark.parametrize("day", ["2026-03-08", "2026-11-01"])
def test_dst_requires_verification_even_with_24_naive_clock_slots(day):
    source = pd.DataFrame({"ds": pd.date_range(day, periods=24, freq="h"), "Inflow_Total": 2})
    daily, meta = quality.build_daily_inflow_outputs(source, now=pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=2))
    assert daily.Daily_Inflow_Total.isna().all()
    assert meta.quality_status.eq("dst_requires_verification").all()


def test_leading_partial_and_legacy_sum_mismatch_are_rejected():
    source = hourly().iloc[1:].copy()
    legacy = legacy_daily(source)
    legacy.loc[legacy.ds.eq(pd.Timestamp("2026-02-04")), "Daily_Inflow_Total"] = 999
    verified, _ = quality.verified_daily_targets(legacy, source, now=NOW)
    result = verified.set_index("ds").Daily_Inflow_Total
    assert pd.isna(result["2026-02-01"]) and pd.isna(result["2026-02-04"])
    assert result.dropna().eq(48).all()


def test_consumer_checks_legacy_daily_against_hourly_without_sidecar():
    source = hourly().drop(index=62)
    daily = legacy_daily(source)
    class Dropbox:
        def __init__(self):
            self.downloads = []
        def files_download(self, path):
            self.downloads.append(path)
            frame = {quality.DAILY_PATH: daily, quality.HOURLY_PATH: source}[path]
            return None, SimpleNamespace(content=frame.to_csv(index=False).encode())
    dbx = Dropbox()
    verified, _ = quality.load_verified_daily(dbx, now=NOW)
    assert dbx.downloads == [quality.DAILY_PATH, quality.HOURLY_PATH]
    assert verified.Daily_Inflow_Total.isna().sum() == 1
    assert verified.attrs["target_quality_version"] == quality.QUALITY_VERSION
    quality.require_latest_completed_day(verified, now=NOW)
    broken = verified.copy()
    broken.loc[broken.ds.eq(pd.Timestamp("2026-02-09")), "Daily_Inflow_Total"] = np.nan
    with pytest.raises(ValueError, match="Latest completed Montreal day"):
        quality.require_latest_completed_day(broken, now=NOW)


def test_explanation_rejects_legacy_or_changed_target_context():
    history = pd.DataFrame({"ds": pd.date_range("2026-02-01", periods=9), "daily_visits": 48.0})
    forecast = pd.DataFrame({"history_days": [9] * 7, "target_quality_version": quality.QUALITY_VERSION,
                             "target_history_sha256": quality.target_history_fingerprint(history)})
    quality.verify_explanation_context(forecast, history)
    changed = history.copy()
    changed.loc[0, "daily_visits"] = 47
    with pytest.raises(ValueError, match="targets differ"):
        quality.verify_explanation_context(forecast, changed)
    with pytest.raises(ValueError, match="predates"):
        quality.verify_explanation_context(forecast.drop(columns="target_quality_version"), history)
    with pytest.raises(ValueError, match="length"):
        quality.verify_explanation_context(forecast, history.iloc[1:])


@pytest.mark.parametrize("problem", ["aware", "off_hour", "duplicate_daily"])
def test_source_contract_changes_fail_closed(problem):
    source = hourly()
    daily = legacy_daily(source)
    if problem == "aware":
        source["ds"] = source.ds.dt.tz_localize("America/Montreal")
    elif problem == "off_hour":
        source.loc[0, "ds"] += pd.Timedelta(minutes=5)
    else:
        daily = pd.concat([daily, daily.iloc[[0]]])
    with pytest.raises(ValueError):
        quality.verified_daily_targets(daily, source, now=NOW)
