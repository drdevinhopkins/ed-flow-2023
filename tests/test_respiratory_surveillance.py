#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from respiratory_surveillance import (  # noqa: E402
    discover_report_links,
    engineer_weekly_features,
    expand_to_daily,
    parse_montreal_report_text,
)


def test_report_link_discovery_preserves_publication_date() -> None:
    html = """
    <html><body>
      <a href="/sites/default/files/documents/influenza/20252026/2026-10.pdf?rapport=10">
        17 août 2026 (26-32)
      </a>
    </body></html>
    """
    reports = discover_report_links(html, base_url="https://www.inspq.qc.ca/influenza")
    assert len(reports) == 1
    report = reports[0]
    assert report.available_date == pd.Timestamp("2026-08-17")
    assert report.surveillance_year == 2026
    assert report.surveillance_week == 32
    assert report.url.startswith("https://www.inspq.qc.ca/sites/")


def test_montreal_pdf_text_parser() -> None:
    text = """
    RSS Montréal (06)
    Influenza A 2 / 905 (0,22 %)
    Influenza B 15 / 905 (1,66 %)
    VRS 2 / 778 (0,26 %)
    SARS-CoV-2 8 / 987 (0,81 %)
    RSS Outaouais (07) 1 / 100 (1,00 %)
    """
    row = parse_montreal_report_text(text)
    assert row["flu_a_positive"] == 2
    assert row["flu_a_tested"] == 905
    assert np.isclose(row["flu_a_pct"], 0.22)
    assert np.isclose(row["flu_b_pct"], 1.66)
    assert np.isclose(row["rsv_pct"], 0.26)
    assert np.isclose(row["covid_pct"], 0.81)


def weekly_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-06-01", periods=8, freq="7D")
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "available_date": date,
                "flu_a_positive": 10 + 2 * index,
                "flu_a_tested": 1000,
                "flu_a_pct": 1.0 + 0.2 * index,
                "flu_b_positive": 20 - index,
                "flu_b_tested": 1000,
                "flu_b_pct": 2.0 - 0.1 * index,
                "rsv_positive": 5 + index,
                "rsv_tested": 800,
                "rsv_pct": 0.5 + 0.1 * index,
                "covid_positive": 30 + 3 * index,
                "covid_tested": 900,
                "covid_pct": 3.0 + 0.3 * index,
            }
        )
    return pd.DataFrame(rows)


def test_weekly_trends_only_use_current_and_prior_reports() -> None:
    featured = engineer_weekly_features(weekly_frame())
    assert np.isclose(featured.iloc[3]["resp_flu_a_pct_delta_1w"], 0.2)
    assert np.isclose(featured.iloc[3]["resp_flu_a_pct_delta_2w"], 0.4)
    assert np.isclose(featured.iloc[3]["resp_flu_a_pct_ma3"], np.mean([1.2, 1.4, 1.6]))
    assert featured.iloc[3]["respiratory_rising_viruses"] == 3

    # Appending a future report must not alter already-computed historical features.
    original = featured.loc[featured["available_date"].eq(pd.Timestamp("2026-06-22"))].iloc[0]
    extended = pd.concat(
        [
            weekly_frame(),
            pd.DataFrame(
                [
                    {
                        "available_date": "2026-08-01",
                        "flu_a_positive": 900,
                        "flu_a_tested": 1000,
                        "flu_a_pct": 90.0,
                        "flu_b_positive": 900,
                        "flu_b_tested": 1000,
                        "flu_b_pct": 90.0,
                        "rsv_positive": 700,
                        "rsv_tested": 800,
                        "rsv_pct": 87.5,
                        "covid_positive": 800,
                        "covid_tested": 900,
                        "covid_pct": 88.9,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    extended_featured = engineer_weekly_features(extended)
    same = extended_featured.loc[
        extended_featured["available_date"].eq(pd.Timestamp("2026-06-22"))
    ].iloc[0]
    for column in (
        "resp_flu_a_pct_delta_1w",
        "resp_flu_a_pct_ma3",
        "respiratory_pressure_index",
    ):
        left, right = original[column], same[column]
        assert (pd.isna(left) and pd.isna(right)) or np.isclose(left, right)


def test_daily_expansion_is_publication_aware() -> None:
    weekly = weekly_frame().iloc[:2].copy()
    daily = expand_to_daily(
        weekly,
        start=pd.Timestamp("2026-05-30"),
        end=pd.Timestamp("2026-06-12"),
    )
    before = daily.loc[daily["ds"].eq(pd.Timestamp("2026-05-31"))].iloc[0]
    first = daily.loc[daily["ds"].eq(pd.Timestamp("2026-06-06"))].iloc[0]
    second = daily.loc[daily["ds"].eq(pd.Timestamp("2026-06-10"))].iloc[0]
    assert pd.isna(before["available_date"])
    assert first["available_date"] == pd.Timestamp("2026-06-01")
    assert first["resp_surveillance_age_days"] == 5
    assert second["available_date"] == pd.Timestamp("2026-06-08")
    assert second["resp_surveillance_age_days"] == 2


def main() -> None:
    test_report_link_discovery_preserves_publication_date()
    test_montreal_pdf_text_parser()
    test_weekly_trends_only_use_current_and_prior_reports()
    test_daily_expansion_is_publication_aware()
    print("respiratory surveillance tests passed")


if __name__ == "__main__":
    main()
