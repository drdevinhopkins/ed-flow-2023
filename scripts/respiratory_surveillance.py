#!/usr/bin/env python3
"""Leakage-aware respiratory-virus surveillance features for daily ED forecasting.

INSPQ publishes weekly clinical-laboratory reports with Montréal (RSS 06) positivity
for influenza A, influenza B, RSV and SARS-CoV-2.  The key forecasting timestamp is the
*report availability date*, not the surveillance week itself: a historical daily forecast
may only use a report once it had actually been published.

This module parses the text-based INSPQ PDFs, derives compact weekly trend features,
and expands them to daily known-at-forecast-time covariates by carrying the most recently
published report forward.  No future surveillance values are interpolated backwards.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from PyPDF2 import PdfReader

INSPQ_INDEX_URL = "https://www.inspq.qc.ca/influenza"

VIRUS_KEYS = ("flu_a", "flu_b", "rsv", "covid")
RESPIRATORY_RAW_COLUMNS = [
    "resp_flu_a_pct",
    "resp_flu_b_pct",
    "resp_flu_total_pct",
    "resp_rsv_pct",
    "resp_covid_pct",
    "respiratory_pct_sum",
]
RESPIRATORY_TREND_COLUMNS = [
    *[f"resp_{virus}_pct_delta_1w" for virus in VIRUS_KEYS],
    *[f"resp_{virus}_pct_delta_2w" for virus in VIRUS_KEYS],
    *[f"resp_{virus}_pct_ma3" for virus in VIRUS_KEYS],
    *[f"resp_{virus}_pct_accel" for virus in VIRUS_KEYS],
    "respiratory_rising_viruses",
    "respiratory_pressure_index",
    "resp_surveillance_age_days",
]
RESPIRATORY_FEATURE_COLUMNS = [*RESPIRATORY_RAW_COLUMNS, *RESPIRATORY_TREND_COLUMNS]

_FRENCH_MONTHS = {
    "janvier": 1,
    "fevrier": 2,
    "février": 2,
    "mars": 3,
    "avril": 4,
    "mai": 5,
    "juin": 6,
    "juillet": 7,
    "aout": 8,
    "août": 8,
    "septembre": 9,
    "octobre": 10,
    "novembre": 11,
    "decembre": 12,
    "décembre": 12,
}


@dataclass(frozen=True)
class ReportLink:
    url: str
    label: str
    available_date: pd.Timestamp
    surveillance_year: int | None = None
    surveillance_week: int | None = None


class _ReportLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._href: str | None = None
        self._parts: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href and ".pdf" in href.lower() and "influenza" in href.lower():
            self._href = href
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            label = " ".join(part.strip() for part in self._parts if part.strip())
            self.links.append((self._href, label))
            self._href = None
            self._parts = []


def parse_french_date(text: str) -> pd.Timestamp | None:
    match = re.search(
        r"\b(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _FRENCH_MONTHS.get(month_name.lower())
    if month is None:
        return None
    return pd.Timestamp(year=int(year), month=month, day=int(day))


def parse_report_week(text: str, url: str = "") -> tuple[int | None, int | None]:
    match = re.search(r"\((\d{2})-(\d{1,2})\)", text)
    if match:
        year2, week = match.groups()
        return 2000 + int(year2), int(week)
    filename = re.search(r"/(20\d{2})-(\d{1,2})\.pdf", url)
    if filename:
        return int(filename.group(1)), int(filename.group(2))
    return None, None


def discover_report_links(html: str, *, base_url: str = INSPQ_INDEX_URL) -> list[ReportLink]:
    parser = _ReportLinkParser()
    parser.feed(html)
    reports: dict[str, ReportLink] = {}
    for href, label in parser.links:
        available_date = parse_french_date(label)
        if available_date is None:
            continue
        url = urljoin(base_url, href)
        year, week = parse_report_week(label, url)
        reports[url] = ReportLink(
            url=url,
            label=label,
            available_date=available_date,
            surveillance_year=year,
            surveillance_week=week,
        )
    return sorted(reports.values(), key=lambda report: (report.available_date, report.url))


def fetch_report_index(url: str = INSPQ_INDEX_URL) -> list[ReportLink]:
    response = requests.get(
        url,
        timeout=60,
        headers={"User-Agent": "ed-flow-2023 respiratory-surveillance collector"},
    )
    response.raise_for_status()
    reports = discover_report_links(response.text, base_url=url)
    if not reports:
        raise ValueError(f"No dated INSPQ influenza PDF report links found at {url}")
    return reports


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("INSPQ PDF contained no extractable text")
    return text


def parse_montreal_report_text(text: str) -> dict[str, float]:
    """Extract Montréal RSS 06 influenza A/B, RSV and SARS-CoV-2 cells."""
    flattened = re.sub(r"\s+", " ", text.replace("\xa0", " "))
    region = re.search(
        r"RSS\s+Montr(?:é|e)al\s*\(06\)\s*(.*?)(?=RSS\s+[A-Za-zÀ-ÿ]|Nombre\s+et\s+pourcentage|$)",
        flattened,
        flags=re.IGNORECASE,
    )
    if not region:
        raise ValueError("Could not locate RSS Montréal (06) in INSPQ report")

    cells = re.findall(
        r"([0-9][0-9 ]*)\s*/\s*([0-9][0-9 ]*)\s*\(([0-9]+(?:[,.][0-9]+)?)\s*%\)",
        region.group(1),
    )
    if len(cells) < 4:
        raise ValueError(f"Expected four Montréal virus cells, found {len(cells)}")

    result: dict[str, float] = {}
    for virus, (positive, tested, pct) in zip(VIRUS_KEYS, cells[:4]):
        result[f"{virus}_positive"] = float(positive.replace(" ", ""))
        result[f"{virus}_tested"] = float(tested.replace(" ", ""))
        result[f"{virus}_pct"] = float(pct.replace(",", "."))
    return result


def fetch_report(report: ReportLink) -> dict[str, object]:
    response = requests.get(
        report.url,
        timeout=90,
        headers={"User-Agent": "ed-flow-2023 respiratory-surveillance collector"},
    )
    response.raise_for_status()
    values = parse_montreal_report_text(extract_pdf_text(response.content))
    return {
        "available_date": report.available_date,
        "surveillance_year": report.surveillance_year,
        "surveillance_week": report.surveillance_week,
        "source_url": report.url,
        *values,
    }


def _pct_from_counts(positive: pd.Series, tested: pd.Series) -> pd.Series:
    return 100.0 * positive / tested.where(tested > 0)


def engineer_weekly_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create raw and trend features using only current/past published reports."""
    if "available_date" not in raw:
        raise ValueError("Respiratory surveillance data require available_date")
    out = raw.copy()
    out["available_date"] = pd.to_datetime(out["available_date"], errors="coerce").dt.normalize()
    out = (
        out.dropna(subset=["available_date"])
        .sort_values("available_date")
        .drop_duplicates("available_date", keep="last")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError("No valid respiratory surveillance rows")

    for virus in VIRUS_KEYS:
        pct = f"{virus}_pct"
        if pct not in out:
            if f"{virus}_positive" not in out or f"{virus}_tested" not in out:
                raise ValueError(f"Missing respiratory fields for {virus}")
            out[pct] = _pct_from_counts(
                pd.to_numeric(out[f"{virus}_positive"], errors="coerce"),
                pd.to_numeric(out[f"{virus}_tested"], errors="coerce"),
            )
        out[pct] = pd.to_numeric(out[pct], errors="coerce")

    out["resp_flu_a_pct"] = out["flu_a_pct"]
    out["resp_flu_b_pct"] = out["flu_b_pct"]
    flu_denominator = pd.concat(
        [
            pd.to_numeric(out.get("flu_a_tested"), errors="coerce"),
            pd.to_numeric(out.get("flu_b_tested"), errors="coerce"),
        ],
        axis=1,
    ).max(axis=1)
    flu_positive = (
        pd.to_numeric(out.get("flu_a_positive"), errors="coerce").fillna(0)
        + pd.to_numeric(out.get("flu_b_positive"), errors="coerce").fillna(0)
    )
    out["resp_flu_total_pct"] = _pct_from_counts(flu_positive, flu_denominator)
    out["resp_rsv_pct"] = out["rsv_pct"]
    out["resp_covid_pct"] = out["covid_pct"]
    out["respiratory_pct_sum"] = out[
        ["resp_flu_a_pct", "resp_flu_b_pct", "resp_rsv_pct", "resp_covid_pct"]
    ].sum(axis=1, min_count=1)

    z_columns: list[str] = []
    for virus in VIRUS_KEYS:
        source = pd.to_numeric(out[f"{virus}_pct"], errors="coerce")
        prefix = f"resp_{virus}_pct"
        out[f"{prefix}_delta_1w"] = source.diff(1)
        out[f"{prefix}_delta_2w"] = source.diff(2)
        out[f"{prefix}_ma3"] = source.rolling(3, min_periods=1).mean()
        out[f"{prefix}_accel"] = out[f"{prefix}_delta_1w"].diff(1)

        expanding_mean = source.expanding(min_periods=6).mean()
        expanding_std = source.expanding(min_periods=6).std(ddof=0).replace(0, np.nan)
        z_name = f"_{virus}_expanding_z"
        out[z_name] = ((source - expanding_mean) / expanding_std).clip(-5, 5)
        z_columns.append(z_name)

    delta_columns = [f"resp_{virus}_pct_delta_1w" for virus in VIRUS_KEYS]
    out["respiratory_rising_viruses"] = out[delta_columns].gt(0).sum(axis=1).astype(float)
    out["respiratory_pressure_index"] = out[z_columns].mean(axis=1, skipna=True)
    out = out.drop(columns=z_columns)
    return out


def expand_to_daily(
    weekly: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Carry the latest *published* report forward to each daily forecast date."""
    if end < start:
        raise ValueError("end must be >= start")
    featured = engineer_weekly_features(weekly)
    days = pd.DataFrame({"ds": pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")})
    source_columns = [
        "available_date",
        *[column for column in RESPIRATORY_FEATURE_COLUMNS if column != "resp_surveillance_age_days"],
    ]
    daily = pd.merge_asof(
        days.sort_values("ds"),
        featured[source_columns].sort_values("available_date"),
        left_on="ds",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )
    daily["resp_surveillance_age_days"] = (
        daily["ds"] - daily["available_date"]
    ).dt.days.astype("float64")
    return daily[["ds", "available_date", *RESPIRATORY_FEATURE_COLUMNS]]


def load_surveillance_csv(source: str | Path) -> pd.DataFrame:
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        frame = pd.read_csv(source_text)
    else:
        frame = pd.read_csv(source_text)
    if "available_date" not in frame:
        raise ValueError("Respiratory CSV must contain available_date")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    return frame.dropna(subset=["available_date"]).sort_values("available_date").reset_index(drop=True)
