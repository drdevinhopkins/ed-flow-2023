#!/usr/bin/env python3
"""Publication-aware Montréal respiratory surveillance features for daily ED forecasts.

INSPQ weekly reports contain regional influenza A/B, RSV and SARS-CoV-2 positivity.
``available_date`` is the information timestamp: a historical forecast may only use a
report after it was published. Daily covariates carry the latest already-published report
forward; future reports are never interpolated backward.

For Montréal we prefer the table *selon la région de résidence* because it represents the
population whose demand may reach the ED. Older reports without that table fall back to
the last Montréal (06) four-virus row found in the report.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
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
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "aout": 8, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12, "décembre": 12,
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
    match = re.search(r"\b(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\s+(\d{4})\b", text, re.I)
    if not match:
        return None
    day, month_name, year = match.groups()
    month = _FRENCH_MONTHS.get(month_name.lower())
    return None if month is None else pd.Timestamp(year=int(year), month=month, day=int(day))


def parse_report_week(text: str, url: str = "") -> tuple[int | None, int | None]:
    match = re.search(r"\((\d{2})-(\d{1,2})\)", text)
    if match:
        year2, week = match.groups()
        return 2000 + int(year2), int(week)
    filename = re.search(r"/(20\d{2})-(\d{1,2})\.pdf", url)
    return (int(filename.group(1)), int(filename.group(2))) if filename else (None, None)


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
        reports[url] = ReportLink(url, label, available_date, year, week)
    return sorted(reports.values(), key=lambda report: (report.available_date, report.url))


def _browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
        ),
        "Accept-Language": "fr-CA,fr;q=0.9,en;q=0.7",
    }


def fetch_report_index(url: str = INSPQ_INDEX_URL) -> list[ReportLink]:
    response = requests.get(url, timeout=60, headers=_browser_headers())
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


_CELL = (
    r"([0-9][0-9 ]*)\s*/\s*([0-9][0-9 ]*)"
    r"(?:\s*\(([0-9]+(?:[,.][0-9]+)?)\s*%\))?"
)
_ROW = re.compile(
    r"Montr(?:é|e)al\s*\(06\)\s*" + _CELL + r"\s*" + _CELL + r"\s*" + _CELL + r"\s*" + _CELL,
    re.I,
)


def _cell_values(groups: tuple[str | None, str | None, str | None]) -> tuple[float, float, float]:
    positive_text, tested_text, pct_text = groups
    positive = float((positive_text or "0").replace(" ", ""))
    tested = float((tested_text or "0").replace(" ", ""))
    pct = float(pct_text.replace(",", ".")) if pct_text else (100.0 * positive / tested if tested else np.nan)
    return positive, tested, pct


def parse_montreal_report_text(text: str) -> dict[str, float]:
    """Extract the four-virus Montréal (06) row, preferring region of residence."""
    flattened = re.sub(r"\s+", " ", text.replace("\xa0", " "))
    residence_marker = re.search(r"selon\s+la\s+r(?:é|e)gion\s+de\s+r(?:é|e)sidence", flattened, re.I)
    search_text = flattened[residence_marker.start():] if residence_marker else flattened
    matches = list(_ROW.finditer(search_text))
    if not matches and residence_marker:
        matches = list(_ROW.finditer(flattened))
    if not matches:
        raise ValueError("Could not locate Montréal (06) four-virus row in INSPQ report")

    # If several compatible tables remain, the later row is the most specific fallback.
    groups = matches[-1].groups()
    if len(groups) != 12:
        raise ValueError(f"Unexpected Montréal row group count: {len(groups)}")
    result: dict[str, float] = {}
    for index, virus in enumerate(VIRUS_KEYS):
        positive, tested, pct = _cell_values(groups[index * 3 : index * 3 + 3])
        result[f"{virus}_positive"] = positive
        result[f"{virus}_tested"] = tested
        result[f"{virus}_pct"] = pct
    return result


def fetch_report(report: ReportLink) -> dict[str, object]:
    response = requests.get(report.url, timeout=90, headers=_browser_headers())
    response.raise_for_status()
    values = parse_montreal_report_text(extract_pdf_text(response.content))
    return {
        "available_date": report.available_date,
        "surveillance_year": report.surveillance_year,
        "surveillance_week": report.surveillance_week,
        "source_url": report.url,
        **values,
    }


def _pct_from_counts(positive: pd.Series, tested: pd.Series) -> pd.Series:
    return 100.0 * positive / tested.where(tested > 0)


def engineer_weekly_features(raw: pd.DataFrame) -> pd.DataFrame:
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
        positive = f"{virus}_positive"
        tested = f"{virus}_tested"
        if pct not in out:
            if positive not in out or tested not in out:
                raise ValueError(f"Missing respiratory fields for {virus}")
            out[pct] = _pct_from_counts(
                pd.to_numeric(out[positive], errors="coerce"),
                pd.to_numeric(out[tested], errors="coerce"),
            )
        out[pct] = pd.to_numeric(out[pct], errors="coerce")

    out["resp_flu_a_pct"] = out["flu_a_pct"]
    out["resp_flu_b_pct"] = out["flu_b_pct"]
    if {"flu_a_positive", "flu_b_positive", "flu_a_tested", "flu_b_tested"}.issubset(out.columns):
        flu_tested = pd.concat(
            [
                pd.to_numeric(out["flu_a_tested"], errors="coerce"),
                pd.to_numeric(out["flu_b_tested"], errors="coerce"),
            ],
            axis=1,
        ).max(axis=1)
        flu_positive = (
            pd.to_numeric(out["flu_a_positive"], errors="coerce").fillna(0)
            + pd.to_numeric(out["flu_b_positive"], errors="coerce").fillna(0)
        )
        out["resp_flu_total_pct"] = _pct_from_counts(flu_positive, flu_tested)
    else:
        out["resp_flu_total_pct"] = out["flu_a_pct"] + out["flu_b_pct"]
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
        mean = source.expanding(min_periods=6).mean()
        std = source.expanding(min_periods=6).std(ddof=0).replace(0, np.nan)
        z_name = f"_{virus}_past_z"
        out[z_name] = ((source - mean) / std).clip(-5, 5)
        z_columns.append(z_name)

    deltas = [f"resp_{virus}_pct_delta_1w" for virus in VIRUS_KEYS]
    out["respiratory_rising_viruses"] = out[deltas].gt(0).sum(axis=1).astype(float)
    out["respiratory_pressure_index"] = out[z_columns].mean(axis=1, skipna=True)
    return out.drop(columns=z_columns)


def expand_to_daily(weekly: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if end < start:
        raise ValueError("end must be >= start")
    featured = engineer_weekly_features(weekly)
    days = pd.DataFrame(
        {"ds": pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D")}
    )
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
    frame = pd.read_csv(str(source))
    if "available_date" not in frame:
        raise ValueError("Respiratory CSV must contain available_date")
    frame["available_date"] = pd.to_datetime(frame["available_date"], errors="coerce")
    return frame.dropna(subset=["available_date"]).sort_values("available_date").reset_index(drop=True)
