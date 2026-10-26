#!/usr/bin/env python3
"""Feature engineering for Montréal-wide emergency-department pressure.

The preferred source is the official MSSS rolling seven-day hourly CSV. GitHub-hosted
runners can receive HTTP 403 from that legacy host, so collection falls back to the
official Données Québec datastore/CSV export, then the Quebec.ca status page.
The status-page fallback has fewer fields but
still supplies patients present, waiting to see a physician, stretcher occupancy and
regional threshold counts.

JGH is explicitly excluded from peer aggregates. Regional pressure is an observed state,
not a known-future covariate: historical forecasts may use values through their cutoff
only, with the cutoff state/trends persisted across the forecast horizon.
"""

from __future__ import annotations

import io
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

DEFAULT_SOURCE_URL = (
    "https://www.msss.gouv.qc.ca/professionnels/statistiques/documents/urgences/"
    "Releve_horaire_urgences_7jours_nbpers.csv"
)
QUEBEC_STATUS_URL = (
    "https://www.quebec.ca/sante/systeme-et-services-de-sante/organisation-des-services/"
    "donnees-systeme-sante-quebecois-services/situation-urgences"
)
DEFAULT_REGION_CODE = "06"
DEFAULT_JGH_PATTERNS = (
    "hopital general juif",
    "hôpital général juif",
    "jewish general hospital",
    "sir mortimer b davis",
    "sir mortimer b. davis",
)

SOURCE_COLUMNS = {
    "rss": "rss",
    "region": "region",
    "nom_etablissement": "establishment",
    "nom_installation": "installation",
    "no_permis_installation": "permit",
    "nombre_de_civieres_fonctionnelles": "stretcher_capacity",
    "nombre_de_civieres_occupees": "stretcher_occupied",
    "nombre_de_patients_sur_civieres_plus_de_24_heures": "stretcher_24h",
    "nombre_de_patients_sur_civiere_plus_de_24_heures": "stretcher_24h",
    "nombre_de_patients_sur_civieres_plus_de_48_heures": "stretcher_48h",
    "nombre_de_patients_sur_civiere_plus_de_48_heures": "stretcher_48h",
    "nombre_total_de_patients_presents_a_lurgence": "patients_present",
    "nombre_total_de_patients_en_attente_de_pec": "waiting_pec",
    "dms_sur_civiere": "dms_stretcher",
    "dms_ambulatoire": "dms_ambulatory",
    # Source headings have appeared with and without a separator after the article.
    "heure_de_lextraction_image": "extraction_time",
    "heure_de_l_extraction_image": "extraction_time",
    "heure_de_lextraction": "extraction_time",
    "heure_de_l_extraction": "extraction_time",
    "mise_a_jour": "updated_at",
}

REGIONAL_STATE_COLUMNS = [
    "regional_peer_installations",
    "regional_stretcher_capacity",
    "regional_stretcher_occupied",
    "regional_stretcher_occupancy",
    "regional_patients_present",
    "regional_waiting_pec",
    "regional_waiting_share",
    "regional_stretcher_24h",
    "regional_stretcher_48h",
    "regional_max_stretcher_occupancy",
    "regional_mean_stretcher_occupancy",
    "regional_ed_over_100pct",
    "regional_ed_over_120pct",
    "regional_ed_over_150pct",
]
TREND_BASE_COLUMNS = [
    "regional_stretcher_occupancy",
    "regional_patients_present",
    "regional_waiting_pec",
    "regional_stretcher_24h",
    "regional_stretcher_48h",
    "regional_max_stretcher_occupancy",
]
REGIONAL_TREND_COLUMNS = [
    f"{column}_delta_{lag}h" for column in TREND_BASE_COLUMNS for lag in (1, 3, 6)
]
REGIONAL_FEATURE_COLUMNS = [*REGIONAL_STATE_COLUMNS, *REGIONAL_TREND_COLUMNS]

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


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/csv;q=0.8,*/*;q=0.7",
        "Accept-Language": "fr-CA,fr;q=0.9,en;q=0.7",
        "Referer": "https://www.donneesquebec.ca/",
    }


def _decode_csv(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


def _page_timestamp(text: str) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="America/Montreal").tz_localize(None)
    match = re.search(
        r"Dernière mise à jour\s*:?\s*(?:[A-Za-zÀ-ÿ]+\s+)?(\d{1,2})\s+"
        r"([A-Za-zÀ-ÿ]+)(?:\s+(\d{4}))?\s+à\s+(\d{1,2})\s*[h:]\s*(\d{2})?",
        text,
        re.I,
    )
    if not match:
        raise ValueError("Quebec.ca status page has no valid observation timestamp")
    day, month_name, year, hour, minute = match.groups()
    month = _FRENCH_MONTHS.get(month_name.lower())
    if month is None:
        raise ValueError("Quebec.ca status page has no valid observation timestamp")
    candidate = pd.Timestamp(
        year=int(year or now.year),
        month=month,
        day=int(day),
        hour=int(hour),
        minute=int(minute or 0),
    )
    if candidate - now > pd.Timedelta(days=30):
        candidate = candidate.replace(year=candidate.year - 1)
    return candidate.floor("h")


def _metric_number(text: str, label_pattern: str) -> float:
    match = re.search(label_pattern + r"\s*:?\s*([0-9][0-9\s\u00a0\u202f]*)", text, re.I)
    if not match:
        return np.nan
    digits = re.sub(r"\D", "", match.group(1))
    return float(digits) if digits else np.nan


def _installation_from_schedule_link(text: str) -> str:
    return re.sub(
        r"^.*?horaire\s+de\s+l[’']installation\s+",
        "",
        text.strip(),
        flags=re.I,
    ).strip()


def parse_quebec_status_html(html: str) -> pd.DataFrame:
    """Compatibility wrapper for the shared official status-page parser."""
    from quebec_ed_status import parse_status_page

    return parse_status_page(html)


def load_quebec_status_feed(url: str = QUEBEC_STATUS_URL, max_pages: int = 13) -> pd.DataFrame:
    """Compatibility wrapper for the shared official status-page loader."""
    from quebec_ed_status import load_quebec_status

    return load_quebec_status(max_pages=max_pages, url=url)


def load_public_feed(source: str | Path = DEFAULT_SOURCE_URL) -> pd.DataFrame:
    source_text = str(source)
    if not source_text.startswith(("http://", "https://")):
        return pd.read_csv(source_text, sep=None, engine="python")
    try:
        response = requests.get(source_text, timeout=60, headers=_browser_headers())
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(_decode_csv(response.content)), sep=None, engine="python")
        normalize_public_feed(frame)
        return frame
    except (requests.RequestException, ValueError) as exc:
        if source_text != DEFAULT_SOURCE_URL:
            raise
        from quebec_ed_status import load_official_fallback

        print(f"Primary MSSS regional source unavailable ({type(exc).__name__}: {exc}); "
              "trying official fallbacks", flush=True)
        return load_official_fallback()


def normalize_public_feed(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {
        column: SOURCE_COLUMNS[_slug(column)]
        for column in frame.columns
        if _slug(column) in SOURCE_COLUMNS
    }
    out = frame.rename(columns=rename).copy()
    required = {"rss", "installation", "patients_present", "waiting_pec", "extraction_time"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError("Emergency feed missing expected columns: " + ", ".join(sorted(missing)))

    for column in (
        "stretcher_capacity",
        "stretcher_occupied",
        "stretcher_24h",
        "stretcher_48h",
        "dms_stretcher",
        "dms_ambulatory",
        "installation_occupancy",
    ):
        if column not in out:
            out[column] = np.nan

    rss_numeric = pd.to_numeric(out["rss"], errors="coerce")
    out["rss"] = rss_numeric.map(lambda value: f"{int(value):02d}" if pd.notna(value) else "")
    out["installation"] = out["installation"].fillna("").astype(str).str.strip()
    for column in (
        "stretcher_capacity",
        "stretcher_occupied",
        "stretcher_24h",
        "stretcher_48h",
        "patients_present",
        "waiting_pec",
        "dms_stretcher",
        "dms_ambulatory",
        "installation_occupancy",
    ):
        values = out[column].astype(str).str.replace(",", ".", regex=False)
        out[column] = pd.to_numeric(values, errors="coerce")

    # The datastore mirror supplies a clock time plus a dated publication timestamp.
    # Never let pandas attach the collector's current date to a clock-only observation.
    extraction = out["extraction_time"].astype("string").str.strip()
    clock_only = extraction.str.fullmatch(r"\d{1,2}:\d{2}(?::\d{2})?").fillna(False)
    iso_date = extraction.str.match(r"\d{4}-\d{2}-\d{2}").fillna(False)
    dated = pd.to_datetime(extraction.where(iso_date), errors="coerce", format="mixed")
    other_dates = pd.to_datetime(
        extraction.mask(clock_only | iso_date), errors="coerce", dayfirst=True, format="mixed"
    )
    dated = dated.fillna(other_dates)
    if clock_only.any():
        if "updated_at" not in out:
            raise ValueError("Clock-only extraction times require a dated Mise_a_jour field")
        published = pd.to_datetime(out["updated_at"], errors="coerce", format="mixed")
        clocks = pd.to_timedelta(
            extraction.where(clock_only).str.replace(r"^(\d{1,2}:\d{2})$", r"\1:00", regex=True),
            errors="coerce",
        )
        observed = published.dt.normalize() + clocks
        # A 23:00 observation can be published shortly after midnight.
        observed = observed.where(observed <= published, observed - pd.Timedelta(days=1))
        dated = dated.where(~clock_only, observed)
        if dated.isna().all():
            raise ValueError("Clock-only extraction times have no valid publication date")
    out["ds"] = dated
    if out["ds"].isna().all() and "updated_at" in out:
        out["ds"] = pd.to_datetime(out["updated_at"], errors="coerce", format="mixed")
    out = out.dropna(subset=["ds"]).copy()
    out["ds"] = out["ds"].dt.floor("h")
    return out.sort_values(["ds", "installation"]).reset_index(drop=True)


def _normalized_patterns(patterns: Iterable[str]) -> tuple[str, ...]:
    return tuple(_slug(pattern) for pattern in patterns if str(pattern).strip())


def is_jgh_installation(
    installation: pd.Series,
    patterns: Iterable[str] = DEFAULT_JGH_PATTERNS,
) -> pd.Series:
    normalized = installation.fillna("").astype(str).map(_slug)
    mask = pd.Series(False, index=installation.index)
    for token in _normalized_patterns(patterns):
        mask |= normalized.str.contains(re.escape(token), regex=True, na=False)
    return mask


def add_regional_trends(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute 1/3/6-hour deltas after archive merging."""
    out = frame.copy().sort_values("ds").reset_index(drop=True)
    for column in TREND_BASE_COLUMNS:
        if column in out:
            values = pd.to_numeric(out[column], errors="coerce")
        else:
            values = pd.Series(np.nan, index=out.index, dtype="float64")
        for lag in (1, 3, 6):
            out[f"{column}_delta_{lag}h"] = values.diff(lag)
    for column in REGIONAL_FEATURE_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[["ds", *REGIONAL_FEATURE_COLUMNS]]


def build_regional_peer_pressure(
    raw: pd.DataFrame,
    *,
    region_code: str = DEFAULT_REGION_CODE,
    jgh_patterns: Iterable[str] = DEFAULT_JGH_PATTERNS,
) -> pd.DataFrame:
    source = normalize_public_feed(raw)
    rows = source.loc[source["rss"].eq(str(region_code).zfill(2))].copy()
    rows = rows.loc[~is_jgh_installation(rows["installation"], jgh_patterns)].copy()
    rows = rows.loc[
        ~rows["installation"].str.lower().isin(
            {"total régional", "total regional", "ensemble du québec"}
        )
    ].copy()
    if rows.empty:
        raise ValueError("No Montréal peer ED rows remain after excluding JGH")

    capacity = pd.to_numeric(rows["stretcher_capacity"], errors="coerce")
    occupied = pd.to_numeric(rows["stretcher_occupied"], errors="coerce")
    calculated_occupancy = occupied / capacity.where(capacity > 0)
    rows["installation_occupancy"] = pd.to_numeric(
        rows["installation_occupancy"], errors="coerce"
    ).fillna(calculated_occupancy)

    grouped = rows.groupby("ds", sort=True)
    result = grouped.agg(
        regional_peer_installations=("installation", "nunique"),
        regional_stretcher_capacity=("stretcher_capacity", lambda x: x.sum(min_count=1)),
        regional_stretcher_occupied=("stretcher_occupied", lambda x: x.sum(min_count=1)),
        regional_patients_present=("patients_present", lambda x: x.sum(min_count=1)),
        regional_waiting_pec=("waiting_pec", lambda x: x.sum(min_count=1)),
        regional_stretcher_24h=("stretcher_24h", lambda x: x.sum(min_count=1)),
        regional_stretcher_48h=("stretcher_48h", lambda x: x.sum(min_count=1)),
        regional_max_stretcher_occupancy=("installation_occupancy", "max"),
        regional_mean_stretcher_occupancy=("installation_occupancy", "mean"),
    ).reset_index()

    thresholds = (
        rows.assign(
            over_100=rows["installation_occupancy"].gt(1.00),
            over_120=rows["installation_occupancy"].gt(1.20),
            over_150=rows["installation_occupancy"].gt(1.50),
        )
        .groupby("ds", sort=True)
        .agg(
            regional_ed_over_100pct=("over_100", "sum"),
            regional_ed_over_120pct=("over_120", "sum"),
            regional_ed_over_150pct=("over_150", "sum"),
        )
        .reset_index()
    )
    result = result.merge(thresholds, on="ds", how="left")

    weighted_occupancy = (
        result["regional_stretcher_occupied"]
        / result["regional_stretcher_capacity"].where(result["regional_stretcher_capacity"] > 0)
    )
    result["regional_stretcher_occupancy"] = weighted_occupancy.fillna(
        result["regional_mean_stretcher_occupancy"]
    )
    result["regional_waiting_share"] = (
        result["regional_waiting_pec"]
        / result["regional_patients_present"].where(result["regional_patients_present"] > 0)
    )
    result = (
        result.sort_values("ds")
        .drop_duplicates("ds", keep="last")
        .reset_index(drop=True)
    )
    return add_regional_trends(result)


def persistence_future(
    pressure_history: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Persist only the latest already-observed regional state through the horizon."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    history = pressure_history.loc[pressure_history["ds"] <= cutoff].sort_values("ds")
    if history.empty:
        raise ValueError(f"No regional pressure available at or before cutoff {cutoff}")
    latest = history.iloc[-1]
    future = pd.DataFrame(
        {"ds": pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")}
    )
    for column in REGIONAL_FEATURE_COLUMNS:
        future[column] = latest.get(column, np.nan)
    return future
