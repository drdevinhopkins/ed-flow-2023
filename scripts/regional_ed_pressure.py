#!/usr/bin/env python3
"""Feature engineering for Montréal-wide emergency-department pressure.

Primary source is the official MSSS rolling hourly CSV.  Some cloud networks receive
HTTP 403 from that legacy host, so the collector has an official Quebec.ca status-page
fallback.  The fallback contains fewer fields but preserves the core peer signals:
patients present, waiting for physician, stretcher occupancy, and threshold counts.

JGH is always excluded.  Regional pressure is observed state rather than known-future
information, so historical forecasts must persist only the cutoff state/trends forward.
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
    "hopital general juif", "hôpital général juif", "jewish general hospital",
    "sir mortimer b davis", "sir mortimer b. davis",
)

SOURCE_COLUMNS = {
    "rss": "rss", "region": "region", "nom_etablissement": "establishment",
    "nom_installation": "installation", "no_permis_installation": "permit",
    "nombre_de_civieres_fonctionnelles": "stretcher_capacity",
    "nombre_de_civieres_occupees": "stretcher_occupied",
    "nombre_de_patients_sur_civieres_plus_de_24_heures": "stretcher_24h",
    "nombre_de_patients_sur_civiere_plus_de_24_heures": "stretcher_24h",
    "nombre_de_patients_sur_civieres_plus_de_48_heures": "stretcher_48h",
    "nombre_de_patients_sur_civiere_plus_de_48_heures": "stretcher_48h",
    "nombre_total_de_patients_presents_a_lurgence": "patients_present",
    "nombre_total_de_patients_en_attente_de_pec": "waiting_pec",
    "dms_sur_civiere": "dms_stretcher", "dms_ambulatoire": "dms_ambulatory",
    "heure_de_lextraction_image": "extraction_time",
    "heure_de_lextraction": "extraction_time", "mise_a_jour": "updated_at",
}
REGIONAL_STATE_COLUMNS = [
    "regional_peer_installations", "regional_stretcher_capacity",
    "regional_stretcher_occupied", "regional_stretcher_occupancy",
    "regional_patients_present", "regional_waiting_pec", "regional_waiting_share",
    "regional_stretcher_24h", "regional_stretcher_48h",
    "regional_max_stretcher_occupancy", "regional_mean_stretcher_occupancy",
    "regional_ed_over_100pct", "regional_ed_over_120pct", "regional_ed_over_150pct",
]
TREND_BASE_COLUMNS = [
    "regional_stretcher_occupancy", "regional_patients_present", "regional_waiting_pec",
    "regional_stretcher_24h", "regional_stretcher_48h", "regional_max_stretcher_occupancy",
]
REGIONAL_TREND_COLUMNS = [f"{c}_delta_{lag}h" for c in TREND_BASE_COLUMNS for lag in (1, 3, 6)]
REGIONAL_FEATURE_COLUMNS = [*REGIONAL_STATE_COLUMNS, *REGIONAL_TREND_COLUMNS]

_FRENCH_MONTHS = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "aout": 8, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11,
    "decembre": 12, "décembre": 12,
}


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c)).lower().replace("’", "'")
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
            pass
    return content.decode("latin-1", errors="replace")


def _page_timestamp(text: str) -> pd.Timestamp:
    now = pd.Timestamp.now(tz="America/Montreal").tz_localize(None)
    match = re.search(
        r"Dernière mise à jour\s*:?\s*(?:[A-Za-zÀ-ÿ]+\s+)?(\d{1,2})\s+"
        r"([A-Za-zÀ-ÿ]+)(?:\s+(\d{4}))?\s+à\s+(\d{1,2})\s*[h:]\s*(\d{2})?",
        text, re.I,
    )
    if not match:
        return now.floor("h")
    day, month_name, year, hour, minute = match.groups()
    month = _FRENCH_MONTHS.get(month_name.lower())
    if month is None:
        return now.floor("h")
    candidate = pd.Timestamp(
        year=int(year or now.year), month=month, day=int(day),
        hour=int(hour), minute=int(minute or 0),
    )
    # Around New Year, a page dated December can belong to the previous calendar year.
    if candidate - now > pd.Timedelta(days=30):
        candidate = candidate.replace(year=candidate.year - 1)
    return candidate.floor("h")


def _metric_number(text: str, label_pattern: str) -> float:
    match = re.search(label_pattern + r"\s*:?\s*([0-9][0-9\s]*)", text, re.I)
    if not match:
        return np.nan
    return float(re.sub(r"\s+", "", match.group(1)))


def parse_quebec_status_html(html: str) -> pd.DataFrame:
    """Parse server-rendered Quebec.ca ED cards into normalized current-state rows."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    page_text = soup.get_text(" ", strip=True)
    ds = _page_timestamp(page_text)
    marker_total = "Nombre total de personnes à l’urgence"
    marker_occ = "Taux d’occupation des civières"
    candidates = []
    for node in soup.find_all(["article", "li", "div"]):
        text = node.get_text(" ", strip=True)
        if marker_total not in text or marker_occ not in text:
            continue
        # Keep the smallest card-like node rather than a parent containing many cards.
        if any(
            marker_total in child.get_text(" ", strip=True) and marker_occ in child.get_text(" ", strip=True)
            for child in node.find_all(["article", "li", "div"], recursive=False)
        ):
            continue
        candidates.append(node)

    rows: list[dict[str, object]] = []
    for node in candidates:
        text = node.get_text(" ", strip=True)
        if "Montréal" not in text:
            continue
        heading = node.find(["h2", "h3", "h4", "h5"])
        if heading is None:
            links = [a.get_text(" ", strip=True) for a in node.find_all("a") if a.get_text(strip=True)]
            installation = links[0] if links else ""
        else:
            installation = heading.get_text(" ", strip=True)
        if not installation or installation.lower().startswith(("résultats", "ensemble du québec")):
            continue
        waiting = _metric_number(
            text,
            r"Nombre de personnes qui attendent de voir un médecin à l[’']urgence",
        )
        present = _metric_number(text, r"Nombre total de personnes à l[’']urgence")
        occupancy_pct = _metric_number(text, r"Taux d[’']occupation des civières")
        if not np.isfinite(present) or not np.isfinite(occupancy_pct):
            continue
        rows.append({
            "rss": "06", "region": "Montréal", "installation": installation,
            "stretcher_capacity": np.nan, "stretcher_occupied": np.nan,
            "stretcher_24h": np.nan, "stretcher_48h": np.nan,
            "patients_present": present, "waiting_pec": waiting,
            "installation_occupancy": occupancy_pct / 100.0,
            "extraction_time": ds,
        })
    return pd.DataFrame(rows)


def load_quebec_status_feed(url: str = QUEBEC_STATUS_URL, max_pages: int = 15) -> pd.DataFrame:
    """Fetch official Quebec.ca status cards; used when the MSSS CSV rejects cloud IPs."""
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    empty_pages = 0
    for page in range(1, max_pages + 1):
        response = requests.get(
            url,
            params={"tx_solr[page]": page},
            timeout=60,
            headers=_browser_headers(),
        )
        response.raise_for_status()
        frame = parse_quebec_status_html(response.text)
        if frame.empty:
            empty_pages += 1
            if empty_pages >= 2 and frames:
                break
            continue
        empty_pages = 0
        new = frame.loc[~frame["installation"].astype(str).isin(seen)].copy()
        seen.update(new["installation"].astype(str))
        if not new.empty:
            frames.append(new)
    if not frames:
        raise ValueError("Quebec.ca fallback returned no parsable Montréal ED cards")
    result = pd.concat(frames, ignore_index=True)
    print(f"MSSS CSV fallback: parsed {len(result)} Montréal ED cards from Quebec.ca", flush=True)
    return result


def load_public_feed(source: str | Path = DEFAULT_SOURCE_URL) -> pd.DataFrame:
    source_text = str(source)
    if not source_text.startswith(("http://", "https://")):
        return pd.read_csv(source_text, sep=None, engine="python")
    response = requests.get(source_text, timeout=60, headers=_browser_headers())
    if response.status_code == 403 and source_text == DEFAULT_SOURCE_URL:
        print("MSSS CSV returned HTTP 403; using official Quebec.ca status-page fallback", flush=True)
        return load_quebec_status_feed()
    response.raise_for_status()
    return pd.read_csv(io.StringIO(_decode_csv(response.content)), sep=None, engine="python")


def normalize_public_feed(frame: pd.DataFrame) -> pd.DataFrame:
    rename = {column: SOURCE_COLUMNS[_slug(column)] for column in frame.columns if _slug(column) in SOURCE_COLUMNS}
    out = frame.rename(columns=rename).copy()
    required = {"rss", "installation", "patients_present", "waiting_pec", "extraction_time"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError("Emergency feed missing expected columns: " + ", ".join(sorted(missing)))
    for column in (
        "stretcher_capacity", "stretcher_occupied", "stretcher_24h", "stretcher_48h",
        "dms_stretcher", "dms_ambulatory", "installation_occupancy",
    ):
        if column not in out:
            out[column] = np.nan
    rss_numeric = pd.to_numeric(out["rss"], errors="coerce")
    out["rss"] = rss_numeric.map(lambda x: f"{int(x):02d}" if pd.notna(x) else "")
    out["installation"] = out["installation"].fillna("").astype(str).str.strip()
    for column in (
        "stretcher_capacity", "stretcher_occupied", "stretcher_24h", "stretcher_48h",
        "patients_present", "waiting_pec", "dms_stretcher", "dms_ambulatory",
        "installation_occupancy",
    ):
        out[column] = pd.to_numeric(
            out[column].astype(str).str.replace(",", ".", regex=False), errors="coerce"
        )
    out["ds"] = pd.to_datetime(out["extraction_time"], errors="coerce", dayfirst=True)
    if out["ds"].isna().all() and "updated_at" in out:
        out["ds"] = pd.to_datetime(out["updated_at"], errors="coerce", dayfirst=True)
    out = out.dropna(subset=["ds"]).copy()
    out["ds"] = out["ds"].dt.floor("h")
    return out.sort_values(["ds", "installation"]).reset_index(drop=True)


def _normalized_patterns(patterns: Iterable[str]) -> tuple[str, ...]:
    return tuple(_slug(p) for p in patterns if str(p).strip())


def is_jgh_installation(installation: pd.Series, patterns: Iterable[str] = DEFAULT_JGH_PATTERNS) -> pd.Series:
    normalized = installation.fillna("").astype(str).map(_slug)
    mask = pd.Series(False, index=installation.index)
    for token in _normalized_patterns(patterns):
        mask |= normalized.str.contains(re.escape(token), regex=True, na=False)
    return mask


def add_regional_trends(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute 1/3/6-hour deltas after archive merging."""
    out = frame.copy().sort_values("ds").reset_index(drop=True)
    for column in TREND_BASE_COLUMNS:
        values = pd.to_numeric(out.get(column), errors="coerce")
        for lag in (1, 3, 6):
            out[f"{column}_delta_{lag}h"] = values.diff(lag)
    for column in REGIONAL_FEATURE_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[["ds", *REGIONAL_FEATURE_COLUMNS]]


def build_regional_peer_pressure(
    raw: pd.DataFrame, *, region_code: str = DEFAULT_REGION_CODE,
    jgh_patterns: Iterable[str] = DEFAULT_JGH_PATTERNS,
) -> pd.DataFrame:
    source = normalize_public_feed(raw)
    rows = source.loc[source["rss"].eq(str(region_code).zfill(2))].copy()
    rows = rows.loc[~is_jgh_installation(rows["installation"], jgh_patterns)].copy()
    rows = rows.loc[~rows["installation"].str.lower().isin({"total régional", "total regional", "ensemble du québec"})]
    if rows.empty:
        raise ValueError("No Montréal peer ED rows remain after excluding JGH")

    capacity = pd.to_numeric(rows["stretcher_capacity"], errors="coerce")
    occupied = pd.to_numeric(rows["stretcher_occupied"], errors="coerce")
    calculated_occ = occupied / capacity.where(capacity > 0)
    rows["installation_occupancy"] = pd.to_numeric(rows["installation_occupancy"], errors="coerce").fillna(calculated_occ)
    grouped = rows.groupby("ds", sort=True)
    sum_min = lambda values: values.sum(min_count=1)
    result = grouped.agg(
        regional_peer_installations=("installation", "nunique"),
        regional_stretcher_capacity=("stretcher_capacity", sum_min),
        regional_stretcher_occupied=("stretcher_occupied", sum_min),
        regional_patients_present=("patients_present", sum_min),
        regional_waiting_pec=("waiting_pec", sum_min),
        regional_stretcher_24h=("stretcher_24h", sum_min),
        regional_stretcher_48h=("stretcher_48h", sum_min),
        regional_max_stretcher_occupancy=("installation_occupancy", "max"),
        regional_mean_stretcher_occupancy=("installation_occupancy", "mean"),
    ).reset_index()
    thresholds = rows.assign(
        over_100=rows["installation_occupancy"].gt(1.00),
        over_120=rows["installation_occupancy"].gt(1.20),
        over_150=rows["installation_occupancy"].gt(1.50),
    ).groupby("ds", sort=True).agg(
        regional_ed_over_100pct=("over_100", "sum"),
        regional_ed_over_120pct=("over_120", "sum"),
        regional_ed_over_150pct=("over_150", "sum"),
    ).reset_index()
    result = result.merge(thresholds, on="ds", how="left")
    weighted = result["regional_stretcher_occupied"] / result["regional_stretcher_capacity"].where(
        result["regional_stretcher_capacity"] > 0
    )
    result["regional_stretcher_occupancy"] = weighted.fillna(result["regional_mean_stretcher_occupancy"])
    result["regional_waiting_share"] = result["regional_waiting_pec"] / result["regional_patients_present"].where(
        result["regional_patients_present"] > 0
    )
    result = result.sort_values("ds").drop_duplicates("ds", keep="last").reset_index(drop=True)
    return add_regional_trends(result)


def persistence_future(pressure_history: pd.DataFrame, *, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    history = pressure_history.loc[pressure_history["ds"] <= cutoff].sort_values("ds")
    if history.empty:
        raise ValueError(f"No regional pressure available at or before cutoff {cutoff}")
    latest = history.iloc[-1]
    future = pd.DataFrame({"ds": pd.date_range(cutoff + pd.Timedelta(hours=1), periods=horizon, freq="h")})
    for column in REGIONAL_FEATURE_COLUMNS:
        future[column] = latest.get(column, np.nan)
    return future
