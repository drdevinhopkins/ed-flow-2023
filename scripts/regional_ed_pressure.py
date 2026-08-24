#!/usr/bin/env python3
"""Feature engineering for Montréal-wide emergency-department pressure.

The public MSSS hourly emergency feed contains a rolling seven-day portrait for each
Québec ED.  This module converts the Montréal installation rows into *peer* pressure
features for JGH forecasting.  JGH itself is excluded from the regional aggregates to
avoid simply re-encoding the target hospital's own current state.

Important forecasting constraint: regional ED pressure is not a known-future covariate.
At forecast time only observations at or before the cutoff are available.  Backtests and
production callers should therefore carry the latest observed regional state/trends
forward across the forecast horizon (or replace that persistence assumption with a
separately validated regional-pressure forecast).  Never merge realized future regional
pressure into a historical forecast replay.
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
    "heure_de_lextraction_image": "extraction_time",
    "heure_de_lextraction": "extraction_time",
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


def _slug(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text


def _decode_csv(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


def load_public_feed(source: str | Path = DEFAULT_SOURCE_URL) -> pd.DataFrame:
    """Read the MSSS rolling feed from a URL or local CSV path."""
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        response = requests.get(
            source_text,
            timeout=60,
            headers={"User-Agent": "ed-flow-2023 regional-pressure collector"},
        )
        response.raise_for_status()
        text = _decode_csv(response.content)
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    return pd.read_csv(source_text, sep=None, engine="python")


def normalize_public_feed(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize MSSS field names/types while tolerating minor source-name changes."""
    rename: dict[object, str] = {}
    for column in frame.columns:
        canonical = SOURCE_COLUMNS.get(_slug(column))
        if canonical:
            rename[column] = canonical
    out = frame.rename(columns=rename).copy()

    required = {
        "rss",
        "installation",
        "stretcher_capacity",
        "stretcher_occupied",
        "patients_present",
        "waiting_pec",
        "extraction_time",
    }
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "MSSS emergency feed is missing expected columns after normalization: "
            + ", ".join(sorted(missing))
        )

    rss_numeric = pd.to_numeric(out["rss"], errors="coerce")
    out["rss"] = rss_numeric.map(lambda value: f"{int(value):02d}" if pd.notna(value) else "")
    out["installation"] = out["installation"].fillna("").astype(str).str.strip()
    if "establishment" in out:
        out["establishment"] = out["establishment"].fillna("").astype(str).str.strip()
    if "permit" in out:
        out["permit"] = out["permit"].fillna("").astype(str).str.strip()

    numeric = [
        "stretcher_capacity",
        "stretcher_occupied",
        "stretcher_24h",
        "stretcher_48h",
        "patients_present",
        "waiting_pec",
        "dms_stretcher",
        "dms_ambulatory",
    ]
    for column in numeric:
        if column in out:
            values = out[column].astype(str).str.replace(",", ".", regex=False)
            out[column] = pd.to_numeric(values, errors="coerce")

    # The source uses Québec-local civil time.  Keep timestamps naive to match the
    # project's existing America/Montreal hourly flow tables and DST handling.
    out["ds"] = pd.to_datetime(out["extraction_time"], errors="coerce", dayfirst=True)
    if out["ds"].isna().all() and "updated_at" in out:
        out["ds"] = pd.to_datetime(out["updated_at"], errors="coerce", dayfirst=True)
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
    tokens = _normalized_patterns(patterns)
    mask = pd.Series(False, index=installation.index)
    for token in tokens:
        mask |= normalized.str.contains(re.escape(token), regex=True, na=False)
    return mask


def build_regional_peer_pressure(
    raw: pd.DataFrame,
    *,
    region_code: str = DEFAULT_REGION_CODE,
    jgh_patterns: Iterable[str] = DEFAULT_JGH_PATTERNS,
) -> pd.DataFrame:
    """Aggregate Montréal peer ED state by hour, explicitly excluding JGH."""
    source = normalize_public_feed(raw)
    rows = source.loc[source["rss"].eq(str(region_code).zfill(2))].copy()
    if rows.empty:
        raise ValueError(f"No MSSS emergency rows found for RSS {region_code}")

    lower_installation = rows["installation"].str.lower()
    rows = rows.loc[
        ~lower_installation.isin({"total régional", "total regional", "ensemble du québec"})
    ].copy()
    rows = rows.loc[~is_jgh_installation(rows["installation"], jgh_patterns)].copy()
    if rows.empty:
        raise ValueError("No Montréal peer ED rows remain after excluding JGH")

    capacity = pd.to_numeric(rows["stretcher_capacity"], errors="coerce")
    occupied = pd.to_numeric(rows["stretcher_occupied"], errors="coerce")
    rows["installation_occupancy"] = occupied / capacity.where(capacity > 0)

    grouped = rows.groupby("ds", sort=True)
    result = grouped.agg(
        regional_peer_installations=("installation", "nunique"),
        regional_stretcher_capacity=("stretcher_capacity", "sum"),
        regional_stretcher_occupied=("stretcher_occupied", "sum"),
        regional_patients_present=("patients_present", "sum"),
        regional_waiting_pec=("waiting_pec", "sum"),
        regional_max_stretcher_occupancy=("installation_occupancy", "max"),
        regional_mean_stretcher_occupancy=("installation_occupancy", "mean"),
    ).reset_index()

    for source_column, output_column in (
        ("stretcher_24h", "regional_stretcher_24h"),
        ("stretcher_48h", "regional_stretcher_48h"),
    ):
        if source_column in rows:
            counts = grouped[source_column].sum(min_count=1).rename(output_column).reset_index()
            result = result.merge(counts, on="ds", how="left")
        else:
            result[output_column] = np.nan

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

    result["regional_stretcher_occupancy"] = (
        result["regional_stretcher_occupied"]
        / result["regional_stretcher_capacity"].where(result["regional_stretcher_capacity"] > 0)
    )
    result["regional_waiting_share"] = (
        result["regional_waiting_pec"]
        / result["regional_patients_present"].where(result["regional_patients_present"] > 0)
    )

    result = result.sort_values("ds").drop_duplicates("ds", keep="last").reset_index(drop=True)
    for column in TREND_BASE_COLUMNS:
        values = pd.to_numeric(result[column], errors="coerce")
        for lag in (1, 3, 6):
            result[f"{column}_delta_{lag}h"] = values.diff(lag)

    for column in REGIONAL_FEATURE_COLUMNS:
        if column not in result:
            result[column] = np.nan
    return result[["ds", *REGIONAL_FEATURE_COLUMNS]]


def persistence_future(
    pressure_history: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    horizon: int,
) -> pd.DataFrame:
    """Create leakage-safe known-future rows by persisting the cutoff state/trends."""
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
