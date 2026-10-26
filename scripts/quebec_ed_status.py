#!/usr/bin/env python3
"""Robust official fallbacks for the Québec emergency-department hourly feed.

The canonical MSSS CSV occasionally rejects GitHub-hosted runners.  We therefore try the
Données Québec CKAN datastore first, then its CSV export, and finally the official
Quebec.ca CPU/MSSS emergency-status page. Empty JSON responses receive bounded retries.
Returned rows use the same normalized fields
accepted by ``regional_ed_pressure.build_regional_peer_pressure``.
"""

from __future__ import annotations

import io
import re
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from regional_ed_pressure import (
    QUEBEC_STATUS_URL, _browser_headers, _decode_csv, _page_timestamp, _slug,
    build_regional_peer_pressure,
)

CKAN_DATASTORE_URL = "https://www.donneesquebec.ca/recherche/api/3/action/datastore_search"
CKAN_RESOURCE_ID = "b256f87f-40ec-4c79-bdba-a23e9c50e741"
CKAN_DUMP_URL = (
    "https://www.donneesquebec.ca/recherche/fr/datastore/dump/"
    + CKAN_RESOURCE_ID
)

WAIT_MARKER = "nombre_de_personnes_qui_attendent_de_voir_un_medecin_a_l_urgence"
TOTAL_MARKER = "nombre_total_de_personnes_a_l_urgence"
OCC_MARKER = "taux_d_occupation_des_civieres"


def load_ckan_datastore(limit: int = 50000) -> pd.DataFrame:
    """Return the Données Québec datastore mirror when the resource is datastore-active."""
    for attempt in range(3):
        response = requests.get(
            CKAN_DATASTORE_URL,
            params={"resource_id": CKAN_RESOURCE_ID, "limit": limit},
            headers=_browser_headers(),
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise ValueError(f"Données Québec datastore_search failed: {payload}")
        result = payload.get("result", {})
        records = result.get("records", [])
        if records:
            if result.get("total", len(records)) > len(records):
                raise ValueError("Données Québec datastore response is truncated")
            break
        if attempt == 2:
            raise ValueError("Données Québec datastore returned no emergency rows after 3 attempts")
        print("Données Québec datastore returned no rows; retrying", flush=True)
        time.sleep(2 ** (attempt + 1))
    frame = pd.DataFrame.from_records(records)
    print(f"Loaded {len(frame)} rows from Données Québec datastore", flush=True)
    return frame


def load_ckan_csv() -> pd.DataFrame:
    """Read the official datastore CSV export when its JSON API is unavailable."""
    response = requests.get(
        CKAN_DUMP_URL, params={"format": "csv"}, headers=_browser_headers(), timeout=90,
    )
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(_decode_csv(response.content)))
    # Reject HTML challenges, empty exports, and incompatible schemas before returning.
    build_regional_peer_pressure(frame)
    print(f"Loaded {len(frame)} rows from Données Québec CSV export", flush=True)
    return frame


def _number_after_marker(text: str, marker: str) -> float:
    """Extract the first numeric value after a French metric label."""
    normalized = text.replace("\xa0", " ").replace("\u202f", " ")
    patterns = {
        WAIT_MARKER: r"Nombre de personnes qui attendent de voir un médecin à l[’']urgence",
        TOTAL_MARKER: r"Nombre total de personnes à l[’']urgence",
        OCC_MARKER: r"Taux d[’']occupation des civières",
    }
    match = re.search(patterns[marker] + r"\s*:?\s*([0-9][0-9 ]*)", normalized, re.I)
    if not match:
        return np.nan
    digits = re.sub(r"\D", "", match.group(1))
    return float(digits) if digits else np.nan


def _schedule_label(link) -> str:
    candidates = [
        link.get_text(" ", strip=True),
        str(link.get("aria-label") or ""),
        str(link.get("title") or ""),
    ]
    return max(candidates, key=len).strip()


def _installation_from_label(label: str) -> str:
    cleaned = re.sub(
        r"^.*?consulter\s+l[’']horaire\s+de\s+l[’']installation\s*",
        "",
        label,
        flags=re.I,
    ).strip()
    return cleaned if cleaned != label.strip() else ""


def _find_card(link):
    """Find the smallest ancestor containing exactly one set of the live metrics."""
    node = link
    best = None
    for _ in range(12):
        node = getattr(node, "parent", None)
        if node is None:
            break
        text = node.get_text(" ", strip=True)
        slug = _slug(text)
        if WAIT_MARKER in slug and TOTAL_MARKER in slug and OCC_MARKER in slug:
            counts = (slug.count(WAIT_MARKER), slug.count(TOTAL_MARKER), slug.count(OCC_MARKER))
            if counts == (1, 1, 1):
                return node
            if best is None:
                best = node
    return best


def _installation_from_card(card, link) -> str:
    label_name = _installation_from_label(_schedule_label(link))
    if label_name:
        return label_name
    # Current Quebec.ca cards put the installation name in the first strong/heading-like
    # element.  Keep several fallbacks because the frontend markup has changed before.
    for selector in ("h2", "h3", "h4", "h5", "strong"):
        candidate = card.find(selector)
        if candidate:
            text = candidate.get_text(" ", strip=True)
            if text and not any(marker in _slug(text) for marker in (WAIT_MARKER, TOTAL_MARKER, OCC_MARKER)):
                return text
    # Last resort: nearest previous non-empty text sibling before the schedule link.
    for previous in link.find_all_previous(string=True, limit=12):
        text = re.sub(r"\s+", " ", str(previous)).strip()
        if not text:
            continue
        slug = _slug(text)
        if "horaire_de_l_installation" in slug or any(marker in slug for marker in (WAIT_MARKER, TOTAL_MARKER, OCC_MARKER)):
            continue
        if re.search(r"\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b", text):
            continue
        if text in {"Montréal", "Montreal"}:
            continue
        return text
    return ""


def parse_status_page(html: str) -> pd.DataFrame:
    """Parse a server-rendered Quebec.ca emergency-status result page."""
    soup = BeautifulSoup(html, "html.parser")
    ds = _page_timestamp(soup.get_text(" ", strip=True))
    rows: list[dict[str, object]] = []

    # The official schedule links currently point to sante.gouv.qc.ca and include a
    # nofiche query parameter.  Text/aria-label matching is retained for markup changes.
    links = []
    for link in soup.find_all("a"):
        href = str(link.get("href") or "")
        combined = " ".join(
            [link.get_text(" ", strip=True), str(link.get("aria-label") or ""), str(link.get("title") or "")]
        )
        if "nofiche=" in href or "horaire_de_l_installation" in _slug(combined):
            links.append(link)

    for link in links:
        card = _find_card(link)
        if card is None:
            continue
        text = card.get_text(" ", strip=True)
        # RSS 06 only.  Use an exact Montréal text token rather than city substrings in
        # institution names or addresses from neighbouring regions.
        tokens = [re.sub(r"\s+", " ", token).strip() for token in card.stripped_strings]
        if not any(token in {"Montréal", "Montreal"} for token in tokens):
            continue
        installation = _installation_from_card(card, link)
        if not installation:
            continue
        waiting = _number_after_marker(text, WAIT_MARKER)
        present = _number_after_marker(text, TOTAL_MARKER)
        occupancy_pct = _number_after_marker(text, OCC_MARKER)
        if not np.isfinite(present) or not np.isfinite(occupancy_pct):
            continue
        rows.append(
            {
                "rss": "06",
                "region": "Montréal",
                "installation": installation,
                "stretcher_capacity": np.nan,
                "stretcher_occupied": np.nan,
                "stretcher_24h": np.nan,
                "stretcher_48h": np.nan,
                "patients_present": present,
                "waiting_pec": waiting,
                "installation_occupancy": occupancy_pct / 100.0,
                "extraction_time": ds,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates("installation", keep="last").reset_index(drop=True)


def load_quebec_status(max_pages: int = 13) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen: set[str] = set()
    diagnostics: list[str] = []
    for page in range(max_pages):
        response = requests.get(
            QUEBEC_STATUS_URL,
            params={"tx_solr[page]": page},
            headers=_browser_headers(),
            timeout=60,
        )
        response.raise_for_status()
        if page == 0:
            body_slug = _slug(response.text)
            diagnostics.append(
                f"page0 bytes={len(response.content)} schedule_marker={'horaire_de_l_installation' in body_slug} "
                f"wait_marker={WAIT_MARKER in body_slug}"
            )
        frame = parse_status_page(response.text)
        if frame.empty:
            continue
        new = frame.loc[~frame["installation"].astype(str).isin(seen)].copy()
        if not new.empty:
            frames.append(new)
            seen.update(new["installation"].astype(str))
    if not frames:
        raise ValueError("Quebec.ca status parser found no Montréal cards; " + "; ".join(diagnostics))
    result = pd.concat(frames, ignore_index=True)
    print(f"Parsed {len(result)} Montréal ED installations from Quebec.ca", flush=True)
    return result


def load_official_fallback() -> pd.DataFrame:
    """Try official structured sources before the paginated HTML status page."""
    errors: list[str] = []
    for name, loader in (
        ("CKAN", load_ckan_datastore),
        ("CKAN CSV", load_ckan_csv),
        ("Quebec.ca", load_quebec_status),
    ):
        try:
            frame = loader()
            build_regional_peer_pressure(frame)
            return frame
        except Exception as exc:
            detail = f"{name} {type(exc).__name__}: {exc}"
            errors.append(detail)
            print(f"Official regional source unavailable: {detail}", flush=True)
    raise RuntimeError("All official regional ED fallbacks failed: " + " | ".join(errors))

