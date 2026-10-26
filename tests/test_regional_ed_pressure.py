#!/usr/bin/env python3
from __future__ import annotations

import sys
from unittest.mock import Mock, patch
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from regional_ed_pressure import (  # noqa: E402
    REGIONAL_FEATURE_COLUMNS,
    build_regional_peer_pressure,
    normalize_public_feed,
    parse_quebec_status_html,
    persistence_future,
)
import quebec_ed_status  # noqa: E402


def source_frame() -> pd.DataFrame:
    rows = []
    hours = pd.date_range("2026-08-23 10:00", periods=7, freq="h")
    for index, ds in enumerate(hours):
        rows.extend(
            [
                {
                    "RSS": 6,
                    "Nom_installation": "Hôpital général juif - Sir Mortimer B. Davis",
                    "Nombre_de_civieres_fonctionnelles": 40,
                    "Nombre_de_civieres_occupees": 60 + index,
                    "Nombre_de_patients_sur_civieres_plus_de_24_heures": 15,
                    "Nombre_de_patients_sur_civieres_plus_de_48_heures": 5,
                    "Nombre_total_de_patients_presents_a_lurgence": 100,
                    "Nombre_total_de_patients_en_attente_de_PEC": 12,
                    "Heure_de_l’extraction_(image)": ds,
                },
                {
                    "RSS": 6,
                    "Nom_installation": "Peer A",
                    "Nombre_de_civieres_fonctionnelles": 20,
                    "Nombre_de_civieres_occupees": 20 + index,
                    "Nombre_de_patients_sur_civieres_plus_de_24_heures": 4 + index,
                    "Nombre_de_patients_sur_civieres_plus_de_48_heures": 1,
                    "Nombre_total_de_patients_presents_a_lurgence": 40 + index,
                    "Nombre_total_de_patients_en_attente_de_PEC": 5 + index,
                    "Heure_de_l’extraction_(image)": ds,
                },
                {
                    "RSS": 6,
                    "Nom_installation": "Peer B",
                    "Nombre_de_civieres_fonctionnelles": 10,
                    "Nombre_de_civieres_occupees": 15 + index,
                    "Nombre_de_patients_sur_civieres_plus_de_24_heures": 2,
                    "Nombre_de_patients_sur_civieres_plus_de_48_heures": 1 + index,
                    "Nombre_total_de_patients_presents_a_lurgence": 30 + index,
                    "Nombre_total_de_patients_en_attente_de_PEC": 3,
                    "Heure_de_l’extraction_(image)": ds,
                },
                {
                    "RSS": 5,
                    "Nom_installation": "Non-Montreal ED",
                    "Nombre_de_civieres_fonctionnelles": 99,
                    "Nombre_de_civieres_occupees": 99,
                    "Nombre_de_patients_sur_civieres_plus_de_24_heures": 99,
                    "Nombre_de_patients_sur_civieres_plus_de_48_heures": 99,
                    "Nombre_total_de_patients_presents_a_lurgence": 99,
                    "Nombre_total_de_patients_en_attente_de_PEC": 99,
                    "Heure_de_l’extraction_(image)": ds,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_peer_aggregation_excludes_jgh_and_other_regions() -> None:
    pressure = build_regional_peer_pressure(source_frame())
    first = pressure.iloc[0]
    assert int(first["regional_peer_installations"]) == 2
    assert first["regional_stretcher_capacity"] == 30
    assert first["regional_stretcher_occupied"] == 35
    assert np.isclose(first["regional_stretcher_occupancy"], 35 / 30)
    assert first["regional_patients_present"] == 70
    assert first["regional_waiting_pec"] == 8
    assert first["regional_stretcher_24h"] == 6
    assert first["regional_stretcher_48h"] == 2
    assert int(first["regional_ed_over_100pct"]) == 1
    assert int(first["regional_ed_over_120pct"]) == 1
    assert int(first["regional_ed_over_150pct"]) == 0


def test_quebec_status_fallback_uses_schedule_link_card() -> None:
    html = """
    <html><body>
      <p>Dernière mise à jour : dimanche 23 août à 23 h 00</p>
      <ul>
        <li class="result-card">
          <strong>Centre hospitalier de l'Université de Montréal</strong>
          <span>1000 rue Saint-Denis, Montréal, H2X 0A9</span><span>Montréal</span>
          <a href="https://sante.gouv.qc.ca/foo">Consulter l’horaire de l’installation Centre hospitalier de l'Université de Montréal</a>
          <div>Nombre de personnes qui attendent de voir un médecin à l’urgence : 35</div>
          <div>Nombre total de personnes à l’urgence : 105</div>
          <div>Taux d’occupation des civières : 122 %</div>
        </li>
        <li class="result-card">
          <strong>Centre hospitalier de Laval</strong><span>Laval</span>
          <a href="https://sante.gouv.qc.ca/bar">Consulter l’horaire de l’installation Centre hospitalier de Laval</a>
          <div>Nombre de personnes qui attendent de voir un médecin à l’urgence : 99</div>
          <div>Nombre total de personnes à l’urgence : 199</div>
          <div>Taux d’occupation des civières : 150 %</div>
        </li>
      </ul>
    </body></html>
    """
    parsed = parse_quebec_status_html(html)
    assert len(parsed) == 1
    row = parsed.iloc[0]
    assert row["installation"] == "Centre hospitalier de l'Université de Montréal"
    assert row["patients_present"] == 105
    assert row["waiting_pec"] == 35
    assert np.isclose(row["installation_occupancy"], 1.22)
    assert row["extraction_time"] == pd.Timestamp("2026-08-23 23:00")


def test_trends_are_backward_looking() -> None:
    pressure = build_regional_peer_pressure(source_frame())
    row = pressure.iloc[6]
    assert np.isclose(row["regional_patients_present_delta_1h"], 2.0)
    assert np.isclose(row["regional_patients_present_delta_3h"], 6.0)
    assert np.isclose(row["regional_patients_present_delta_6h"], 12.0)
    assert np.isnan(pressure.iloc[0]["regional_patients_present_delta_1h"])


def test_future_uses_only_cutoff_state() -> None:
    pressure = build_regional_peer_pressure(source_frame())
    cutoff = pd.Timestamp("2026-08-23 15:00")
    future = persistence_future(pressure, cutoff=cutoff, horizon=4)
    assert list(future["ds"]) == list(pd.date_range("2026-08-23 16:00", periods=4, freq="h"))
    cutoff_row = pressure.loc[pressure["ds"].eq(cutoff)].iloc[-1]
    for column in REGIONAL_FEATURE_COLUMNS:
        left = pd.to_numeric(future[column], errors="coerce")
        expected = cutoff_row[column]
        if pd.isna(expected):
            assert left.isna().all()
        else:
            assert np.allclose(left.to_numpy(dtype=float), float(expected))


def test_clock_only_extraction_uses_source_date() -> None:
    frame = source_frame().iloc[:4].copy()
    frame["Heure_de_l’extraction_(image)"] = "23:00:00"
    frame["Mise_a_jour"] = "2026-10-02T00:45:00"
    normalized = normalize_public_feed(frame)
    assert normalized["ds"].eq(pd.Timestamp("2026-10-01 23:00")).all()
    frame["Heure_de_l’extraction_(image)"] = "13:00"
    normalized = normalize_public_feed(frame)
    assert normalized["ds"].eq(pd.Timestamp("2026-10-01 13:00")).all()
    # Without an explicit source date, a clock cannot establish freshness.
    assert normalize_public_feed(frame.drop(columns="Mise_a_jour")).empty


def test_timestamp_fallback_is_per_row() -> None:
    frame = source_frame().iloc[:4].copy()
    frame["Mise_a_jour"] = "2026-08-23T10:45:00"
    frame.loc[0, "Heure_de_l’extraction_(image)"] = None
    normalized = normalize_public_feed(frame)
    assert normalized["ds"].eq(pd.Timestamp("2026-08-23 10:00")).all()


def test_ckan_retries_empty_response() -> None:
    empty = Mock()
    empty.json.return_value = {"success": True, "result": {"records": []}}
    valid = Mock()
    valid.json.return_value = {
        "success": True, "result": {"records": [{"RSS": 6}], "total": 1},
    }
    with patch.object(quebec_ed_status.requests, "get", side_effect=[empty, valid]) as get:
        with patch.object(quebec_ed_status.time, "sleep") as sleep:
            assert len(quebec_ed_status.load_ckan_datastore()) == 1
    assert get.call_count == 2
    sleep.assert_called_once_with(2)


def test_ckan_empty_retries_are_bounded() -> None:
    empty = Mock()
    empty.json.return_value = {"success": True, "result": {"records": []}}
    with patch.object(quebec_ed_status.requests, "get", return_value=empty) as get:
        with patch.object(quebec_ed_status.time, "sleep") as sleep:
            try:
                quebec_ed_status.load_ckan_datastore()
            except ValueError as exc:
                assert "after 3 attempts" in str(exc)
            else:
                raise AssertionError("An empty datastore must fail")
    assert get.call_count == 3
    assert sleep.call_count == 2


def test_official_csv_export_and_fallback_order() -> None:
    response = Mock(content=source_frame().to_csv(index=False).encode("utf-8"))
    with patch.object(quebec_ed_status.requests, "get", return_value=response):
        assert len(quebec_ed_status.load_ckan_csv()) == len(source_frame())
    with patch.object(quebec_ed_status, "load_ckan_datastore", side_effect=ValueError("empty")):
        with patch.object(quebec_ed_status, "load_ckan_csv", return_value=source_frame()) as csv:
            with patch.object(quebec_ed_status, "load_quebec_status") as html:
                assert not quebec_ed_status.load_official_fallback().empty
    csv.assert_called_once()
    html.assert_not_called()


def test_official_fallback_rejects_bad_schema_and_fails_closed() -> None:
    with patch.object(quebec_ed_status, "load_ckan_datastore", return_value=pd.DataFrame({"bad": [1]})):
        with patch.object(quebec_ed_status, "load_ckan_csv", side_effect=ValueError("blocked")):
            with patch.object(quebec_ed_status, "load_quebec_status", side_effect=ValueError("no cards")):
                try:
                    quebec_ed_status.load_official_fallback()
                except RuntimeError as exc:
                    assert all(name in str(exc) for name in ("CKAN", "CKAN CSV", "Quebec.ca"))
                else:
                    raise AssertionError("Unavailable feeds must fail without fabricated rows")


def test_csv_export_rejects_html_challenge() -> None:
    response = Mock(content=b"<html>JavaScript is disabled</html>")
    with patch.object(quebec_ed_status.requests, "get", return_value=response):
        try:
            quebec_ed_status.load_ckan_csv()
        except ValueError:
            pass
        else:
            raise AssertionError("A challenge page is not an ED feed")


def main() -> None:
    test_peer_aggregation_excludes_jgh_and_other_regions()
    test_quebec_status_fallback_uses_schedule_link_card()
    test_trends_are_backward_looking()
    test_future_uses_only_cutoff_state()
    test_clock_only_extraction_uses_source_date()
    test_timestamp_fallback_is_per_row()
    test_ckan_retries_empty_response()
    test_ckan_empty_retries_are_bounded()
    test_official_csv_export_and_fallback_order()
    test_official_fallback_rejects_bad_schema_and_fails_closed()
    test_csv_export_rejects_html_challenge()
    print("regional ED pressure tests passed")


if __name__ == "__main__":
    main()

