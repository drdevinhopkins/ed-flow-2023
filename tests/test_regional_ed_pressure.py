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
    parse_quebec_status_html,
    persistence_future,
    normalize_public_feed,
    load_public_feed,
)


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



def test_datastore_clock_uses_publication_date() -> None:
    frame = source_frame().iloc[:4].copy()
    frame["Heure_de_l’extraction_(image)"] = ["13:00:00", "23:00:00", "12:00", "13:00:00"]
    frame["Mise_a_jour"] = ["2026-10-25T13:45:00", "2026-10-26T00:45:00",
                            "2026-10-25T12:45:00", "2026-10-25T13:45:00"]
    parsed = normalize_public_feed(frame).set_index("installation")
    assert parsed.loc["Peer A", "ds"] == pd.Timestamp("2026-10-25 23:00")
    assert parsed.loc["Peer B", "ds"] == pd.Timestamp("2026-10-25 12:00")
    assert parsed.loc["Non-Montreal ED", "ds"] == pd.Timestamp("2026-10-25 13:00")
    try:
        normalize_public_feed(frame.drop(columns="Mise_a_jour"))
    except ValueError as exc:
        assert "dated Mise_a_jour" in str(exc)
    else:
        raise AssertionError("Clock-only timestamps must not get today's date")



def test_iso_dates_and_missing_publication_dates() -> None:
    frame = source_frame().iloc[:4].copy()
    frame["Heure_de_l’extraction_(image)"] = [
        "2026-10-05 13:00:00", "13:00:00", "13:00:00", "13:00:00"
    ]
    frame["Mise_a_jour"] = ["2026-10-05T13:45:00", "2026-10-05T13:45:00", None, None]
    parsed = normalize_public_feed(frame)
    assert len(parsed) == 2
    assert parsed["ds"].eq(pd.Timestamp("2026-10-05 13:00")).all()


def test_empty_datastore_uses_official_csv_export() -> None:
    from quebec_ed_status import CKAN_DUMP_URL, load_official_fallback

    empty = Mock()
    empty.json.return_value = {"success": True, "result": {"records": []}}
    export = Mock(content=source_frame().to_csv(index=False).encode("utf-8"))
    with patch("quebec_ed_status.requests.get", side_effect=[empty, export]) as get:
        with patch("quebec_ed_status.load_quebec_status") as status:
            frame = load_official_fallback()
    assert get.call_args_list[1].args[0] == CKAN_DUMP_URL
    status.assert_not_called()
    pressure = build_regional_peer_pressure(frame)
    assert len(pressure) == 7
    assert pressure.iloc[0]["regional_patients_present"] == 70


def test_blocked_status_page_is_rejected() -> None:
    from quebec_ed_status import load_quebec_status

    blocked = Mock(text="<html><title>Verify you are human</title></html>", content=b"x" * 2015)
    with patch("quebec_ed_status.requests.get", return_value=blocked) as get:
        try:
            load_quebec_status()
        except ValueError as exc:
            assert "no valid observation timestamp" in str(exc)
        else:
            raise AssertionError("Access challenge must not become an observation")
    assert get.call_count == 1


def test_later_blocked_page_does_not_publish_partial_region() -> None:
    from quebec_ed_status import load_quebec_status

    valid = Mock(
        text="""
        <p>Dernière mise à jour : lundi 26 octobre 2026 à 13 h 00</p>
        <section><strong>Peer A</strong><span>Montréal</span>
          <a href="?nofiche=1">Consulter l’horaire de l’installation Peer A</a>
          <div>Nombre de personnes qui attendent de voir un médecin à l’urgence : 5</div>
          <div>Nombre total de personnes à l’urgence : 40</div>
          <div>Taux d’occupation des civières : 100 %</div>
        </section>
        """,
        content=b"valid",
    )
    blocked = Mock(text="<html>Access denied</html>", content=b"blocked")
    with patch("quebec_ed_status.requests.get", side_effect=[valid, blocked]):
        try:
            load_quebec_status()
        except ValueError as exc:
            assert "no valid observation timestamp" in str(exc)
        else:
            raise AssertionError("A blocked later page must fail the whole collection")


def test_default_source_falls_back_once_and_custom_source_fails() -> None:
    import requests

    with patch("regional_ed_pressure.requests.get", side_effect=requests.HTTPError("403")):
        with patch("quebec_ed_status.load_official_fallback", return_value=source_frame()) as fallback:
            assert not load_public_feed().empty
            fallback.assert_called_once()
            try:
                load_public_feed("https://example.org/custom.csv")
            except requests.HTTPError:
                pass
            else:
                raise AssertionError("Custom source failures must propagate")
            fallback.assert_called_once()


def test_failed_official_sources_report_all_errors() -> None:
    from quebec_ed_status import load_official_fallback

    with patch("quebec_ed_status.load_ckan_datastore", side_effect=ValueError("empty API")):
        with patch("quebec_ed_status.load_ckan_dump", side_effect=ValueError("bad CSV")):
            with patch("quebec_ed_status.load_quebec_status", side_effect=ValueError("blocked HTML")):
                try:
                    load_official_fallback()
                except RuntimeError as exc:
                    assert all(message in str(exc) for message in ("empty API", "bad CSV", "blocked HTML"))
                else:
                    raise AssertionError("No source should produce no archive update")


def main() -> None:
    test_peer_aggregation_excludes_jgh_and_other_regions()
    test_quebec_status_fallback_uses_schedule_link_card()
    test_trends_are_backward_looking()
    test_future_uses_only_cutoff_state()
    test_datastore_clock_uses_publication_date()
    test_iso_dates_and_missing_publication_dates()
    test_empty_datastore_uses_official_csv_export()
    test_blocked_status_page_is_rejected()
    test_later_blocked_page_does_not_publish_partial_region()
    test_default_source_falls_back_once_and_custom_source_fails()
    test_failed_official_sources_report_all_errors()
    print("regional ED pressure tests passed")


if __name__ == "__main__":
    main()
