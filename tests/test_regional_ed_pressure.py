#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from regional_ed_pressure import (  # noqa: E402
    REGIONAL_FEATURE_COLUMNS,
    build_regional_peer_pressure,
    persistence_future,
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


def main() -> None:
    test_peer_aggregation_excludes_jgh_and_other_regions()
    test_trends_are_backward_looking()
    test_future_uses_only_cutoff_state()
    print("regional ED pressure tests passed")


if __name__ == "__main__":
    main()
