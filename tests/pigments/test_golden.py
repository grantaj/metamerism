"""Tests for GOLDEN Heavy Body loaders."""

from __future__ import annotations

import csv
from pathlib import Path

import colour
import numpy as np
import pytest

from metamerism.core import SpectrumDomain
from metamerism.pigments import (
    GOLDEN_KS_WIDE,
    GOLDEN_PAINTS,
    GOLDEN_REFLECTANCE_FRACTION_WIDE,
    GoldenPaintSpectrum,
    load_golden_heavy_body_paint_by_name,
    load_golden_heavy_body_paints,
)


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return next(csv.DictReader(file))


def test_loader_reads_all_paints():
    paints = load_golden_heavy_body_paints()

    assert len(paints) == 78
    assert all(isinstance(paint, GoldenPaintSpectrum) for paint in paints)


def test_loader_preserves_first_row():
    paint = load_golden_heavy_body_paints()[0]
    row = _first_csv_row(GOLDEN_REFLECTANCE_FRACTION_WIDE)
    ks_row = _first_csv_row(GOLDEN_KS_WIDE)
    paint_row = _first_csv_row(GOLDEN_PAINTS)

    assert paint.product_id == int(row["product_id"])
    assert paint.paint_name == row["paint_name"]
    assert paint.lab_d65_10 == pytest.approx(
        (
            float(paint_row["lab_l_d65_10"]),
            float(paint_row["lab_a_d65_10"]),
            float(paint_row["lab_b_d65_10"]),
        )
    )
    assert paint.reflectance.domain is SpectrumDomain.REFLECTANCE
    assert paint.ks.domain is SpectrumDomain.K_S
    assert paint.reflectance.n_samples == 31
    assert paint.ks.n_samples == 31
    assert paint.reflectance.wavelengths[0] == pytest.approx(400.0)
    assert paint.reflectance.wavelengths[-1] == pytest.approx(700.0)
    assert paint.reflectance.values[0] == pytest.approx(float(row["wl_400"]) / 100.0)
    assert paint.ks.values[0] == pytest.approx(float(ks_row["wl_400"]))


def test_loader_finds_named_paint():
    paint = load_golden_heavy_body_paint_by_name("Alizarin Crimson Hue")

    assert paint.product_id == 1450
    assert paint.paint_name == "Alizarin Crimson Hue"
    assert "product_id=1450" in (paint.reflectance.provenance.notes or "")
    assert (
        paint.reflectance.provenance.source
        == "GOLDEN Heavy Body reflectance: Alizarin Crimson Hue"
    )
    assert paint.ks.provenance.source == "GOLDEN Heavy Body K/S: Alizarin Crimson Hue"


def test_measured_ks_mix_of_blue_and_yellow_is_greenish():
    from metamerism.mixing import mix_spectra
    from metamerism.visualization import perceived_colour_rgb

    cobalt = load_golden_heavy_body_paint_by_name("Cobalt Blue")
    hansa = load_golden_heavy_body_paint_by_name("Hansa Yellow Light")

    mixture = mix_spectra([cobalt.ks, hansa.ks], [1.0, 1.0], mode="ks")
    rgb = perceived_colour_rgb(mixture)

    assert rgb[1] > rgb[0]
    assert rgb[1] > rgb[2]


def test_reflectance_to_lab_matches_measured_data():
    paint_rows = {
        row["paint_name"]: row
        for row in csv.DictReader(GOLDEN_PAINTS.open("r", encoding="utf-8"))
    }
    cmfs = colour.MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
    illuminant = colour.SDS_ILLUMINANTS["D65"]

    for name in ["Anthraquinone Blue", "Cerulean Blue Deep", "Cobalt Blue"]:
        paint = load_golden_heavy_body_paint_by_name(name)
        measured = np.array(
            [
                float(paint_rows[name]["lab_l_d65_10"]),
                float(paint_rows[name]["lab_a_d65_10"]),
                float(paint_rows[name]["lab_b_d65_10"]),
            ],
            dtype=float,
        )
        xyz = colour.sd_to_XYZ(
            paint.reflectance.sd.copy().align(colour.SpectralShape(380, 780, 5)),
            cmfs=cmfs.copy().align(colour.SpectralShape(380, 780, 5)),
            illuminant=illuminant.copy().align(colour.SpectralShape(380, 780, 5)),
        )
        computed = colour.XYZ_to_Lab(xyz)

        assert computed == pytest.approx(measured, abs=0.75)
