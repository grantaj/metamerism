"""Tests for GOLDEN Heavy Body loaders."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from metamerism.core import SpectrumDomain
from metamerism.pigments import (
    GOLDEN_REFLECTANCE_FRACTION_WIDE,
    GoldenPaintSpectrum,
    load_golden_heavy_body_reflectance,
    load_golden_heavy_body_reflectance_by_name,
)


def _first_csv_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return next(csv.DictReader(file))


def test_loader_reads_all_paints():
    paints = load_golden_heavy_body_reflectance()

    assert len(paints) == 78
    assert all(isinstance(paint, GoldenPaintSpectrum) for paint in paints)


def test_loader_preserves_first_row():
    paint = load_golden_heavy_body_reflectance()[0]
    row = _first_csv_row(GOLDEN_REFLECTANCE_FRACTION_WIDE)

    assert paint.product_id == int(row["product_id"])
    assert paint.paint_name == row["paint_name"]
    assert paint.spectrum.domain is SpectrumDomain.REFLECTANCE
    assert paint.spectrum.n_samples == 31
    assert paint.spectrum.wavelengths[0] == pytest.approx(400.0)
    assert paint.spectrum.wavelengths[-1] == pytest.approx(700.0)
    assert paint.spectrum.values[0] == pytest.approx(float(row["wl_400"]))


def test_loader_finds_named_paint():
    paint = load_golden_heavy_body_reflectance_by_name("Alizarin Crimson Hue")

    assert paint.product_id == 1450
    assert paint.paint_name == "Alizarin Crimson Hue"
    assert "product_id=1450" in (paint.spectrum.provenance.notes or "")
    assert paint.spectrum.provenance.source == "GOLDEN Heavy Body: Alizarin Crimson Hue"
