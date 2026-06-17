"""Loaders for GOLDEN Heavy Body spectral data."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from metamerism.core import Provenance, Spectrum, SpectrumDomain

GOLDEN_REFLECTANCE_FRACTION_WIDE = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "golden_hb_reflectance_fraction_wide.csv"
)
"""Repository path to the cleaned GOLDEN reflectance table."""


@dataclass(frozen=True)
class GoldenPaintSpectrum:
    """A GOLDEN Heavy Body paint paired with its reflectance spectrum."""

    product_id: int
    paint_name: str
    spectrum: Spectrum


def _read_reflectance_rows(
    csv_path: Path,
) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def load_golden_heavy_body_reflectance(
    csv_path: Path | None = None,
) -> list[GoldenPaintSpectrum]:
    """Load GOLDEN Heavy Body reflectance spectra from the cleaned CSV export."""
    path = csv_path or GOLDEN_REFLECTANCE_FRACTION_WIDE
    paints: list[GoldenPaintSpectrum] = []

    for row in _read_reflectance_rows(path):
        product_id = int(row["product_id"])
        paint_name = row["paint_name"]
        wavelengths: list[float] = []
        values: list[float] = []

        for key, value in row.items():
            if not key.startswith("wl_"):
                continue
            wavelengths.append(float(key.removeprefix("wl_")))
            values.append(float(value))

        provenance = Provenance(
            source=f"GOLDEN Heavy Body: {paint_name}",
            notes=f"product_id={product_id}; source={path.name}",
        )
        spectrum = Spectrum.from_arrays(
            wavelengths,
            values,
            domain=SpectrumDomain.REFLECTANCE,
            provenance=provenance,
        )
        paints.append(
            GoldenPaintSpectrum(
                product_id=product_id,
                paint_name=paint_name,
                spectrum=spectrum,
            )
        )

    return paints


def load_golden_heavy_body_reflectance_by_name(
    paint_name: str,
    csv_path: Path | None = None,
) -> GoldenPaintSpectrum:
    """Load one GOLDEN Heavy Body paint by exact name."""
    matches = [
        paint
        for paint in load_golden_heavy_body_reflectance(csv_path=csv_path)
        if paint.paint_name == paint_name
    ]
    if not matches:
        raise KeyError(f"Paint not found: {paint_name!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple paints matched {paint_name!r}")
    return matches[0]
