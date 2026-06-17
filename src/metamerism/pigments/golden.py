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

GOLDEN_KS_WIDE = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "golden_hb_ks_wide.csv"
)
"""Repository path to the cleaned GOLDEN K/S table."""


@dataclass(frozen=True)
class GoldenPaintSpectrum:
    """A GOLDEN Heavy Body paint with measured reflectance and K/S spectra."""

    product_id: int
    paint_name: str
    reflectance: Spectrum
    ks: Spectrum


def _read_wide_rows(
    csv_path: Path,
) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file)


def _spectrum_from_row(
    row: dict[str, str],
    *,
    csv_name: str,
    domain: SpectrumDomain,
) -> Spectrum:
    wavelengths: list[float] = []
    values: list[float] = []

    for key, value in row.items():
        if not key.startswith("wl_"):
            continue
        wavelengths.append(float(key.removeprefix("wl_")))
        values.append(float(value))

    product_id = int(row["product_id"])
    paint_name = row["paint_name"]
    domain_label = "reflectance" if domain is SpectrumDomain.REFLECTANCE else "K/S"
    provenance = Provenance(
        source=f"GOLDEN Heavy Body {domain_label}: {paint_name}",
        notes=f"product_id={product_id}; source={csv_name}",
    )
    return Spectrum.from_arrays(
        wavelengths,
        values,
        domain=domain,
        provenance=provenance,
    )


def load_golden_heavy_body_paints(
    reflectance_csv_path: Path | None = None,
    ks_csv_path: Path | None = None,
) -> list[GoldenPaintSpectrum]:
    """Load GOLDEN Heavy Body paints with measured reflectance and K/S spectra."""
    reflectance_path = reflectance_csv_path or GOLDEN_REFLECTANCE_FRACTION_WIDE
    ks_path = ks_csv_path or GOLDEN_KS_WIDE

    reflectance_rows = {
        int(row["product_id"]): row for row in _read_wide_rows(reflectance_path)
    }
    ks_rows = {int(row["product_id"]): row for row in _read_wide_rows(ks_path)}

    if reflectance_rows.keys() != ks_rows.keys():
        raise ValueError("reflectance and K/S tables must contain the same paints")

    paints: list[GoldenPaintSpectrum] = []
    for product_id, reflectance_row in reflectance_rows.items():
        ks_row = ks_rows[product_id]
        if reflectance_row["paint_name"] != ks_row["paint_name"]:
            raise ValueError(
                "reflectance and K/S tables disagree on paint names for "
                f"product_id={product_id}"
            )

        paints.append(
            GoldenPaintSpectrum(
                product_id=product_id,
                paint_name=reflectance_row["paint_name"],
                reflectance=_spectrum_from_row(
                    reflectance_row,
                    csv_name=reflectance_path.name,
                    domain=SpectrumDomain.REFLECTANCE,
                ),
                ks=_spectrum_from_row(
                    ks_row,
                    csv_name=ks_path.name,
                    domain=SpectrumDomain.K_S,
                ),
            )
        )

    return paints


def load_golden_heavy_body_paint_by_name(
    paint_name: str,
    reflectance_csv_path: Path | None = None,
    ks_csv_path: Path | None = None,
) -> GoldenPaintSpectrum:
    """Load one GOLDEN Heavy Body paint by exact name."""
    matches = [
        paint
        for paint in load_golden_heavy_body_paints(
            reflectance_csv_path=reflectance_csv_path,
            ks_csv_path=ks_csv_path,
        )
        if paint.paint_name == paint_name
    ]
    if not matches:
        raise KeyError(f"Paint not found: {paint_name!r}")
    if len(matches) > 1:
        raise ValueError(f"Multiple paints matched {paint_name!r}")
    return matches[0]
