"""Pigment data structures and importers."""

from metamerism.pigments.golden import (
    GOLDEN_REFLECTANCE_FRACTION_WIDE,
    GoldenPaintSpectrum,
    load_golden_heavy_body_reflectance,
    load_golden_heavy_body_reflectance_by_name,
)

__all__ = [
    "GOLDEN_REFLECTANCE_FRACTION_WIDE",
    "GoldenPaintSpectrum",
    "load_golden_heavy_body_reflectance",
    "load_golden_heavy_body_reflectance_by_name",
]
