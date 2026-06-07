"""Pigment and paint data structures."""

from dataclasses import dataclass

from metamerism.spectra.core import SpectralCurve


@dataclass(frozen=True)
class Pigment:
    """A paint or pigment with an associated reflectance spectrum."""

    name: str
    reflectance: SpectralCurve
    pigment_code: str | None = None
    brand: str | None = None
    notes: str | None = None
