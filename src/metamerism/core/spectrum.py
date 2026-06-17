"""
metamerism.core.spectrum
~~~~~~~~~~~~~~~~~~~~~~~~

Project metadata around colour-science spectral distributions.

`colour-science` owns spectral representation and numerical operations.
`Spectrum` only adds domain and provenance metadata needed by this project.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import colour
import numpy as np
from numpy.typing import NDArray

from metamerism.core.exceptions import DomainError

CANONICAL_GRID: NDArray[np.float64] = np.linspace(380.0, 780.0, 81)
"""Convenience grid for tests and examples: 380-780 nm at 5 nm."""

PROJECT_SHAPE = colour.SpectralShape(380.0, 780.0, 5.0)
"""Project-preferred spectral shape for explicit colour-science alignment."""


class SpectrumDomain(enum.Enum):
    """Physical meaning of a spectral distribution's values."""

    REFLECTANCE = "reflectance"
    K_S = "ks"
    ILLUMINANT = "illuminant"
    CMF = "cmf"
    EMISSION = "emission"
    TRANSMITTANCE = "transmittance"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Provenance:
    """Immutable record of where a spectrum came from."""

    source: str = "unknown"
    instrument: str | None = None
    measurement_geometry: str | None = None
    illuminant_used: str | None = None
    substrate: str | None = None
    medium: str | None = None
    film_thickness_um: float | None = None
    confidence: float = 1.0
    notes: str | None = None

    def with_note(self, note: str) -> Provenance:
        """Return a copy with *note* appended to the notes field."""
        existing = self.notes or ""
        separator = "; " if existing else ""
        return Provenance(
            source=self.source,
            instrument=self.instrument,
            measurement_geometry=self.measurement_geometry,
            illuminant_used=self.illuminant_used,
            substrate=self.substrate,
            medium=self.medium,
            film_thickness_um=self.film_thickness_um,
            confidence=self.confidence,
            notes=existing + separator + note,
        )

    def with_confidence(self, confidence: float) -> Provenance:
        """Return a copy with updated confidence."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")
        return Provenance(
            source=self.source,
            instrument=self.instrument,
            measurement_geometry=self.measurement_geometry,
            illuminant_used=self.illuminant_used,
            substrate=self.substrate,
            medium=self.medium,
            film_thickness_um=self.film_thickness_um,
            confidence=confidence,
            notes=self.notes,
        )


@dataclass(frozen=True)
class Spectrum:
    """A colour-science spectral distribution with project metadata."""

    sd: colour.SpectralDistribution
    domain: SpectrumDomain = SpectrumDomain.UNKNOWN
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def from_arrays(
        cls,
        wavelengths: NDArray[np.float64] | list[float],
        values: NDArray[np.float64] | list[float],
        *,
        domain: SpectrumDomain = SpectrumDomain.UNKNOWN,
        provenance: Provenance | None = None,
        validate: bool = True,
    ) -> Spectrum:
        """Construct a Spectrum from sampled wavelength/value arrays."""
        wl = np.asarray(wavelengths, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)

        if wl.ndim != 1:
            raise ValueError(f"wavelengths must be 1-D, got shape {wl.shape}")
        if v.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {v.shape}")
        if len(wl) != len(v):
            raise ValueError(
                "wavelengths and values must have the same length, "
                f"got {len(wl)} and {len(v)}"
            )
        if len(wl) < 2:
            raise ValueError("A spectrum must have at least two samples")
        if not np.all(np.diff(wl) > 0):
            raise ValueError("wavelengths must be strictly monotonically increasing")
        if validate:
            _validate_domain_values(v, domain)

        prov = provenance or Provenance()
        sd = colour.SpectralDistribution(
            {float(w): float(value) for w, value in zip(wl, v, strict=True)},
            name=prov.source,
        )
        return cls(sd=sd, domain=domain, provenance=prov)

    @classmethod
    def from_colour(
        cls,
        sd: colour.SpectralDistribution,
        *,
        domain: SpectrumDomain = SpectrumDomain.UNKNOWN,
        provenance: Provenance | None = None,
        validate: bool = True,
    ) -> Spectrum:
        """Construct a Spectrum from a colour-science spectral distribution."""
        if not isinstance(sd, colour.SpectralDistribution):
            sd = colour.SpectralDistribution(sd)
        values = np.asarray(sd.values, dtype=np.float64)
        if validate:
            _validate_domain_values(values, domain)
        prov = provenance or Provenance(source=getattr(sd, "name", "colour.sd"))
        sd_copy = sd.copy()
        sd_copy.name = prov.source
        return cls(sd=sd_copy, domain=domain, provenance=prov)

    @property
    def wavelengths(self) -> NDArray[np.float64]:
        """Wavelength samples from the underlying colour distribution."""
        return np.asarray(self.sd.wavelengths, dtype=np.float64)

    @property
    def values(self) -> NDArray[np.float64]:
        """Spectral values from the underlying colour distribution."""
        return np.asarray(self.sd.values, dtype=np.float64)

    @property
    def n_samples(self) -> int:
        """Number of wavelength samples."""
        return len(self.wavelengths)

    @property
    def wl_min(self) -> float:
        """Minimum wavelength in nanometres."""
        return float(self.wavelengths[0])

    @property
    def wl_max(self) -> float:
        """Maximum wavelength in nanometres."""
        return float(self.wavelengths[-1])

    @property
    def spacing(self) -> float | None:
        """Uniform spacing in nanometres, or None if the grid is non-uniform."""
        diffs = np.diff(self.wavelengths)
        if np.allclose(diffs, diffs[0], rtol=1e-6):
            return float(diffs[0])
        return None

    def is_on_canonical_grid(self) -> bool:
        """Return True if this spectrum matches :data:`CANONICAL_GRID`."""
        return len(self.wavelengths) == len(CANONICAL_GRID) and bool(
            np.allclose(self.wavelengths, CANONICAL_GRID, atol=1e-9)
        )

    def allclose(
        self,
        other: Spectrum,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        """Return True if grids and values are element-wise close."""
        if len(self.wavelengths) != len(other.wavelengths):
            return False
        if not np.allclose(self.wavelengths, other.wavelengths, atol=1e-9):
            return False
        return bool(np.allclose(self.values, other.values, rtol=rtol, atol=atol))

    def __eq__(self, other: object) -> bool:
        """Exact equality of wavelengths, values, and domain."""
        if not isinstance(other, Spectrum):
            return NotImplemented
        return (
            np.array_equal(self.wavelengths, other.wavelengths)
            and np.array_equal(self.values, other.values)
            and self.domain == other.domain
        )

    def __hash__(self) -> int:
        return hash((self.wavelengths.tobytes(), self.values.tobytes(), self.domain))

    def __repr__(self) -> str:
        spacing = self.spacing
        grid_desc = (
            f"{spacing:.1f} nm spacing" if spacing is not None else "non-uniform"
        )
        return (
            "Spectrum("
            f"domain={self.domain.value!r}, "
            f"range=[{self.wl_min:.0f}, {self.wl_max:.0f}] nm, "
            f"n={self.n_samples}, "
            f"{grid_desc}, "
            f"confidence={self.provenance.confidence:.2f}"
            ")"
        )


def _validate_domain_values(
    values: NDArray[np.float64],
    domain: SpectrumDomain,
) -> None:
    """Raise DomainError if *values* violate constraints for *domain*."""
    if domain in (SpectrumDomain.REFLECTANCE, SpectrumDomain.TRANSMITTANCE):
        if np.any(values < -1e-6):
            raise DomainError(
                f"Negative values found in {domain.value} spectrum "
                f"(min={values.min():.6g}). "
                "Reflectance and transmittance must be in [0, 1]."
            )
        if np.any(values > 1.0 + 1e-6):
            raise DomainError(
                f"Values exceeding 1 found in {domain.value} spectrum "
                f"(max={values.max():.6g}). "
                "Reflectance and transmittance must be in [0, 1]."
            )
    elif domain in (SpectrumDomain.ILLUMINANT, SpectrumDomain.EMISSION):
        if np.any(values < -1e-9):
            raise DomainError(
                f"Negative values found in {domain.value} spectrum "
                f"(min={values.min():.6g}). "
                "Power distributions must be non-negative."
            )
