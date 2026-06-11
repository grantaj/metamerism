"""
metamerism.core.spectrum
~~~~~~~~~~~~~~~~~~~~~~~~

Provenance-aware spectrum container and project spectral constants.

`Spectrum` is an immutable wrapper around sampled spectral data with
project-specific domain and provenance metadata. Numerical spectral
operations are intentionally delegated to `colour-science`.

Typical usage::

    from metamerism.core.spectrum import Spectrum, PROJECT_SHAPE, SpectrumDomain

    r = Spectrum.from_arrays(wavelengths, values, domain=SpectrumDomain.REFLECTANCE)
    sd = r.to_colour().copy().align(PROJECT_SHAPE)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import colour
import numpy as np
from numpy.typing import NDArray

from metamerism.core.exceptions import (
    DomainError,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Canonical wavelength grid
# ---------------------------------------------------------------------------

#: The single authoritative wavelength grid used for all colorimetric work.
#:
#: * Range  : 380-780 nm  (covers the full photopic visible range)
#: * Spacing: 5 nm        (matches CIE standard observer tables exactly)
#: * Points : 81
#:
#: Data from other grids (e.g. GOLDEN 400-700 nm at 10 nm) must be
#: resampled onto this grid on ingestion.  Resampling is logged via the
#: Provenance attached to the resulting Spectrum.
CANONICAL_GRID: NDArray[np.float64] = np.linspace(380.0, 780.0, 81)

# Project-preferred spectral shape expressed for convenience.  This is a
# convenience alias to a colour.SpectralShape rather than a bespoke internal
# storage model; use it when aligning or constructing spectral distributions.
PROJECT_SHAPE = colour.SpectralShape(380.0, 780.0, 5.0)


# ---------------------------------------------------------------------------
# Domain enumeration
# ---------------------------------------------------------------------------


class SpectrumDomain(enum.Enum):
    """Physical meaning of the sampled values.

    The domain determines valid value ranges and downstream usage intent.
    It is stored as metadata and checked by colorimetric functions that
    require a specific domain.
    """

    REFLECTANCE = "reflectance"
    """Fraction of incident power reflected: values in [0, 1]."""

    ILLUMINANT = "illuminant"
    """Relative spectral power distribution: values >= 0, arbitrary scale."""

    CMF = "cmf"
    """Colour matching function: one column per observer primary."""

    EMISSION = "emission"
    """Absolute or relative emitted power: values >= 0."""

    TRANSMITTANCE = "transmittance"
    """Fraction of incident power transmitted: values in [0, 1]."""

    UNKNOWN = "unknown"
    """Domain not specified; no range checks applied."""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Immutable record of where a spectrum came from.

    Every :class:`Spectrum` carries a Provenance.  Operations that
    produce new spectra (resampling, mixing, scaling) record what was
    done in the *notes* field rather than erasing the chain of custody.

    All fields are optional so that provenance can be built up
    incrementally, but *source* should always be provided.
    """

    source: str = "unknown"
    """Human-readable description of the data origin.

    Examples: ``"GOLDEN Heavy Body Cadmium Red Medium, manufacturer data"``,
    ``"CIE 1931 2-degree standard observer"``,
    ``"Gaussian approximation, WS2812B datasheet"``.
    """

    instrument: str | None = None
    """Measurement instrument, if applicable.  E.g. ``"X-Rite i1Pro 3"``."""

    measurement_geometry: str | None = None
    """Measurement geometry.  E.g. ``"45°/0°"``, ``"di:8°"``."""

    illuminant_used: str | None = None
    """Illuminant used during measurement, if applicable.  E.g. ``"D65"``."""

    substrate: str | None = None
    """Substrate description for paint measurements."""

    medium: str | None = None
    """Paint medium, if applicable.  E.g. ``"Golden Acrylic Glazing Liquid"``."""

    film_thickness_um: float | None = None
    """Approximate film thickness in micrometres, if known."""

    confidence: float = 1.0
    """Subjective confidence in [0, 1].

    * 1.0 - directly measured under controlled conditions
    * 0.7 - manufacturer or reference data, assumed reliable
    * 0.4 - approximated (e.g. Gaussian LED model, extrapolated values)
    * 0.1 - very rough guess or placeholder

    Colorimetric functions may propagate or warn on low confidence values.
    """

    notes: str | None = None
    """Free-text notes, including resampling history."""

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


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrum:
    """An immutable spectrum with project metadata.

    Parameters
    ----------
    wavelengths:
        Monotonically increasing wavelength values in nanometres.
        Must have the same length as *values*.
    values:
        Sampled spectral values.  Interpretation depends on *domain*.
    domain:
        Physical meaning of the values; determines valid ranges and
        downstream usage semantics.
    provenance:
        Record of data origin and processing history.

    Numerical operations should be performed on the underlying
    :class:`colour.SpectralDistribution` from :meth:`to_colour`.
    """

    wavelengths: NDArray[np.float64]
    values: NDArray[np.float64]
    domain: SpectrumDomain = SpectrumDomain.UNKNOWN
    provenance: Provenance = field(default_factory=Provenance)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

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
        """Construct a Spectrum from raw arrays, with validation.

        Parameters
        ----------
        wavelengths:
            Wavelength values in nanometres.
        values:
            Spectral values corresponding to each wavelength.
        domain:
            Physical domain of the spectrum.
        provenance:
            Provenance record.  Defaults to an empty Provenance.
        validate:
            If True (default), check value ranges for the given domain
            and raise :class:`DomainError` on violations.
        """
        wl = np.asarray(wavelengths, dtype=np.float64)
        v = np.asarray(values, dtype=np.float64)

        if wl.ndim != 1:
            raise ValueError(f"wavelengths must be 1-D, got shape {wl.shape}")
        if v.ndim != 1:
            raise ValueError(f"values must be 1-D, got shape {v.shape}")
        if len(wl) != len(v):
            raise ValueError(
                f"wavelengths and values must have the same length, "
                f"got {len(wl)} and {len(v)}"
            )
        if len(wl) < 2:
            raise ValueError("A spectrum must have at least two samples")
        if not np.all(np.diff(wl) > 0):
            raise ValueError("wavelengths must be strictly monotonically increasing")

        if validate:
            _validate_domain_values(v, domain)

        return cls(
            wavelengths=wl,
            values=v,
            domain=domain,
            provenance=provenance or Provenance(),
        )

    # ------------------------------------------------------------------
    # Interop with colour-science (Phase 1)
    # ------------------------------------------------------------------

    @classmethod
    def from_colour(
        cls,
        sd: "colour.SpectralDistribution",
        *,
        domain: SpectrumDomain = SpectrumDomain.UNKNOWN,
        provenance: Provenance | None = None,
    ) -> "Spectrum":
        """Construct a Spectrum from a colour.SpectralDistribution.

        This is a thin interop helper that copies wavelength and value data
        from an existing `colour` object and preserves provenance.
        """
        # Allow anything accepted by colour to be passed through
        if not isinstance(sd, colour.SpectralDistribution):
            sd = colour.SpectralDistribution(sd)

        wl = np.asarray(sd.wavelengths, dtype=np.float64)
        v = np.asarray(sd.values, dtype=np.float64)

        prov = provenance or Provenance(source=getattr(sd, "name", "colour.sd"))
        return cls(wavelengths=wl, values=v, domain=domain, provenance=prov)

    def to_colour(self, name: str | None = None) -> "colour.SpectralDistribution":
        """Return a colour.SpectralDistribution representing this Spectrum.

        The returned object uses the Spectrum provenance.source as the name
        unless *name* is provided.
        """
        data = {float(w): float(v) for w, v in zip(self.wavelengths, self.values)}
        sd = colour.SpectralDistribution(data, name=name or self.provenance.source)
        return sd

    @property
    def sd(self) -> "colour.SpectralDistribution":
        """Convenience property returning a colour SpectralDistribution."""
        return self.to_colour()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
        """Return True if this spectrum is already on :data:`CANONICAL_GRID`."""
        return len(self.wavelengths) == len(CANONICAL_GRID) and bool(
            np.allclose(self.wavelengths, CANONICAL_GRID, atol=1e-9)
        )

    # ------------------------------------------------------------------
    # Comparison and identity
    # ------------------------------------------------------------------

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
        """Exact equality of wavelengths and values (use allclose for numerics)."""
        if not isinstance(other, Spectrum):
            return NotImplemented
        return (
            np.array_equal(self.wavelengths, other.wavelengths)
            and np.array_equal(self.values, other.values)
            and self.domain == other.domain
        )

    def __hash__(self) -> int:
        return hash((self.wavelengths.tobytes(), self.values.tobytes(), self.domain))

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        spacing = self.spacing
        grid_desc = (
            f"{spacing:.1f} nm spacing" if spacing is not None else "non-uniform"
        )
        return (
            f"Spectrum("
            f"domain={self.domain.value!r}, "
            f"range=[{self.wl_min:.0f}, {self.wl_max:.0f}] nm, "
            f"n={self.n_samples}, "
            f"{grid_desc}, "
            f"confidence={self.provenance.confidence:.2f}"
            f")"
        )


# ---------------------------------------------------------------------------
# Domain validation (private)
# ---------------------------------------------------------------------------


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
                f"Reflectance and transmittance must be in [0, 1]."
            )
        if np.any(values > 1.0 + 1e-6):
            raise DomainError(
                f"Values exceeding 1 found in {domain.value} spectrum "
                f"(max={values.max():.6g}). "
                f"Reflectance and transmittance must be in [0, 1]."
            )
    elif domain in (SpectrumDomain.ILLUMINANT, SpectrumDomain.EMISSION):
        if np.any(values < -1e-9):
            raise DomainError(
                f"Negative values found in {domain.value} spectrum "
                f"(min={values.min():.6g}). "
                f"Power distributions must be non-negative."
            )
