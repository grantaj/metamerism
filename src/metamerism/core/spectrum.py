"""
metamerism.core.spectrum
~~~~~~~~~~~~~~~~~~~~~~~~

Immutable spectral value type and canonical wavelength grid.

All spectral data in this package is represented as a :class:`Spectrum`.
A Spectrum is a frozen, self-describing vector of values sampled on a
wavelength grid measured in nanometres.  Operations that change a
spectrum (resampling, scaling, pointwise products) return new Spectrum
objects; no method mutates the receiver.

The canonical internal grid is defined by :data:`CANONICAL_GRID`.  All
spectra should be resampled onto this grid before any colorimetric
calculation.  Resampling is always explicit; the package never silently
converts between grids.

Typical usage::

    from metamerism.core.spectrum import Spectrum, CANONICAL_GRID, SpectrumDomain

    r = Spectrum.from_arrays(wavelengths, values, domain=SpectrumDomain.REFLECTANCE)
    r_canon = r.resample(CANONICAL_GRID)
    product  = r_canon * illuminant_canon   # element-wise (Hadamard)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import colour
from numpy.typing import NDArray
from scipy.interpolate import interp1d
# Note: interpolation usage remains only to support legacy tests during
# phased migration. Long term such operations should call into colour's
# interpolation/extrapolation helpers or operate on SpectralDistribution objects.

from metamerism.core.exceptions import (
    DomainError,
    ExtrapolationError,
    GridMismatchError,
    SpectrumError,
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

    The domain determines valid value ranges and appropriate interpolation
    behaviour.  It is stored as metadata and checked by colorimetric
    functions that require a specific domain.
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
# Interpolation strategy
# ---------------------------------------------------------------------------


class InterpolationMethod(enum.Enum):
    """Interpolation strategy used when resampling a spectrum.

    The choice of method is recorded in provenance.
    """

    LINEAR = "linear"
    """Piecewise linear.  Safe for illuminants and CMFs."""

    CUBIC = "cubic"
    """Cubic spline.  Preferred for reflectance; smoother transitions."""

    PCHIP = "pchip"
    """Piecewise cubic Hermite.  Monotone; avoids overshoot at sharp edges."""


_DEFAULT_INTERPOLATION: dict[SpectrumDomain, InterpolationMethod] = {
    SpectrumDomain.REFLECTANCE: InterpolationMethod.PCHIP,
    SpectrumDomain.ILLUMINANT: InterpolationMethod.LINEAR,
    SpectrumDomain.CMF: InterpolationMethod.LINEAR,
    SpectrumDomain.EMISSION: InterpolationMethod.LINEAR,
    SpectrumDomain.TRANSMITTANCE: InterpolationMethod.PCHIP,
    SpectrumDomain.UNKNOWN: InterpolationMethod.LINEAR,
}


# ---------------------------------------------------------------------------
# Extrapolation policy
# ---------------------------------------------------------------------------


class ExtrapolationPolicy(enum.Enum):
    """What to do when resampling requires values outside the source range."""

    ZERO = "zero"
    """Fill with zero.  Appropriate for illuminants and emission spectra."""

    CLAMP = "clamp"
    """Repeat the nearest boundary value."""

    RAISE = "raise"
    """Raise :class:`ExtrapolationError`."""


_DEFAULT_EXTRAPOLATION: dict[SpectrumDomain, ExtrapolationPolicy] = {
    SpectrumDomain.REFLECTANCE: ExtrapolationPolicy.CLAMP,
    SpectrumDomain.ILLUMINANT: ExtrapolationPolicy.ZERO,
    SpectrumDomain.CMF: ExtrapolationPolicy.ZERO,
    SpectrumDomain.EMISSION: ExtrapolationPolicy.ZERO,
    SpectrumDomain.TRANSMITTANCE: ExtrapolationPolicy.CLAMP,
    SpectrumDomain.UNKNOWN: ExtrapolationPolicy.ZERO,
}


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Spectrum:
    """An immutable, self-describing spectral vector.

    Parameters
    ----------
    wavelengths:
        Monotonically increasing wavelength values in nanometres.
        Must have the same length as *values*.
    values:
        Sampled spectral values.  Interpretation depends on *domain*.
    domain:
        Physical meaning of the values; determines valid ranges and
        default interpolation behaviour.
    provenance:
        Record of data origin and processing history.

    Notes
    -----
    Construct via :meth:`from_arrays` rather than directly, to benefit
    from input validation.

    All arithmetic operators return new Spectrum objects on the same
    wavelength grid.  Operations on spectra with different grids raise
    :class:`GridMismatchError`; call :meth:`resample` first.
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
            Wavelength values in nanometres.  Need not be on the
            canonical grid; call :meth:`resample` to move to the
            canonical grid.
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
    # Resampling
    # ------------------------------------------------------------------

    def resample(
        self,
        target_grid: NDArray[np.float64],
        *,
        method: InterpolationMethod | None = None,
        extrapolation: ExtrapolationPolicy | None = None,
        reduce_confidence: bool = True,
    ) -> Spectrum:
        """Removed: use metamerism.core.ops.resample_spectrum or colour alignment.

        This method was intentionally removed to eliminate in-library numerical
        resampling logic. Convert to a colour.SpectralDistribution via
        Spectrum.to_colour() and use colour's align/interpolate utilities, or
        call metamerism.core.ops.resample_spectrum.
        """
        raise NotImplementedError(
            "Spectrum.resample was removed; use metamerism.core.ops.resample_spectrum "
            "or convert to a colour.SpectralDistribution via Spectrum.to_colour() and "
            "use colour's alignment helpers."
        )

            )
            new_values = interp_fn(target_grid)
            out_of_range = np.isnan(new_values)

        if np.any(out_of_range):
            if extrapolation is ExtrapolationPolicy.ZERO:
                new_values = np.where(out_of_range, 0.0, new_values)
            elif extrapolation is ExtrapolationPolicy.CLAMP:
                left_mask = target_grid < self.wl_min
                right_mask = target_grid > self.wl_max
                new_values = np.where(left_mask, self.values[0], new_values)
                new_values = np.where(right_mask, self.values[-1], new_values)

        # Build provenance note
        note = (
            f"resampled from [{self.wl_min:.1f}, {self.wl_max:.1f}] nm "
            f"({self.n_samples} pts) to [{target_grid[0]:.1f}, "
            f"{target_grid[-1]:.1f}] nm ({len(target_grid)} pts) "
            f"via {method.value}"
        )
        if needs_extrapolation:
            note += f"; extrapolation: {extrapolation.value}"

        new_provenance = self.provenance.with_note(note)
        if needs_extrapolation and reduce_confidence:
            new_confidence = max(0.0, self.provenance.confidence - 0.1)
            new_provenance = new_provenance.with_confidence(new_confidence)

        return Spectrum(
            wavelengths=target_grid,
            values=new_values,
            domain=self.domain,
            provenance=new_provenance,
        )

    def to_canonical(self) -> Spectrum:
        """Removed: use Spectrum.to_colour().align(PROJECT_SHAPE) or ops helper.

        Kept as an explicit removal point to avoid accidental reliance on
        previously available implicit canonicalisation behavior.
        """
        raise NotImplementedError(
            "Spectrum.to_canonical was removed; align via Spectrum.to_colour().align()"
        )

    # ------------------------------------------------------------------
    # Arithmetic (all require matching grids; resample first)
    # ------------------------------------------------------------------

    def _check_grid_compatibility(self, other: Spectrum) -> None:
        if len(self.wavelengths) != len(other.wavelengths) or not np.allclose(
            self.wavelengths, other.wavelengths, atol=1e-9
        ):
            raise GridMismatchError(
                f"Spectra have different wavelength grids. "
                f"Self: [{self.wl_min:.1f}, {self.wl_max:.1f}] nm "
                f"({self.n_samples} pts), "
                f"Other: [{other.wl_min:.1f}, {other.wl_max:.1f}] nm "
                f"({other.n_samples} pts). "
                f"Call .resample() or .to_canonical() first."
            )

    def _combined_provenance(self, other: Spectrum, operation: str) -> Provenance:
        combined_confidence = min(
            self.provenance.confidence, other.provenance.confidence
        )
        note = (
            f"{operation} of '{self.provenance.source}' and '{other.provenance.source}'"
        )
        return Provenance(
            source=f"computed: {note}",
            confidence=combined_confidence,
            notes=note,
        )

    def __mul__(self, other: Spectrum | float | int) -> Spectrum:
        raise NotImplementedError(
            "Spectrum.__mul__ removed; use metamerism.core.ops.multiply_spectra or "
            "operate on colour.SpectralDistribution via Spectrum.to_colour()."
        )

    def __rmul__(self, other: float | int) -> Spectrum:
        return self.__mul__(other)

    def __add__(self, other: Spectrum | float | int) -> Spectrum:
        raise NotImplementedError(
            "Spectrum.__add__ removed; use metamerism.core.ops.add_spectra or "
            "operate on colour.SpectralDistribution via Spectrum.to_colour()."
        )

    def __radd__(self, other: float | int) -> Spectrum:
        return self.__add__(other)

    def __sub__(self, other: Spectrum | float | int) -> Spectrum:
        raise NotImplementedError(
            "Spectrum.__sub__ removed; use metamerism.core.ops.subtract_spectra or "
            "operate on colour.SpectralDistribution via Spectrum.to_colour()."
        )

    def __truediv__(self, other: float | int) -> Spectrum:
        """Scale by the reciprocal of a scalar."""
        if other == 0:
            raise ZeroDivisionError("Cannot divide a Spectrum by zero")
        return self.__mul__(1.0 / other)

    def __neg__(self) -> Spectrum:
        return Spectrum(
            wavelengths=self.wavelengths,
            values=-self.values,
            domain=self.domain,
            provenance=self.provenance.with_note("negated"),
        )

    # ------------------------------------------------------------------
    # Normalisation and clipping
    # ------------------------------------------------------------------

    def normalise(self, *, to: float = 1.0) -> Spectrum:
        """Removed: normalisation should be performed on colour.SpectralDistribution or via ops.

        This method was removed to centralise numerical operations outside the
        thin Spectrum wrapper. Use metamerism.core.ops.normalise_spectrum or
        convert to a colour.SpectralDistribution and call its normalise method.
        """
        raise NotImplementedError(
            "Spectrum.normalise was removed; normalise on the underlying colour.SpectralDistribution "
            "or use metamerism.core.ops.normalise_spectrum."
        )

    def clip(
        self,
        low: float = 0.0,
        high: float = 1.0,
        *,
        warn: bool = True,
    ) -> Spectrum:
        """Return a copy with values clipped to [*low*, *high*].

        Parameters
        ----------
        low, high:
            Clip bounds.
        warn:
            If True and any clipping occurs, append a note to provenance.
        """
        clipped = np.clip(self.values, low, high)
        n_clipped = int(np.sum(clipped != self.values))
        prov = self.provenance
        if warn and n_clipped > 0:
            prov = prov.with_note(f"clipped {n_clipped} value(s) to [{low}, {high}]")
        return Spectrum(
            wavelengths=self.wavelengths,
            values=clipped,
            domain=self.domain,
            provenance=prov,
        )

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def integrate(self) -> float:
        """Removed: integration should be done via colour or ops helpers.

        Use metamerism.core.ops.integrate_spectrum or convert to a
        colour.SpectralDistribution and use colour's integration utilities.
        """
        raise NotImplementedError(
            "Spectrum.integrate was removed; integrate on the underlying colour.SpectralDistribution "
            "or use metamerism.core.ops.integrate_spectrum."
        )

    def dot(self, other: Spectrum) -> float:
        """Removed: use colour alignment and ops for dot/integration.

        Use metamerism.core.ops.dot_spectra or align both spectra as
        colour.SpectralDistribution objects and compute the integral there.
        """
        raise NotImplementedError(
            "Spectrum.dot was removed; compute integrals on the underlying colour.SpectralDistribution "
            "objects or use metamerism.core.ops.dot_spectra."
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
