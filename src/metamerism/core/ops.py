"""Project helpers operating on Spectra via colour-science.

These utilities are the recommended replacement for the numerical methods
previously implemented directly on :class:`Spectrum`.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import colour

from metamerism.core.spectrum import Spectrum, Provenance, PROJECT_SHAPE, CANONICAL_GRID
from metamerism.core.spectrum import SpectrumDomain


def _combined_provenance(a: Spectrum, b: Spectrum, operation: str) -> Provenance:
    combined_confidence = min(a.provenance.confidence, b.provenance.confidence)
    note = f"{operation} of '{a.provenance.source}' and '{b.provenance.source}'"
    return Provenance(source=f"computed: {note}", confidence=combined_confidence, notes=note)


def _align_sd(sd: colour.SpectralDistribution) -> colour.SpectralDistribution:
    return sd.copy().align(PROJECT_SHAPE)


def multiply_spectra(a: Spectrum, b: Spectrum) -> Spectrum:
    """Pointwise (Hadamard) product using colour alignment.

    Both inputs are aligned to PROJECT_SHAPE using colour, multiplied,
    and returned as a new :class:`Spectrum` with combined provenance.
    """
    sd_a = _align_sd(a.to_colour())
    sd_b = _align_sd(b.to_colour())

    product_values = sd_a.values * sd_b.values
    wavelengths = np.asarray(sd_a.wavelengths)
    data = {float(w): float(v) for w, v in zip(wavelengths, product_values)}
    sd_product = colour.SpectralDistribution(data)

    prov = _combined_provenance(a, b, "pointwise product")
    return Spectrum.from_colour(sd_product, domain=SpectrumDomain.UNKNOWN, provenance=prov)


def add_spectra(a: Spectrum, b: Spectrum) -> Spectrum:
    """Pointwise addition of two spectra after alignment to PROJECT_SHAPE."""
    sd_a = _align_sd(a.to_colour())
    sd_b = _align_sd(b.to_colour())
    sum_values = sd_a.values + sd_b.values
    wavelengths = np.asarray(sd_a.wavelengths)
    data = {float(w): float(v) for w, v in zip(wavelengths, sum_values)}
    sd_sum = colour.SpectralDistribution(data)
    prov = _combined_provenance(a, b, "sum")
    return Spectrum.from_colour(sd_sum, domain=SpectrumDomain.UNKNOWN, provenance=prov)


def subtract_spectra(a: Spectrum, b: Spectrum) -> Spectrum:
    """Pointwise subtraction (a - b) after alignment to PROJECT_SHAPE."""
    sd_a = _align_sd(a.to_colour())
    sd_b = _align_sd(b.to_colour())
    diff_values = sd_a.values - sd_b.values
    wavelengths = np.asarray(sd_a.wavelengths)
    data = {float(w): float(v) for w, v in zip(wavelengths, diff_values)}
    sd_diff = colour.SpectralDistribution(data)
    prov = _combined_provenance(a, b, "difference")
    return Spectrum.from_colour(sd_diff, domain=SpectrumDomain.UNKNOWN, provenance=prov)


def scale_spectrum(s: Spectrum, factor: float) -> Spectrum:
    """Scale a spectrum by a scalar factor after alignment to PROJECT_SHAPE."""
    sd = _align_sd(s.to_colour())
    scaled_values = sd.values * float(factor)
    wavelengths = np.asarray(sd.wavelengths)
    data = {float(w): float(v) for w, v in zip(wavelengths, scaled_values)}
    sd_scaled = colour.SpectralDistribution(data)
    prov = s.provenance.with_note(f"scaled by {factor}")
    return Spectrum.from_colour(sd_scaled, domain=s.domain, provenance=prov)


def dot_spectra(a: Spectrum, b: Spectrum) -> float:
    """Inner product (integral) of two spectra after aligning to PROJECT_SHAPE.

    Returns a scalar float.
    """
    sd_a = _align_sd(a.to_colour())
    sd_b = _align_sd(b.to_colour())
    wavelengths = np.asarray(sd_a.wavelengths)
    values = sd_a.values * sd_b.values
    # Use numpy.trapezoid for compatibility with NumPy 2.0+
    return float(np.trapezoid(values, wavelengths))


def integrate_spectrum(s: Spectrum) -> float:
    """Integrate spectrum after alignment to PROJECT_SHAPE."""
    sd = _align_sd(s.to_colour())
    wavelengths = np.asarray(sd.wavelengths)
    # Use numpy.trapezoid for compatibility with NumPy 2.0+
    return float(np.trapezoid(sd.values, wavelengths))


def normalise_spectrum(s: Spectrum, *, to: float = 1.0) -> Spectrum:
    """Normalise using colour's normalise and return a new Spectrum."""
    sd = _align_sd(s.to_colour())
    sd.normalise()  # colour mutates; normalise to a peak of 1.0 by default
    if to != 1.0:
        # scale values
        sd = colour.SpectralDistribution({float(w): float(v * to) for w, v in zip(sd.wavelengths, sd.values)})
    return Spectrum.from_colour(sd, domain=s.domain, provenance=s.provenance.with_note(f"normalised to {to}"))
