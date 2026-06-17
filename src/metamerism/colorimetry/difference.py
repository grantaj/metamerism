"""Project-specific spectral distance metrics."""

import numpy as np
from scipy.integrate import trapezoid

from metamerism.core import GridMismatchError, Spectrum, SpectrumDomain


def spectral_distance(reflectance_a: Spectrum, reflectance_b: Spectrum) -> float:
    """Compute RMS spectral distance between two reflectance spectra.

    This is the root-mean-square difference in reflectance values:

        d = √((1 / Δλ) ∫ (R_a(λ) - R_b(λ))² dλ)

    This metric is independent of viewing conditions and illumination.

    Args:
        reflectance_a: First reflectance spectrum.
        reflectance_b: Second reflectance spectrum.

    Returns:
        RMS spectral distance (typically 0-1 for reflectance).

    Raises:
        GridMismatchError: If wavelength grids don't match.
        ValueError: If spectra are not reflectance spectra.
    """
    if (
        reflectance_a.domain is not SpectrumDomain.REFLECTANCE
        or reflectance_b.domain is not SpectrumDomain.REFLECTANCE
    ):
        msg = "spectral_distance requires reflectance spectra"
        raise ValueError(msg)

    if not (
        reflectance_a.wavelengths.shape == reflectance_b.wavelengths.shape
        and np.allclose(reflectance_a.wavelengths, reflectance_b.wavelengths)
    ):
        msg = "reflectance spectra must have matching wavelength grids"
        raise GridMismatchError(msg)

    squared_diff = (reflectance_a.values - reflectance_b.values) ** 2
    integral = trapezoid(squared_diff, x=reflectance_a.wavelengths)
    wavelength_span = reflectance_a.wl_max - reflectance_a.wl_min
    return float(np.sqrt(integral / wavelength_span))
