"""Color difference metrics.

Provides ΔE76, ΔE00, and spectral distance measures using colour-science.
"""

import numpy as np
from colour.difference import delta_E_CIE1976, delta_E_CIE2000
from numpy.typing import NDArray
from scipy.integrate import trapezoid

from metamerism.core import GridMismatchError, Spectrum, SpectrumDomain

# ---------------------------------------------------------------------------
# Standard D65 whitepoint (2° observer)
# ---------------------------------------------------------------------------

D65_WHITEPOINT = np.array([0.95047, 1.00000, 1.08883])
"""CIE standard D65 illuminant whitepoint (2° observer).

Reference:
    CIE 2004. Colorimetry (3rd ed.). CIE Publication 15.
"""


def delta_e_76(lab_a: NDArray[np.float64], lab_b: NDArray[np.float64]) -> float:
    """Compute CIE ΔE*76 (simple Euclidean distance in Lab).

    Args:
        lab_a: Lab array, shape (3,), as [L, a, b].
        lab_b: Lab array, shape (3,), as [L, a, b].

    Returns:
        ΔE*76 distance (typically 0-5 for small differences).
    """
    return float(delta_E_CIE1976(lab_a, lab_b))


def delta_e_00(lab_a: NDArray[np.float64], lab_b: NDArray[np.float64]) -> float:
    """Compute CIE ΔE*00 (improved color difference metric).

    ΔE*00 is more perceptually uniform than ΔE*76, especially in neutral regions.

    Args:
        lab_a: Lab array, shape (3,), as [L, a, b].
        lab_b: Lab array, shape (3,), as [L, a, b].

    Returns:
        ΔE*00 distance (typically 0-5 for small differences).
    """
    return float(delta_E_CIE2000(lab_a, lab_b))


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
