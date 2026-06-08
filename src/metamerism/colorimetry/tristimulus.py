"""Spectral to tristimulus (XYZ) conversion."""

import numpy as np
from numpy.typing import NDArray

from metamerism.core import (
    CANONICAL_GRID,
    DomainError,
    GridMismatchError,
    Observer,
    Spectrum,
    SpectrumDomain,
    SpectrumError,
)


def xyz(
    reflectance: Spectrum,
    illuminant: Spectrum,
    observer: Observer,
) -> NDArray[np.float64]:
    """Compute CIE XYZ tristimulus values from spectral data.

    Integrates the product of illuminant, reflectance, and colour matching
    functions over the wavelength grid using the trapezoidal rule.

    The standard integration is:
        X = ∫ I(λ) R(λ) x̄(λ) dλ
        Y = ∫ I(λ) R(λ) ȳ(λ) dλ
        Z = ∫ I(λ) R(λ) z̄(λ) dλ

    Args:
        reflectance: Spectrum object with reflectance values. Domain must be
            REFLECTANCE.
        illuminant: Spectrum object with illuminant spectral power distribution.
        observer: Observer object with colour matching functions.

    Returns:
        XYZ as shape (3,) array [X, Y, Z].

    Raises:
        DomainError: If spectrum domains are invalid.
        GridMismatchError: If wavelength grids don't match.
        ValueError: If spectra are not on the canonical grid.
    """
    if reflectance.domain is not SpectrumDomain.REFLECTANCE:
        msg = "reflectance domain must be REFLECTANCE"
        raise DomainError(msg)
    if illuminant.domain is not SpectrumDomain.ILLUMINANT:
        msg = "illuminant domain must be ILLUMINANT"
        raise DomainError(msg)

    if not (
        reflectance.wavelengths.shape == illuminant.wavelengths.shape
        and np.allclose(reflectance.wavelengths, illuminant.wavelengths)
    ):
        msg = "reflectance and illuminant wavelengths must match"
        raise GridMismatchError(msg)

    if not np.allclose(reflectance.wavelengths, CANONICAL_GRID):
        msg = "spectra must be on canonical grid (380-780 nm, 5 nm spacing)"
        raise ValueError(msg)

    product = illuminant.values * reflectance.values
    return np.asarray(product @ (observer.cmf * observer.integration_weights[:, None]))


def normalize_xyz_to_illuminant(
    xyz_values: NDArray[np.float64],
    illuminant: Spectrum,
    observer: Observer,
    *,
    white_y: float = 1.0,
) -> NDArray[np.float64]:
    """Normalize XYZ values relative to a perfect reflecting diffuser.

    A perfect reflecting diffuser (R(λ) = 1.0) under the given illuminant
    defines the white point. This is often called Yn normalization.

    Args:
        xyz_values: XYZ array [X, Y, Z] to normalize.
        illuminant: The illuminant used in the original calculation.
        observer: The observer used in the original calculation.
        white_y: Target Y value for the perfect diffuser. Use 1.0 for
            colour-science's default XYZ conversion functions, or 100.0 for
            percentage-scale CIE workflows.

    Returns:
        Normalized XYZ array relative to the illuminant white.
    """
    # Create a perfect reflecting diffuser spectrum
    perfect_diffuser = Spectrum.from_arrays(
        wavelengths=illuminant.wavelengths,
        values=np.ones_like(illuminant.values),
        domain=SpectrumDomain.REFLECTANCE,
    )

    # Compute its XYZ
    white_xyz = xyz(perfect_diffuser, illuminant, observer)
    if white_xyz[1] <= 0:
        msg = "cannot normalize XYZ with a zero-luminance illuminant white"
        raise SpectrumError(msg)

    scale = white_y / white_xyz[1]
    return xyz_values * scale
