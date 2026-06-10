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

    Prefer delegation to the `colour` library when the provided *observer*
    corresponds to a standard CIE observer loaded from `colour`. For custom
    observers (e.g. tests using a synthetic Observer), fall back to the
    original numeric integration.
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

    # If observer.name corresponds to one of the standard CIE observers we
    # expose, use colour.sd_to_XYZ which is well-tested and flexible.
    std_names = {
        "CIE 1931 2 Degree": "CIE 1931 2 Degree Standard Observer",
        "CIE 2015 2 Degree": "CIE 2015 2 Degree Standard Observer",
        "CIE 2015 10 Degree": "CIE 2015 10 Degree Standard Observer",
    }

    if observer.name in std_names:
        # Align spectra to project-preferred shape and build a product SD.
        sd_ref = reflectance.to_colour().copy().align(PROJECT_SHAPE)
        sd_ill = illuminant.to_colour().copy().align(PROJECT_SHAPE)
        product_values = sd_ill.values * sd_ref.values
        wavelengths = np.asarray(sd_ref.wavelengths)
        data = {float(w): float(v) for w, v in zip(wavelengths, product_values)}
        sd_product = colour.SpectralDistribution(data)

        cmf_key = std_names[observer.name]
        cmfs = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER[cmf_key].copy().align(
            PROJECT_SHAPE
        )

        xyz_vals = colour.sd_to_XYZ(sd_product, cmfs=cmfs)
        return np.asarray(xyz_vals)

    # Fallback: numeric integration using the provided Observer arrays.
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
