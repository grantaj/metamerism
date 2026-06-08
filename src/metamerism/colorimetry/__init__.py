"""Colorimetry module: spectral to colour conversions.

This module computes perceived colour from spectral data using standard
colorimetric functions. All computation functions work with numpy arrays
following scientific Python conventions (scipy, pandas, colour-science).

Workflow::

    import numpy as np
    from colour import XYZ_to_Lab, XYZ_to_sRGB
    from metamerism.core.observer import CIE_1931_2_DEGREE
    from metamerism.colorimetry import normalize_xyz_to_illuminant, xyz, delta_e_76

    # Compute XYZ from spectra (only novel calculation in metamerism)
    raw_xyz = xyz(reflectance, illuminant, CIE_1931_2_DEGREE)  # shape (3,)
    xyz_array = normalize_xyz_to_illuminant(
        raw_xyz,
        illuminant,
        CIE_1931_2_DEGREE,
    )

    # For colour space conversions, use colour-science directly
    lab_array = XYZ_to_Lab(xyz_array)
    srgb_array = XYZ_to_sRGB(xyz_array)

    # Compare colours
    delta_e = delta_e_76(lab_a, lab_b)
"""

from metamerism.colorimetry.difference import (
    D65_WHITEPOINT,
    delta_e_00,
    delta_e_76,
    spectral_distance,
)
from metamerism.colorimetry.tristimulus import (
    normalize_xyz_to_illuminant,
    xyz,
)

__all__ = [
    "D65_WHITEPOINT",
    "delta_e_00",
    "delta_e_76",
    "normalize_xyz_to_illuminant",
    "spectral_distance",
    "xyz",
]
