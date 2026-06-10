"""Colorimetry module: spectral to colour conversions.

This module computes perceived colour from spectral data. Numerical colour
conversions and colour space transforms are delegated to the
`colour-science` library. metamerism provides provenance-aware wrappers
and small convenience helpers where project semantics are required.

Workflow::

    import numpy as np
    from colour import XYZ_to_Lab, XYZ_to_sRGB
    from metamerism.core import DEFAULT_OBSERVER
    from metamerism.colorimetry import normalize_xyz_to_illuminant, xyz, delta_e_76

    # Compute XYZ from spectra using colour-science where possible
    raw_xyz = xyz(reflectance, illuminant, DEFAULT_OBSERVER)
    xyz_array = normalize_xyz_to_illuminant(
        raw_xyz,
        illuminant,
        DEFAULT_OBSERVER,
    )

    # For colour space conversions, use colour-science directly
    lab_array = XYZ_to_Lab(xyz_array)
    srgb_array = XYZ_to_sRGB(xyz_array)

    # For ΔE, prefer colour-science implementations (wrappers are provided)
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
