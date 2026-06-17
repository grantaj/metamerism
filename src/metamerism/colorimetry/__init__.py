"""Project-specific colorimetry helpers.

Use `colour-science` directly for standard colour operations such as
`sd_to_XYZ`, `XYZ_to_Lab`, `XYZ_to_sRGB`, illuminants, observers, and ΔE.
"""

from metamerism.colorimetry.difference import spectral_distance

__all__ = ["spectral_distance"]
