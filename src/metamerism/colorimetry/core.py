"""Colour science calculations.

This module will wrap `colour-science` and project-specific assumptions such as
wavelength grids, normalisation, observer choice, and white-point handling.
"""

import numpy as np
from numpy.typing import NDArray


def reflected_spectrum(
    reflectance: NDArray[np.float64],
    illuminant: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the reflected spectrum for a surface under an illuminant."""
    if reflectance.shape != illuminant.shape:
        msg = "reflectance and illuminant must have the same shape"
        raise ValueError(msg)
    return reflectance * illuminant
