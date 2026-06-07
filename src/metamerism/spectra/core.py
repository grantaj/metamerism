"""Core spectral data structures."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SpectralCurve:
    """A sampled spectral curve on a wavelength grid.

    Values may represent reflectance, relative spectral power, colour matching
    functions, or other wavelength-indexed quantities depending on context.
    """

    wavelengths_nm: NDArray[np.float64]
    values: NDArray[np.float64]
    name: str | None = None

    def __post_init__(self) -> None:
        if self.wavelengths_nm.shape != self.values.shape:
            msg = "wavelengths_nm and values must have the same shape"
            raise ValueError(msg)
        if self.wavelengths_nm.ndim != 1:
            msg = "spectral curves must be one-dimensional"
            raise ValueError(msg)
