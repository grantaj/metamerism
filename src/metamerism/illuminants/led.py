"""Programmable LED illuminant models."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metamerism.spectra.core import SpectralCurve


@dataclass(frozen=True)
class LEDChannel:
    """A single LED spectral basis channel."""

    name: str
    spectrum: SpectralCurve


@dataclass(frozen=True)
class LEDSource:
    """A programmable multi-channel LED source.

    The illuminant is modelled as:

        I(lambda; u) = sum_j u_j E_j(lambda)

    where E_j are measured or modelled channel spectra.
    """

    name: str
    channels: tuple[LEDChannel, ...]

    def illuminant(self, drive: NDArray[np.float64]) -> SpectralCurve:
        """Return the illuminant spectrum for a channel drive vector."""
        if drive.shape != (len(self.channels),):
            msg = "drive vector length must match number of LED channels"
            raise ValueError(msg)

        wavelengths = self.channels[0].spectrum.wavelengths_nm
        values = np.zeros_like(wavelengths, dtype=np.float64)

        for coeff, channel in zip(drive, self.channels, strict=True):
            if not np.array_equal(channel.spectrum.wavelengths_nm, wavelengths):
                msg = "all LED channel spectra must use the same wavelength grid"
                raise ValueError(msg)
            values += coeff * channel.spectrum.values

        return SpectralCurve(
            wavelengths_nm=wavelengths,
            values=values,
            name=f"{self.name} illuminant",
        )
