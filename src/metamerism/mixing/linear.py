"""Simple linear reflectance mixture model."""

import numpy as np

from metamerism.spectra.core import SpectralCurve


def linear_mix(
    components: tuple[SpectralCurve, ...],
    weights: tuple[float, ...],
    *,
    name: str | None = None,
) -> SpectralCurve:
    """Mix spectra by direct weighted reflectance averaging.

    This is not a physically strong paint-mixing model, but it is useful as a
    baseline and for early pipeline testing.
    """
    if len(components) != len(weights):
        msg = "components and weights must have the same length"
        raise ValueError(msg)
    if not components:
        msg = "at least one component is required"
        raise ValueError(msg)

    wavelengths = components[0].wavelengths_nm
    values = np.zeros_like(wavelengths, dtype=np.float64)

    for component, weight in zip(components, weights, strict=True):
        if not np.array_equal(component.wavelengths_nm, wavelengths):
            msg = "all components must use the same wavelength grid"
            raise ValueError(msg)
        values += weight * component.values

    return SpectralCurve(
        wavelengths_nm=wavelengths,
        values=values,
        name=name,
    )
