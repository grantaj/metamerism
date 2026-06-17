"""Tests for project-specific colorimetry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from metamerism.colorimetry import spectral_distance
from metamerism.core import CANONICAL_GRID, GridMismatchError, Spectrum, SpectrumDomain


def spectrum(
    value: float,
    domain: SpectrumDomain,
    grid: np.ndarray | None = None,
) -> Spectrum:
    wavelengths = CANONICAL_GRID if grid is None else grid
    return Spectrum.from_arrays(
        wavelengths,
        np.full_like(wavelengths, value, dtype=np.float64),
        domain=domain,
    )


def test_spectral_distance_is_rms_reflectance_difference():
    reflectance_a = spectrum(0.0, SpectrumDomain.REFLECTANCE)
    reflectance_b = spectrum(1.0, SpectrumDomain.REFLECTANCE)

    assert spectral_distance(reflectance_a, reflectance_b) == pytest.approx(1.0)


def test_spectral_distance_uses_actual_wavelength_grid():
    grid = np.linspace(400.0, 700.0, 31)
    reflectance_a = spectrum(0.0, SpectrumDomain.REFLECTANCE, grid)
    reflectance_b = spectrum(1.0, SpectrumDomain.REFLECTANCE, grid)

    assert spectral_distance(reflectance_a, reflectance_b) == pytest.approx(1.0)


def test_spectral_distance_requires_matching_grids():
    reflectance_a = spectrum(0.0, SpectrumDomain.REFLECTANCE)
    reflectance_b = spectrum(
        1.0,
        SpectrumDomain.REFLECTANCE,
        np.linspace(400.0, 700.0, 31),
    )

    with pytest.raises(GridMismatchError):
        spectral_distance(reflectance_a, reflectance_b)


def test_spectral_distance_requires_reflectance_domains():
    reflectance = spectrum(0.0, SpectrumDomain.REFLECTANCE)
    illuminant = spectrum(1.0, SpectrumDomain.ILLUMINANT)

    with pytest.raises(ValueError, match="reflectance"):
        spectral_distance(reflectance, illuminant)
