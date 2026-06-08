"""Tests for spectral colorimetry calculations."""

from __future__ import annotations

import numpy as np
import pytest

from metamerism.colorimetry import (
    normalize_xyz_to_illuminant,
    spectral_distance,
    xyz,
)
from metamerism.core import (
    CANONICAL_GRID,
    DomainError,
    GridMismatchError,
    Observer,
    Spectrum,
    SpectrumDomain,
    SpectrumError,
)


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


def simple_observer(weights: np.ndarray | None = None) -> Observer:
    cmf = np.column_stack(
        [
            np.ones_like(CANONICAL_GRID),
            np.full_like(CANONICAL_GRID, 2.0),
            np.full_like(CANONICAL_GRID, 3.0),
        ]
    )
    integration_weights = np.ones_like(CANONICAL_GRID) if weights is None else weights
    return Observer(
        name="simple",
        cmf=cmf,
        integration_weights=integration_weights,
    )


def test_xyz_uses_observer_integration_weights():
    reflectance = spectrum(1.0, SpectrumDomain.REFLECTANCE)
    illuminant = spectrum(1.0, SpectrumDomain.ILLUMINANT)

    result = xyz(reflectance, illuminant, simple_observer())

    assert result == pytest.approx([81.0, 162.0, 243.0])


def test_xyz_requires_reflectance_domain():
    reflectance = spectrum(1.0, SpectrumDomain.TRANSMITTANCE)
    illuminant = spectrum(1.0, SpectrumDomain.ILLUMINANT)

    with pytest.raises(DomainError, match="REFLECTANCE"):
        xyz(reflectance, illuminant, simple_observer())


def test_xyz_requires_illuminant_domain():
    reflectance = spectrum(1.0, SpectrumDomain.REFLECTANCE)
    illuminant = spectrum(1.0, SpectrumDomain.EMISSION)

    with pytest.raises(DomainError, match="ILLUMINANT"):
        xyz(reflectance, illuminant, simple_observer())


def test_xyz_requires_matching_grids():
    reflectance = spectrum(1.0, SpectrumDomain.REFLECTANCE)
    illuminant = spectrum(
        1.0,
        SpectrumDomain.ILLUMINANT,
        np.linspace(380.0, 780.0, 41),
    )

    with pytest.raises(GridMismatchError):
        xyz(reflectance, illuminant, simple_observer())


def test_normalize_xyz_to_illuminant_defaults_to_relative_y_one():
    reflectance = spectrum(0.5, SpectrumDomain.REFLECTANCE)
    illuminant = spectrum(1.0, SpectrumDomain.ILLUMINANT)
    observer = simple_observer()

    result = normalize_xyz_to_illuminant(
        xyz(reflectance, illuminant, observer),
        illuminant,
        observer,
    )

    assert result == pytest.approx([0.25, 0.5, 0.75])


def test_normalize_xyz_to_illuminant_allows_percentage_scale():
    reflectance = spectrum(0.5, SpectrumDomain.REFLECTANCE)
    illuminant = spectrum(1.0, SpectrumDomain.ILLUMINANT)
    observer = simple_observer()

    result = normalize_xyz_to_illuminant(
        xyz(reflectance, illuminant, observer),
        illuminant,
        observer,
        white_y=100.0,
    )

    assert result == pytest.approx([25.0, 50.0, 75.0])


def test_normalize_xyz_to_illuminant_rejects_zero_luminance_white():
    illuminant = spectrum(0.0, SpectrumDomain.ILLUMINANT)
    observer = simple_observer()

    with pytest.raises(SpectrumError, match="zero-luminance"):
        normalize_xyz_to_illuminant(np.ones(3), illuminant, observer)


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
