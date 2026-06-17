"""Tests for spectral mixing helpers."""

from __future__ import annotations

import numpy as np
import pytest

from metamerism.core import (
    CANONICAL_GRID,
    GridMismatchError,
    Provenance,
    Spectrum,
    SpectrumDomain,
)
from metamerism.mixing import mix_spectra


def spectrum(
    value: float,
    *,
    domain: SpectrumDomain = SpectrumDomain.REFLECTANCE,
) -> Spectrum:
    return Spectrum.from_arrays(
        CANONICAL_GRID,
        np.full_like(CANONICAL_GRID, value, dtype=np.float64),
        domain=domain,
        provenance=Provenance(source=f"{domain.value}:{value}"),
    )


def test_reflectance_mix_is_weighted_average():
    red = spectrum(0.2)
    yellow = spectrum(0.8)

    mix = mix_spectra([red, yellow], [0.25, 0.75], mode="reflectance")

    assert mix.domain is SpectrumDomain.REFLECTANCE
    assert mix.values[0] == pytest.approx(0.65)
    assert "mix mode=reflectance" in (mix.provenance.notes or "")


def test_ks_mix_converts_and_returns_reflectance():
    dark = spectrum(0.2)
    light = spectrum(0.8)

    mix = mix_spectra([dark, light], [0.5, 0.5], mode="ks")

    ks_dark = ((1.0 - 0.2) ** 2) / (2.0 * 0.2)
    ks_light = ((1.0 - 0.8) ** 2) / (2.0 * 0.8)
    expected_ks = 0.5 * (ks_dark + ks_light)
    expected_reflectance = 1.0 + expected_ks - np.sqrt(
        expected_ks**2 + 2.0 * expected_ks
    )

    assert mix.domain is SpectrumDomain.REFLECTANCE
    assert mix.values[0] == pytest.approx(expected_reflectance)
    assert "mix mode=K/S" in (mix.provenance.notes or "")


def test_mix_requires_matching_grids():
    other_grid = np.linspace(400.0, 700.0, 31)
    a = spectrum(0.2)
    b = Spectrum.from_arrays(
        other_grid,
        np.full_like(other_grid, 0.8, dtype=np.float64),
        domain=SpectrumDomain.REFLECTANCE,
    )

    with pytest.raises(GridMismatchError):
        mix_spectra([a, b], [0.5, 0.5])


def test_mix_rejects_negative_weights():
    a = spectrum(0.2)
    b = spectrum(0.8)

    with pytest.raises(ValueError, match="non-negative"):
        mix_spectra([a, b], [1.0, -1.0])
