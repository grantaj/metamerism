"""Mixing helpers for spectral paint combinations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metamerism.core import GridMismatchError, Provenance, Spectrum, SpectrumDomain

KS_EPS = 1e-9


@dataclass(frozen=True)
class MixComponent:
    """One spectrum and its mixture weight."""

    spectrum: Spectrum
    weight: float


def _normalise_weights(weights: NDArray[np.float64]) -> NDArray[np.float64]:
    if weights.ndim != 1:
        raise ValueError("weights must be 1-D")
    if len(weights) == 0:
        raise ValueError("at least one weight is required")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")

    total = float(np.sum(weights))
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    return weights / total


def _validate_same_grid(spectra: list[Spectrum]) -> None:
    first = spectra[0]
    for spectrum in spectra[1:]:
        if not (
            first.wavelengths.shape == spectrum.wavelengths.shape
            and np.allclose(first.wavelengths, spectrum.wavelengths)
        ):
            raise GridMismatchError("spectra must share the same wavelength grid")


def _mix_reflectance_values(
    spectra: list[Spectrum],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    matrix = np.stack([spectrum.values for spectrum in spectra], axis=0)
    return np.sum(matrix * weights[:, None], axis=0)


def _reflectance_to_ks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(values, KS_EPS, 1.0)
    return ((1.0 - clipped) ** 2) / (2.0 * clipped)


def _ks_to_reflectance(values: NDArray[np.float64]) -> NDArray[np.float64]:
    ks = np.clip(values, 0.0, None)
    return 1.0 + ks - np.sqrt(ks * ks + 2.0 * ks)


def _arithmetic_ks_mix(
    ks_matrix: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Blend K/S values as a weighted arithmetic mean.

    The correct K-M mixing formula is (K/S)_mix = ΣwᵢKᵢ / ΣwᵢSᵢ, which
    requires K and S separately.  Single-substrate (white-only) drawdown data
    gives only K/S ratios, so K and S cannot be separated without black-substrate
    measurements.  When scattering S is approximately equal across all components
    (true for organic pigments in the same vehicle), the formula reduces to the
    arithmetic mean: (K/S)_mix = Σwᵢ·(K/S)ᵢ.

    A proxy-based S estimate (e.g. 1/min(K/S)) only works for pigments that are
    transparent at some wavelength so that min(K/S) ≈ K_vehicle/S_total.  For
    full-spectrum absorbers (Phthalo Blue, Chromium Oxide Dark, cadmium reds) it
    underestimates S by orders of magnitude, collapsing the mix to the most
    transparent component regardless of proportions.  Arithmetic averaging avoids
    that failure mode and is the physically correct choice given the available data.
    """
    stack = np.stack(ks_matrix, axis=0)
    return np.sum(stack * weights[:, None], axis=0)


def mix_spectra(
    spectra: list[Spectrum],
    weights: list[float] | NDArray[np.float64],
    *,
    mode: str = "reflectance",
    provenance: Provenance | None = None,
) -> Spectrum:
    """Mix spectra using either direct reflectance averaging or K/S blending."""
    if len(spectra) == 0:
        raise ValueError("at least one spectrum is required")

    _validate_same_grid(spectra)

    weights_array = _normalise_weights(np.asarray(weights, dtype=np.float64))
    if len(weights_array) != len(spectra):
        raise ValueError("weights must match the number of spectra")

    if mode == "reflectance":
        if any(
            spectrum.domain is not SpectrumDomain.REFLECTANCE
            for spectrum in spectra
        ):
            raise ValueError("reflectance mixing requires reflectance spectra")
        values = _mix_reflectance_values(spectra, weights_array)
        domain = SpectrumDomain.REFLECTANCE
        model_label = "reflectance"
    elif mode == "ks":
        if any(
            spectrum.domain not in (SpectrumDomain.REFLECTANCE, SpectrumDomain.K_S)
            for spectrum in spectra
        ):
            raise ValueError("K/S mixing requires reflectance or K/S spectra")

        ks_matrix = []
        for spectrum in spectra:
            if spectrum.domain is SpectrumDomain.REFLECTANCE:
                ks_matrix.append(_reflectance_to_ks(spectrum.values))
            else:
                ks_matrix.append(spectrum.values)
        mixed_ks = _arithmetic_ks_mix(ks_matrix, weights_array)
        values = _ks_to_reflectance(mixed_ks)
        domain = SpectrumDomain.REFLECTANCE
        model_label = "K/S"
    else:
        raise ValueError(f"unsupported mixing mode: {mode!r}")

    provenance = provenance or Provenance(source="mixed spectrum")
    component_notes = ", ".join(
        f"{weight:.3f}:{spectrum.provenance.source}"
        for spectrum, weight in zip(spectra, weights_array, strict=True)
    )
    output_provenance = provenance.with_note(
        f"mix mode={model_label}; components={component_notes}"
    )
    return Spectrum.from_arrays(
        spectra[0].wavelengths,
        values,
        domain=domain,
        provenance=output_provenance,
        validate=False,
    )
