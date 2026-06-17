"""Visualization helpers for spectra and perceived colour."""

from __future__ import annotations

import colour
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from metamerism.core import PROJECT_SHAPE, Spectrum, SpectrumDomain

DEFAULT_ILLUMINANT = colour.SDS_ILLUMINANTS["D65"]
DEFAULT_CMFS = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]


def plot_spectrum(
    spectrum: Spectrum,
    ax: Axes | None = None,
    *,
    label: str | None = None,
    color: str | None = None,
    linewidth: float = 2.0,
) -> Axes:
    """Plot a spectrum as a line chart."""
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(
        spectrum.wavelengths,
        spectrum.values,
        label=label or spectrum.provenance.source,
        color=color,
        linewidth=linewidth,
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(
        "Reflectance"
        if spectrum.domain is SpectrumDomain.REFLECTANCE
        else "Value"
    )
    ax.set_xlim(float(spectrum.wl_min), float(spectrum.wl_max))
    ax.set_ylim(bottom=min(0.0, float(np.min(spectrum.values))))
    return ax


def perceived_colour_rgb(
    reflectance: Spectrum,
    *,
    illuminant: colour.SpectralDistribution = DEFAULT_ILLUMINANT,
    cmfs: colour.MultiSpectralDistributions = DEFAULT_CMFS,
) -> np.ndarray:
    """Return clipped sRGB values for a reflectance spectrum under an illuminant."""
    if reflectance.domain is not SpectrumDomain.REFLECTANCE:
        raise ValueError("perceived_colour_rgb expects a reflectance spectrum")

    aligned_reflectance = reflectance.sd.copy().align(PROJECT_SHAPE)
    aligned_illuminant = illuminant.copy().align(PROJECT_SHAPE)
    aligned_cmfs = cmfs.copy().align(PROJECT_SHAPE)

    product = colour.SpectralDistribution(
        {
            float(wavelength): float(reflectance_value * illuminant_value)
            for wavelength, reflectance_value, illuminant_value in zip(
                aligned_reflectance.wavelengths,
                aligned_reflectance.values,
                aligned_illuminant.values,
                strict=True,
            )
        }
    )
    xyz = colour.sd_to_XYZ(product, cmfs=aligned_cmfs)
    rgb = colour.XYZ_to_sRGB(xyz / 100.0)
    return np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)


def plot_perceived_colour(
    reflectance: Spectrum,
    ax: Axes | None = None,
    *,
    illuminant: colour.SpectralDistribution = DEFAULT_ILLUMINANT,
    cmfs: colour.MultiSpectralDistributions = DEFAULT_CMFS,
    title: str | None = None,
) -> Axes:
    """Plot a perceived-colour swatch for a reflectance spectrum."""
    if ax is None:
        _, ax = plt.subplots()

    rgb = perceived_colour_rgb(reflectance, illuminant=illuminant, cmfs=cmfs)
    ax.add_patch(
        Rectangle((0, 0), 1, 1, facecolor=rgb, edgecolor="black", linewidth=1.0)
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_title(title or reflectance.provenance.source)
    return ax
