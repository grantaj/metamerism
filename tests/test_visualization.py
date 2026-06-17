"""Tests for visualisation helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from metamerism.pigments import load_golden_heavy_body_reflectance_by_name
from metamerism.visualization import (
    perceived_colour_rgb,
    plot_perceived_colour,
    plot_spectrum,
)

plt.switch_backend("Agg")


def test_plot_spectrum_draws_one_line():
    paint = load_golden_heavy_body_reflectance_by_name("Alizarin Crimson Hue")
    fig, ax = plt.subplots()
    try:
        result = plot_spectrum(paint.spectrum, ax=ax)
        fig.canvas.draw()

        assert result is ax
        assert len(ax.lines) == 1
        assert ax.lines[0].get_xdata().shape[0] == paint.spectrum.n_samples
    finally:
        plt.close(fig)


def test_perceived_colour_rgb_returns_clipped_rgb():
    paint = load_golden_heavy_body_reflectance_by_name("Alizarin Crimson Hue")
    rgb = perceived_colour_rgb(paint.spectrum)

    assert rgb.shape == (3,)
    assert np.all(rgb >= 0.0)
    assert np.all(rgb <= 1.0)


def test_plot_perceived_colour_draws_patch():
    paint = load_golden_heavy_body_reflectance_by_name("Alizarin Crimson Hue")
    fig, ax = plt.subplots()
    try:
        result = plot_perceived_colour(paint.spectrum, ax=ax)
        fig.canvas.draw()

        assert result is ax
        assert len(ax.patches) == 1
        assert result.get_title() == paint.spectrum.provenance.source
    finally:
        plt.close(fig)
