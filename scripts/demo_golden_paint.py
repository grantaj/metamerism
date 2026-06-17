"""Demo loader and visualiser for GOLDEN Heavy Body spectra."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from metamerism.pigments import (
    load_golden_heavy_body_reflectance,
    load_golden_heavy_body_reflectance_by_name,
)
from metamerism.visualization import plot_perceived_colour, plot_spectrum


def build_figure(paint) -> Figure:
    """Build a two-panel preview figure for a GOLDEN paint."""
    fig, (ax_spectrum, ax_colour) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    plot_spectrum(
        paint.spectrum,
        ax=ax_spectrum,
        label=f"{paint.paint_name} ({paint.product_id})",
        color="black",
    )
    ax_spectrum.legend(loc="upper right")
    ax_spectrum.set_title("Reflectance spectrum")

    plot_perceived_colour(
        paint.spectrum,
        ax=ax_colour,
        title="Perceived colour under D65",
    )

    fig.suptitle(paint.paint_name)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview a GOLDEN Heavy Body paint spectrum and its D65 colour."
    )
    parser.add_argument(
        "--paint",
        default=None,
        help="Exact GOLDEN paint name. Defaults to the first loaded paint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the figure is shown.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.paint is None:
        paint = load_golden_heavy_body_reflectance()[0]
    else:
        paint = load_golden_heavy_body_reflectance_by_name(args.paint)

    fig = build_figure(paint)
    if args.output is not None:
        fig.savefig(args.output, dpi=160)
    else:
        plt.show()


if __name__ == "__main__":
    main()
