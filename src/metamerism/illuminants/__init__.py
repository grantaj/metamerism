"""Illuminant and LED utilities."""

import colour
import numpy as np

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain


def get_illuminant(name: str) -> Spectrum:
    """Return a named CIE illuminant wrapped as a project Spectrum.

    Accepted names are any key in ``colour.SDS_ILLUMINANTS``, e.g.
    ``"D65"``, ``"D50"``, ``"A"``, ``"FL11"``.

    The returned Spectrum has ``domain=SpectrumDomain.ILLUMINANT`` and
    complete provenance. Values are aligned to the project spectral shape.
    """
    available = colour.SDS_ILLUMINANTS
    if name not in available:
        raise ValueError(
            f"Unknown illuminant {name!r}. "
            f"Available: {sorted(available.keys())[:10]}..."
        )
    sd_raw = available[name]
    sd_aligned = sd_raw.copy().align(PROJECT_SHAPE)
    values = np.asarray(sd_aligned.values, dtype=np.float64)
    wavelengths = np.asarray(sd_aligned.wavelengths, dtype=np.float64)
    provenance = Provenance(
        source=f"CIE illuminant {name}",
        notes=f"colour-science SDS_ILLUMINANTS[{name!r}], aligned to project shape",
    )
    return Spectrum.from_arrays(
        wavelengths,
        values,
        domain=SpectrumDomain.ILLUMINANT,
        provenance=provenance,
        validate=True,
    )


__all__ = ["get_illuminant"]
