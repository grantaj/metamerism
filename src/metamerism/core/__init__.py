"""Spectral utilities and core types.

This module provides foundational types and utilities used by higher-level
modules. This package defers numerical colour and spectral computation to
colour-science; metamerism is responsible for provenance, ingestion and
project-specific workflows.

Exposed concepts:

* :class:`Spectrum` - provenance-aware wrapper around spectral data
* :class:`SpectrumDomain` - physical meaning of spectral values
* :class:`Provenance` - data origin and processing history
* :class:`Observer` - convenience aliases for standard colour-science CMFs
* Exception types: :exc:`SpectrumError`, :exc:`GridMismatchError`,
  :exc:`ExtrapolationError`, :exc:`DomainError`

Canonical wavelength grid:
* :data:`CANONICAL_GRID` - 380-780 nm, 5 nm spacing, 81 points
* :data:`PROJECT_SHAPE` - convenience alias to a colour.SpectralShape
"""

from metamerism.core.exceptions import (
    DomainError,
    ExtrapolationError,
    GridMismatchError,
    SpectrumError,
)
from metamerism.core.observer import (
    CIE_1931_2_DEGREE,
    CIE_2006_2_DEGREE,
    CIE_2006_10_DEGREE,
    DEFAULT_OBSERVER,
    Observer,
)
from metamerism.core.spectrum import (
    CANONICAL_GRID,
    PROJECT_SHAPE,
    Provenance,
    Spectrum,
    SpectrumDomain,
)

__all__ = [
    "CANONICAL_GRID",
    "CIE_1931_2_DEGREE",
    "CIE_2006_2_DEGREE",
    "CIE_2006_10_DEGREE",
    "DEFAULT_OBSERVER",
    "DomainError",
    "ExtrapolationError",
    "GridMismatchError",
    "Observer",
    "PROJECT_SHAPE",
    "Provenance",
    "Spectrum",
    "SpectrumDomain",
    "SpectrumError",
]
