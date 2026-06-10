"""
metamerism.core.observer
~~~~~~~~~~~~~~~~~~~~~~~~

Standard colour matching functions and observer definitions.

An :class:`Observer` encapsulates the colour matching functions (CMF) for a
particular standard observer. It is always passed explicitly as a parameter
to colorimetric calculations; there are no implicit global defaults.

The observer is a genuine scientific decision with artistic implications. For
example, the 10° observer is more physiologically accurate for paintings
viewed from typical gallery distances, whereas the 2° observer matches
most published ΔE formulae and is appropriate for small colour patches.

Standard observers provided:

* :data:`CIE_1931_2_DEGREE` — CIE 1931 2° standard; default
* :data:`CIE_2006_2_DEGREE` — CIE 2006 2°; more physiologically accurate
* :data:`CIE_2006_10_DEGREE` — CIE 2006 10°; for distant viewing

Reference:
    CIE 2004. Colorimetry (3rd ed.). CIE Publication 15.
    https://cie.co.at/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass

# Defer import of colour library to avoid hard dependency during type checking
# when TYPE_CHECKING is False


# ---------------------------------------------------------------------------
# Observer dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observer:
    """Immutable record of a standard colour observer and its CMF.

    The observer is an explicit parameter in all colorimetric calculations.
    It is never a hidden global; users always make an intentional choice of
    which observer model to use.

    Parameters
    ----------
    name:
        Human-readable identifier for the observer.
        Examples: ``"CIE_1931_2_degree"``, ``"CIE_2006_10_degree"``.
    cmf:
        Colour matching functions on the canonical wavelength grid (380-780 nm,
        5 nm spacing, 81 points). Shape must be (81, 3) with columns
        corresponding to X, Y, Z tristimulus components.
    integration_weights:
        Numerical integration weights corresponding to each wavelength.
        For uniform 5 nm spacing, this uses trapezoidal endpoint weights
        (2.5 nm at the ends, 5 nm internally).
        Shape must be (81,).

    Notes
    -----
    All spectra and CMFs are assumed to be on the canonical wavelength grid
    defined by :data:`~metamerism.core.spectrum.CANONICAL_GRID`. Do not
    construct Observer instances by hand; use the module-level constants
    instead or call :func:`from_colour_library`.
    """

    name: str
    """Name of this observer, e.g. ``"CIE_1931_2_degree"``."""

    cmf: NDArray[np.float64]
    """Colour matching functions: shape (81, 3).

    Columns are X, Y, Z tristimulus components on the canonical grid.
    """

    integration_weights: NDArray[np.float64]
    """Numerical integration weights: shape (81,).

    For the canonical 5 nm grid, this typically uses trapezoidal weights.
    """

    def __post_init__(self) -> None:
        """Validate shapes and values after construction."""
        if self.cmf.shape != (81, 3):
            raise ValueError(f"cmf must have shape (81, 3), got {self.cmf.shape}")
        if self.integration_weights.shape != (81,):
            raise ValueError(
                "integration_weights must have shape (81,), "
                f"got {self.integration_weights.shape}"
            )
        if not np.all(self.integration_weights > 0):
            raise ValueError("integration_weights must be strictly positive")


# ---------------------------------------------------------------------------
# Standard observers from the colour library
# ---------------------------------------------------------------------------


def _load_observers() -> tuple[Observer, Observer, Observer]:
    """Load standard CIE observers from the colour library.

    Returns
    -------
    cie_1931_2, cie_2006_2, cie_2006_10
        Three Observer instances for the CIE 1931 2°, CIE 2006 2°,
        and CIE 2006 10° observers respectively.

    Notes
    -----
    The colour library provides tabulated CMF data. This data is extracted,
    resampled to the canonical wavelength grid, and wrapped in Observer instances.
    """
    import colour
    import numpy as np

    from metamerism.core.spectrum import CANONICAL_GRID, PROJECT_SHAPE

    # Access the colour library's CMF database
    cmfs_db = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER

    # Select the appropriate observers from the colour library and align to
    # the project-preferred shape using colour's own alignment routines. This
    # avoids reimplementing interpolation behaviour.
    cmf_1931 = cmfs_db["CIE 1931 2 Degree Standard Observer"].copy().align(PROJECT_SHAPE)
    cmf_2006_2 = cmfs_db["CIE 2015 2 Degree Standard Observer"].copy().align(PROJECT_SHAPE)
    cmf_2006_10 = cmfs_db["CIE 2015 10 Degree Standard Observer"].copy().align(PROJECT_SHAPE)

    # Extract values (shape (81, 3)) directly from the aligned CMF objects
    cmf_1931_vals = np.asarray(cmf_1931.values)
    cmf_2006_2_vals = np.asarray(cmf_2006_2.values)
    cmf_2006_10_vals = np.asarray(cmf_2006_10.values)

    # Trapezoidal integration weights for 5 nm spacing (from 380 to 780 nm).
    # Keeping explicit weights preserves behaviour in xyz() which multiplies
    # CMFs by integration_weights; these weights are independent of the
    # interpolation method used above.
    weights = np.full(len(CANONICAL_GRID), float(PROJECT_SHAPE.interval))
    weights[0] = float(PROJECT_SHAPE.interval) / 2.0
    weights[-1] = float(PROJECT_SHAPE.interval) / 2.0

    # Create Observer instances as thin wrappers over the numeric arrays.
    obs_1931 = Observer(
        name="CIE 1931 2 Degree",
        cmf=cmf_1931_vals,
        integration_weights=weights,
    )

    obs_2006_2 = Observer(
        name="CIE 2015 2 Degree",
        cmf=cmf_2006_2_vals,
        integration_weights=weights,
    )

    obs_2006_10 = Observer(
        name="CIE 2015 10 Degree",
        cmf=cmf_2006_10_vals,
        integration_weights=weights,
    )

    return obs_1931, obs_2006_2, obs_2006_10


# Load standard observers at module import time
_obs_1931, _obs_2006_2, _obs_2006_10 = _load_observers()


#: CIE 1931 2° standard observer (default).
#:
#: Matches most published colour difference formulae (e.g., ΔE*₇₆).
#: Appropriate for small colour patches or when compatibility with
#: published data is important.
CIE_1931_2_DEGREE = _obs_1931

#: CIE 2006 2° standard observer.
#:
#: More physiologically accurate than the 1931 observer. Preferred when
#: human visual response is the primary concern.
CIE_2006_2_DEGREE = _obs_2006_2

#: CIE 2006 10° standard observer.
#:
#: Appropriate for colours viewed at larger visual angles, such as paintings
#: viewed from typical gallery distances (typically 1-2 m away from a 1x1 m
#: work).
CIE_2006_10_DEGREE = _obs_2006_10

# Default observer for convenience
#: Default observer used when not explicitly specified. Set to
#: :data:`CIE_1931_2_DEGREE` for compatibility with published work;
#: change to :data:`CIE_2006_10_DEGREE` for gallery/installation contexts.
DEFAULT_OBSERVER = CIE_1931_2_DEGREE
