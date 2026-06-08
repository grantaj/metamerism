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
    from scipy.interpolate import interp1d

    from metamerism.core.spectrum import CANONICAL_GRID

    # Access the colour library's CMF database
    cmfs_db = colour.colorimetry.MSDS_CMFS_STANDARD_OBSERVER

    # Select the appropriate observers
    # CIE 1931 2-degree
    cmf_1931 = cmfs_db["CIE 1931 2 Degree Standard Observer"]

    # CIE 2015 2-degree (colour library uses 2015 instead of 2006 for 2-degree)
    cmf_2006_2 = cmfs_db["CIE 2015 2 Degree Standard Observer"]

    # CIE 2015 10-degree (colour library uses 2015 instead of 2006 for 10-degree)
    cmf_2006_10 = cmfs_db["CIE 2015 10 Degree Standard Observer"]

    def resample_to_canonical(cmf_obj) -> NDArray[np.float64]:
        """Resample CMF data to the canonical wavelength grid.

        Parameters
        ----------
        cmf_obj : colour.colorimetry.cmfs.XYZ_ColourMatchingFunctions
            CMF data from the colour library.

        Returns
        -------
        resampled : NDArray[np.float64]
            CMF values on CANONICAL_GRID, shape (81, 3).
        """
        wavelengths = np.asarray(cmf_obj.wavelengths)
        values = np.asarray(cmf_obj.values)  # shape (n, 3) for X, Y, Z

        # Interpolate each of the three CMF components to the canonical grid
        resampled = np.zeros((len(CANONICAL_GRID), 3))
        for i in range(3):
            # Linear interpolation (appropriate for CMFs)
            f = interp1d(
                wavelengths,
                values[:, i],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,  # Zero outside the measured range
            )
            resampled[:, i] = f(CANONICAL_GRID)

        return resampled

    # Resample all three observers to the canonical grid
    cmf_1931_vals = resample_to_canonical(cmf_1931)
    cmf_2006_2_vals = resample_to_canonical(cmf_2006_2)
    cmf_2006_10_vals = resample_to_canonical(cmf_2006_10)

    # Trapezoidal integration weights for 5 nm spacing (from 380 to 780 nm).
    weights = np.full(81, 5.0)
    weights[0] = 2.5
    weights[-1] = 2.5

    # Create Observer instances
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
