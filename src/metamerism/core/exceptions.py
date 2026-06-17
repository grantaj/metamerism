"""
metamerism.core.exceptions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Package-wide exception types for spectral and colorimetric operations.

All exceptions raised by the metamerism package are defined here and imported
from this module. This provides a single, authoritative list of custom exceptions.
"""


class SpectrumError(Exception):
    """Base class for all spectrum-related errors.

    Raised when an operation on a :class:`Spectrum` object violates a constraint
    or precondition.
    """


class GridMismatchError(SpectrumError):
    """Raised when an operation requires spectra on matching grids but they differ.

    For example, project-specific spectrum comparison metrics require matching
    wavelength samples. Use colour-science alignment explicitly before calling
    those metrics when inputs are sampled differently.
    """


class ExtrapolationError(SpectrumError):
    """Raised when extrapolation is required but not permitted by caller policy."""


class DomainError(SpectrumError):
    """Raised when a domain-specific constraint is violated.

    Each :class:`SpectrumDomain` has valid value ranges:

    * ``REFLECTANCE`` - values in [0, 1]
    * ``TRANSMITTANCE`` - values in [0, 1]
    * ``ILLUMINANT`` - values >= 0
    * ``EMISSION`` - values >= 0

    This exception is raised when constructing or modifying a spectrum if the
    values violate the range for the declared domain. Pass ``validate=False``
    to :meth:`Spectrum.from_arrays` to skip validation (appropriate for
    intermediate results that may temporarily violate constraints).
    """
