"""Metameric candidate scoring.

This module will hold the first vertical-slice calculation:

    pair of reflectance spectra
    + conceal/reveal illuminants
    -> conceal/reveal perceptual differences
    -> candidate score
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class MetamerScore:
    """Summary score for a conceal/reveal candidate."""

    delta_e_conceal: float
    delta_e_reveal: float

    @property
    def strength_ratio(self) -> float:
        """Return a simple reveal/conceal strength ratio."""
        eps = 1e-9
        return self.delta_e_reveal / (self.delta_e_conceal + eps)
