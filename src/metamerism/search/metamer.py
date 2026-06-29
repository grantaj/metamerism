"""Metameric candidate scoring: public domain types and scores."""

from __future__ import annotations

from dataclasses import dataclass

import colour
import numpy as np
from numpy.typing import NDArray

from metamerism.core import PROJECT_SHAPE, Spectrum, SpectrumDomain

DEFAULT_CONCEAL_DELTA_E00_LIMIT = 1.0
# One just-noticeable difference. Pairs within this threshold are imperceptible
# under the conceal illuminant. Tighten to 0.5 for stricter searches.


@dataclass(frozen=True)
class ViewingCondition:
    """A named illuminant/observer pair used for metameric scoring."""

    name: str
    illuminant: Spectrum
    observer_name: str = "CIE 1931 2 Degree Standard Observer"

    def __post_init__(self) -> None:
        if self.illuminant.domain is not SpectrumDomain.ILLUMINANT:
            raise ValueError(
                f"illuminant must have domain ILLUMINANT, got {self.illuminant.domain}"
            )


@dataclass(frozen=True)
class ConditionResult:
    """Per-condition colorimetric comparison of two spectra."""

    condition_name: str
    xyz_first: NDArray[np.float64]
    xyz_second: NDArray[np.float64]
    lab_first: NDArray[np.float64]
    lab_second: NDArray[np.float64]
    delta_e00: float

    def to_dict(self) -> dict:
        return {
            "condition_name": self.condition_name,
            "xyz_first": self.xyz_first.tolist(),
            "xyz_second": self.xyz_second.tolist(),
            "lab_first": self.lab_first.tolist(),
            "lab_second": self.lab_second.tolist(),
            "delta_e00": self.delta_e00,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConditionResult:
        return cls(
            condition_name=d["condition_name"],
            xyz_first=np.array(d["xyz_first"]),
            xyz_second=np.array(d["xyz_second"]),
            lab_first=np.array(d["lab_first"]),
            lab_second=np.array(d["lab_second"]),
            delta_e00=d["delta_e00"],
        )


@dataclass(frozen=True)
class MetamerScore:
    """Summary score for a conceal/reveal candidate pair."""

    conceal: ConditionResult
    reveal: tuple[ConditionResult, ...]
    minimum_reveal_delta_e00: float

    def to_dict(self) -> dict:
        return {
            "conceal": self.conceal.to_dict(),
            "reveal": [r.to_dict() for r in self.reveal],
            "minimum_reveal_delta_e00": self.minimum_reveal_delta_e00,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetamerScore:
        return cls(
            conceal=ConditionResult.from_dict(d["conceal"]),
            reveal=tuple(ConditionResult.from_dict(r) for r in d["reveal"]),
            minimum_reveal_delta_e00=d["minimum_reveal_delta_e00"],
        )


def _align_to_project(spectrum: Spectrum) -> colour.SpectralDistribution:
    """Return the spectrum's SD aligned to the project spectral shape."""
    sd = spectrum.sd.copy()
    if sd.shape != PROJECT_SHAPE:
        sd = sd.align(PROJECT_SHAPE)
    return sd


def score_condition(
    first: Spectrum,
    second: Spectrum,
    condition: ViewingCondition,
) -> ConditionResult:
    """Score two spectra under one viewing condition via direct colour calls.

    Both spectra are aligned to the project grid before conversion.
    XYZ is computed with Y=100 normalisation (perfect white reflector = 100).
    """
    illuminant_sd = condition.illuminant.sd.copy()
    if illuminant_sd.shape != PROJECT_SHAPE:
        illuminant_sd = illuminant_sd.align(PROJECT_SHAPE)

    cmfs = colour.colorimetry.MSDS_CMFS[condition.observer_name]

    sd_first = _align_to_project(first)
    sd_second = _align_to_project(second)

    xyz_first = colour.sd_to_XYZ(
        sd_first, cmfs=cmfs, illuminant=illuminant_sd, shape=PROJECT_SHAPE
    )
    xyz_second = colour.sd_to_XYZ(
        sd_second, cmfs=cmfs, illuminant=illuminant_sd, shape=PROJECT_SHAPE
    )

    # XYZ white point for the illuminant (needed for Lab conversion)
    white_sd = colour.SpectralDistribution(
        {w: 1.0 for w in illuminant_sd.wavelengths},
        name="perfect_white",
    )
    xyz_white = colour.sd_to_XYZ(
        white_sd, cmfs=cmfs, illuminant=illuminant_sd, shape=PROJECT_SHAPE
    )

    lab_first = colour.XYZ_to_Lab(xyz_first / 100.0, illuminant=xyz_white / 100.0)
    lab_second = colour.XYZ_to_Lab(xyz_second / 100.0, illuminant=xyz_white / 100.0)

    delta_e = float(colour.delta_E(lab_first, lab_second, method="CIE 2000"))

    return ConditionResult(
        condition_name=condition.name,
        xyz_first=np.array(xyz_first, dtype=np.float64),
        xyz_second=np.array(xyz_second, dtype=np.float64),
        lab_first=np.array(lab_first, dtype=np.float64),
        lab_second=np.array(lab_second, dtype=np.float64),
        delta_e00=delta_e,
    )


def score_metameric_pair(
    first: Spectrum,
    second: Spectrum,
    conceal_condition: ViewingCondition,
    reveal_conditions: tuple[ViewingCondition, ...] | list[ViewingCondition],
) -> MetamerScore:
    """Score a spectrum pair under one conceal and one or more reveal conditions.

    Both spectra must have domain=REFLECTANCE. The conceal condition defines
    the illuminant under which the pair should appear identical; reveal
    conditions define illuminants that expose the difference.

    Args:
        first: First reflectance spectrum.
        second: Second reflectance spectrum.
        conceal_condition: Condition under which the pair should match.
        reveal_conditions: One or more conditions under which they should differ.

    Returns:
        MetamerScore with per-condition results and the minimum reveal ΔE00.
    """
    for sp, label in ((first, "first"), (second, "second")):
        if sp.domain is not SpectrumDomain.REFLECTANCE:
            raise ValueError(
                f"{label} spectrum must have domain REFLECTANCE, got {sp.domain}"
            )

    if not reveal_conditions:
        raise ValueError("At least one reveal condition is required")

    conceal_result = score_condition(first, second, conceal_condition)

    reveal_results = tuple(
        score_condition(first, second, cond) for cond in reveal_conditions
    )

    min_reveal = min(r.delta_e00 for r in reveal_results)

    return MetamerScore(
        conceal=conceal_result,
        reveal=reveal_results,
        minimum_reveal_delta_e00=min_reveal,
    )
