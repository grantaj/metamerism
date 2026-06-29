"""Conceal/reveal scoring filters, metrics, and ranking policy.

Ranking policy (in priority order):
1. Conceal ΔE₀₀ must be below threshold.
2. Reveal mismatch is maximised (primary ranking criterion).
3. Roughness is penalised or screened.
4. Boundary saturation is reported and optionally penalised.
5. Spectral distance is a secondary diagnostic.

All penalties and filters are explicit config values, never hidden regularisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metamerism.colorimetry import spectral_distance
from metamerism.core import Spectrum
from metamerism.search.generate import MetamerCandidate, spectral_roughness
from metamerism.search.metamer import (
    DEFAULT_CONCEAL_DELTA_E00_LIMIT,
    MetamerScore,
    ViewingCondition,
    score_metameric_pair,
)

DEFAULT_ROUGHNESS_LIMIT: float | None = None
# No hard cutoff by default; roughness is used for ranking rather than hard
# rejection. Calibrate a cutoff by computing the 95th-percentile roughness
# of the Golden paint catalogue and use that as a ceiling if desired.

DEFAULT_BOUNDARY_SATURATION_THRESHOLD = 0.02
# Fraction of wavelength samples within this distance of 0 or 1 counts as
# "saturated". Used for diagnostics and optional penalty.

_BOUNDARY_NEAR = 0.02
# Default proximity to [0, 1] bounds for saturation counting.


# ---------------------------------------------------------------------------
# Spectral metrics
# ---------------------------------------------------------------------------


def boundary_saturation(
    values: NDArray[np.float64],
    *,
    near: float = _BOUNDARY_NEAR,
) -> float:
    """Return the fraction of samples within `near` of the [0, 1] boundary.

    A value of 0.0 means no samples are near a boundary.
    A value of 1.0 means all samples are near 0 or 1.

    Args:
        values: Reflectance values, shape (n,).
        near: Proximity threshold (default 0.02, i.e. within 2% of bounds).
    """
    v = np.asarray(values, dtype=np.float64)
    near_low = v <= near
    near_high = v >= (1.0 - near)
    return float(np.mean(near_low | near_high))


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateScore:
    """Full diagnostic score for one MetamerCandidate against a reference.

    Attributes:
        conceal_delta_e00: ΔE₀₀ under the conceal illuminant (from candidate).
        reveal_delta_e00s: Per-reveal-condition ΔE₀₀ values (same order as
            reveal_conditions passed to score_candidate).
        minimum_reveal_delta_e00: min over reveal conditions (primary ranking).
        weighted_reveal_score: Σ w_j * ΔE₀₀_j (0 if no weights supplied).
        roughness: Second-difference roughness of the candidate spectrum.
        boundary_saturation: Fraction of samples near [0, 1] bounds.
        spectral_dist: RMS spectral distance between candidate and reference.
    """

    conceal_delta_e00: float
    reveal_delta_e00s: tuple[float, ...]
    minimum_reveal_delta_e00: float
    weighted_reveal_score: float
    roughness: float
    boundary_saturation: float
    spectral_dist: float

    def to_dict(self) -> dict:
        return {
            "conceal_delta_e00": self.conceal_delta_e00,
            "reveal_delta_e00s": list(self.reveal_delta_e00s),
            "minimum_reveal_delta_e00": self.minimum_reveal_delta_e00,
            "weighted_reveal_score": self.weighted_reveal_score,
            "roughness": self.roughness,
            "boundary_saturation": self.boundary_saturation,
            "spectral_dist": self.spectral_dist,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CandidateScore:
        return cls(
            conceal_delta_e00=d["conceal_delta_e00"],
            reveal_delta_e00s=tuple(d["reveal_delta_e00s"]),
            minimum_reveal_delta_e00=d["minimum_reveal_delta_e00"],
            weighted_reveal_score=d["weighted_reveal_score"],
            roughness=d["roughness"],
            boundary_saturation=d["boundary_saturation"],
            spectral_dist=d["spectral_dist"],
        )


def score_candidate(
    candidate: MetamerCandidate,
    reveal_conditions: list[ViewingCondition],
    *,
    conceal_condition: ViewingCondition | None = None,
    reveal_weights: list[float] | None = None,
) -> CandidateScore:
    """Compute the full diagnostic score for a candidate against its reference.

    The conceal ΔE₀₀ is taken directly from the stored candidate value (which
    was already verified by direct colour during generation). The reveal scores
    are computed fresh via score_metameric_pair.

    Args:
        candidate: A MetamerCandidate from generate_directional_metamers.
        reveal_conditions: One or more reveal viewing conditions.
        conceal_condition: If provided, re-verify conceal ΔE₀₀ from scratch.
            Otherwise, the stored candidate.conceal_delta_e00 is used.
        reveal_weights: Per-condition weights for weighted_reveal_score.
            Must match len(reveal_conditions). Defaults to uniform weights.

    Returns:
        CandidateScore with all diagnostic fields populated.
    """
    if not reveal_conditions:
        raise ValueError("At least one reveal condition is required")

    weights = (
        reveal_weights if reveal_weights is not None else [1.0] * len(reveal_conditions)
    )
    if len(weights) != len(reveal_conditions):
        raise ValueError(
            f"reveal_weights length {len(weights)} != "
            f"reveal_conditions length {len(reveal_conditions)}"
        )

    if conceal_condition is not None:
        metamer_score = score_metameric_pair(
            candidate.source_reference,
            candidate.spectrum,
            conceal_condition,
            reveal_conditions,
        )
        conceal_de = metamer_score.conceal.delta_e00
        reveal_des = tuple(r.delta_e00 for r in metamer_score.reveal)
    else:
        # Use stored conceal ΔE₀₀ and compute only reveal scores
        from metamerism.search.metamer import score_condition as _score_condition

        conceal_de = candidate.conceal_delta_e00
        reveal_des = tuple(
            _score_condition(
                candidate.source_reference, candidate.spectrum, cond
            ).delta_e00
            for cond in reveal_conditions
        )

    min_reveal = min(reveal_des)
    weighted = float(sum(w * de for w, de in zip(weights, reveal_des, strict=True)))

    roughness = spectral_roughness(candidate.spectrum.values)
    sat = boundary_saturation(candidate.spectrum.values)

    ref_on_grid = candidate.source_reference
    cand_on_grid = candidate.spectrum
    dist = spectral_distance(ref_on_grid, cand_on_grid)

    return CandidateScore(
        conceal_delta_e00=conceal_de,
        reveal_delta_e00s=reveal_des,
        minimum_reveal_delta_e00=min_reveal,
        weighted_reveal_score=weighted,
        roughness=roughness,
        boundary_saturation=sat,
        spectral_dist=dist,
    )


# ---------------------------------------------------------------------------
# Filtering and ranking
# ---------------------------------------------------------------------------


def passes_conceal_threshold(
    score: MetamerScore | CandidateScore,
    limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT,
) -> bool:
    """Return True if the conceal ΔE00 is within the allowed limit."""
    if isinstance(score, MetamerScore):
        return score.conceal.delta_e00 <= limit
    return score.conceal_delta_e00 <= limit


def passes_roughness_limit(
    score: CandidateScore,
    limit: float | None = DEFAULT_ROUGHNESS_LIMIT,
) -> bool:
    """Return True if roughness is below the optional hard limit.

    If limit is None (the default), all candidates pass.
    """
    if limit is None:
        return True
    return score.roughness <= limit


@dataclass(frozen=True)
class RankingConfig:
    """Policy parameters for candidate ranking.

    Attributes:
        conceal_limit: Maximum conceal ΔE₀₀ to accept.
        roughness_limit: Optional hard roughness ceiling (None = no hard cap).
        roughness_weight: Weight applied to roughness penalty in composite score.
            Set to 0.0 to rank by reveal only.
        saturation_weight: Weight applied to boundary saturation penalty.
            Set to 0.0 to ignore saturation in ranking.
        primary: Primary ranking key. One of "minimum_reveal", "weighted_reveal".
    """

    conceal_limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT
    roughness_limit: float | None = DEFAULT_ROUGHNESS_LIMIT
    roughness_weight: float = 0.0
    saturation_weight: float = 0.0
    primary: str = "minimum_reveal"

    def __post_init__(self) -> None:
        valid = {"minimum_reveal", "weighted_reveal"}
        if self.primary not in valid:
            raise ValueError(f"primary must be one of {valid}, got {self.primary!r}")

    def to_dict(self) -> dict:
        return {
            "conceal_limit": self.conceal_limit,
            "roughness_limit": self.roughness_limit,
            "roughness_weight": self.roughness_weight,
            "saturation_weight": self.saturation_weight,
            "primary": self.primary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RankingConfig:
        return cls(
            conceal_limit=d.get("conceal_limit", DEFAULT_CONCEAL_DELTA_E00_LIMIT),
            roughness_limit=d.get("roughness_limit", DEFAULT_ROUGHNESS_LIMIT),
            roughness_weight=d.get("roughness_weight", 0.0),
            saturation_weight=d.get("saturation_weight", 0.0),
            primary=d.get("primary", "minimum_reveal"),
        )


def _composite_score(cs: CandidateScore, cfg: RankingConfig) -> float:
    """Higher is better. Reveal reward minus roughness/saturation penalties."""
    if cfg.primary == "weighted_reveal":
        reveal = cs.weighted_reveal_score
    else:
        reveal = cs.minimum_reveal_delta_e00
    penalty = cfg.roughness_weight * cs.roughness + (
        cfg.saturation_weight * cs.boundary_saturation
    )
    return reveal - penalty


def rank_scored_candidates(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
    config: RankingConfig | None = None,
) -> list[tuple[MetamerCandidate, CandidateScore]]:
    """Filter and rank (candidate, score) pairs by the ranking policy.

    Filtering steps (in order):
    1. Conceal threshold.
    2. Optional roughness hard limit.

    Remaining candidates are sorted descending by composite score.

    Args:
        scored: List of (MetamerCandidate, CandidateScore) pairs.
        config: Ranking policy. Defaults to RankingConfig() if None.

    Returns:
        Filtered and sorted list, best first.
    """
    cfg = config if config is not None else RankingConfig()

    passing = [
        (c, s)
        for c, s in scored
        if passes_conceal_threshold(s, limit=cfg.conceal_limit)
        and passes_roughness_limit(s, limit=cfg.roughness_limit)
    ]
    return sorted(passing, key=lambda x: _composite_score(x[1], cfg), reverse=True)


# ---------------------------------------------------------------------------
# Legacy helpers (Phase A surface — preserved for backward compatibility)
# ---------------------------------------------------------------------------


def rank_candidates(
    candidates: list[tuple[Spectrum, MetamerScore]],
    *,
    conceal_limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT,
) -> list[tuple[Spectrum, MetamerScore]]:
    """Filter and rank (spectrum, MetamerScore) pairs by reveal strength.

    Candidates failing the conceal threshold are excluded. The remainder are
    sorted descending by minimum_reveal_delta_e00.

    Args:
        candidates: List of (spectrum, score) pairs.
        conceal_limit: Maximum allowed conceal ΔE00.

    Returns:
        Filtered and sorted list of (spectrum, score) pairs.
    """
    passing = [
        (sp, sc)
        for sp, sc in candidates
        if passes_conceal_threshold(sc, limit=conceal_limit)
    ]
    return sorted(passing, key=lambda x: x[1].minimum_reveal_delta_e00, reverse=True)
