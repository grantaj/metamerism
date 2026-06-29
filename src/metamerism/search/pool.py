"""Candidate pool selection: Pareto filtering and diversity for metamer search.

Phase E: given a pool of scored metamer candidates, identify the Pareto-optimal
subset and apply greedy farthest-point diversity selection to return a useful
spread of high-value, non-redundant candidates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metamerism.core import Provenance, Spectrum, SpectrumDomain
from metamerism.search.diagnostics import PoolDiagnostics, summarise_pool
from metamerism.search.generate import (
    CandidateGenerationConfig,
    MetamerCandidate,
    generate_directional_metamers,
)
from metamerism.search.metamer import ViewingCondition
from metamerism.search.objectives import (
    CandidateScore,
    RankingConfig,
    rank_scored_candidates,
    score_candidate,
)

# ---------------------------------------------------------------------------
# Pareto utilities
# ---------------------------------------------------------------------------

_ScoredPair = tuple[MetamerCandidate, CandidateScore]


def _dominates(a: CandidateScore, b: CandidateScore) -> bool:
    """Return True if a weakly dominates b on all four objectives.

    Objectives (with direction):
      - weighted_reveal_score: maximise
      - minimum_reveal_delta_e00: maximise
      - roughness: minimise
      - boundary_saturation: minimise
    """
    at_least_as_good = (
        a.weighted_reveal_score >= b.weighted_reveal_score
        and a.minimum_reveal_delta_e00 >= b.minimum_reveal_delta_e00
        and a.roughness <= b.roughness
        and a.boundary_saturation <= b.boundary_saturation
    )
    strictly_better = (
        a.weighted_reveal_score > b.weighted_reveal_score
        or a.minimum_reveal_delta_e00 > b.minimum_reveal_delta_e00
        or a.roughness < b.roughness
        or a.boundary_saturation < b.boundary_saturation
    )
    return at_least_as_good and strictly_better


def pareto_front(
    scored: list[_ScoredPair],
) -> list[_ScoredPair]:
    """Return the Pareto-optimal subset of scored candidates.

    A candidate is non-dominated if no other candidate is at least as good
    on all four objectives (weighted_reveal_score, minimum_reveal_delta_e00,
    roughness, boundary_saturation) and strictly better on at least one.

    Input order is preserved among non-dominated candidates.

    Args:
        scored: List of (MetamerCandidate, CandidateScore) pairs.

    Returns:
        Non-dominated subset of scored, in the same relative order.
    """
    if not scored:
        return []

    n = len(scored)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if _dominates(scored[i][1], scored[j][1]):
                dominated[j] = True

    return [pair for pair, dom in zip(scored, dominated, strict=True) if not dom]


# ---------------------------------------------------------------------------
# Diversity selection
# ---------------------------------------------------------------------------


def diversity_select(
    scored: list[_ScoredPair],
    n: int,
) -> list[_ScoredPair]:
    """Select up to n diverse candidates by greedy farthest-point in spectral space.

    The first candidate (index 0) is always selected first. Each subsequent
    pick is the remaining candidate that maximises the minimum L2 distance to
    all already-selected spectra. This spreads the selection across the
    feasible spectral region.

    Args:
        scored: Candidates to select from, in preference order (best first).
            The first element gets priority over diversity for the initial pick.
        n: Maximum number of candidates to select.

    Returns:
        Up to n candidates. Input preference wins ties and determines the
        first selection; diversity governs subsequent picks.
    """
    if not scored or n <= 0:
        return []

    if len(scored) <= n:
        return list(scored)

    spectra: NDArray[np.float64] = np.stack(
        [c.spectrum.values for c, _ in scored]
    )  # (m, n_wavelengths)

    selected_indices = [0]
    remaining = list(range(1, len(scored)))

    while len(selected_indices) < n and remaining:
        selected_spectra = spectra[selected_indices]  # (k, n_wavelengths)
        # Candidate with maximum min-distance to all already-selected spectra
        best = max(
            remaining,
            key=lambda i: float(
                np.min(np.linalg.norm(spectra[i] - selected_spectra, axis=1))
            ),
        )
        selected_indices.append(best)
        remaining.remove(best)

    return [scored[i] for i in selected_indices]


# ---------------------------------------------------------------------------
# SearchConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchConfig:
    """Full configuration for one end-to-end metamer search run.

    Attributes:
        conceal_condition: Illuminant/observer pair under which the pair
            should appear identical.
        reveal_conditions: One or more conditions under which the pair
            should differ.
        generation: Candidate generation parameters (seed, n_directions, …).
        ranking: Ranking and filtering policy.
        reveal_weights: Per-condition weights for the weighted_reveal_score.
            None means uniform weights. Must be positive if supplied.
        n_diverse: Maximum number of diverse candidates to return from the
            Pareto front.
    """

    conceal_condition: ViewingCondition
    reveal_conditions: tuple[ViewingCondition, ...]
    generation: CandidateGenerationConfig
    ranking: RankingConfig
    reveal_weights: tuple[float, ...] | None = None
    n_diverse: int = 10

    def __post_init__(self) -> None:
        if not self.reveal_conditions:
            raise ValueError("At least one reveal condition is required")
        if self.reveal_weights is not None:
            if len(self.reveal_weights) != len(self.reveal_conditions):
                raise ValueError(
                    f"reveal_weights length {len(self.reveal_weights)} "
                    f"!= reveal_conditions length {len(self.reveal_conditions)}"
                )
            if any(w <= 0 for w in self.reveal_weights):
                raise ValueError("All reveal_weights must be positive")
        if self.n_diverse < 1:
            raise ValueError(f"n_diverse must be >= 1, got {self.n_diverse}")

    def to_dict(self) -> dict:
        def _vc_to_dict(vc: ViewingCondition) -> dict:
            return {
                "name": vc.name,
                "illuminant_wavelengths": vc.illuminant.wavelengths.tolist(),
                "illuminant_values": vc.illuminant.values.tolist(),
                "observer_name": vc.observer_name,
            }

        return {
            "conceal_condition": _vc_to_dict(self.conceal_condition),
            "reveal_conditions": [
                _vc_to_dict(vc) for vc in self.reveal_conditions
            ],
            "reveal_weights": (
                list(self.reveal_weights)
                if self.reveal_weights is not None
                else None
            ),
            "generation": self.generation.to_dict(),
            "ranking": self.ranking.to_dict(),
            "n_diverse": self.n_diverse,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SearchConfig:
        def _vc_from_dict(vc_d: dict) -> ViewingCondition:
            wavelengths = np.array(
                vc_d["illuminant_wavelengths"], dtype=np.float64
            )
            values = np.array(vc_d["illuminant_values"], dtype=np.float64)
            illuminant = Spectrum.from_arrays(
                wavelengths,
                values,
                domain=SpectrumDomain.ILLUMINANT,
                provenance=Provenance(
                    source=f"illuminant/{vc_d['name']}", notes="restored"
                ),
                validate=False,
            )
            return ViewingCondition(
                name=vc_d["name"],
                illuminant=illuminant,
                observer_name=vc_d["observer_name"],
            )

        rw = d.get("reveal_weights")
        return cls(
            conceal_condition=_vc_from_dict(d["conceal_condition"]),
            reveal_conditions=tuple(
                _vc_from_dict(vc_d) for vc_d in d["reveal_conditions"]
            ),
            reveal_weights=tuple(rw) if rw is not None else None,
            generation=CandidateGenerationConfig.from_dict(d["generation"]),
            ranking=RankingConfig.from_dict(d["ranking"]),
            n_diverse=d.get("n_diverse", 10),
        )


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """Output of a complete metamer search run.

    Attributes:
        reference: The reference reflectance spectrum used as the search origin.
        config: The search configuration that produced this result.
        all_scored: All generated candidates with scores (pre-ranking).
        pareto_scored: Pareto-optimal subset of the ranked candidates.
        selected: Diversity-selected candidates from the Pareto front.
        pool_diagnostics: Summary statistics over the full scored pool.
    """

    reference: Spectrum
    config: SearchConfig
    all_scored: list[_ScoredPair]
    pareto_scored: list[_ScoredPair]
    selected: list[_ScoredPair]
    pool_diagnostics: PoolDiagnostics

    def to_dict(self) -> dict:
        def _pair(c: MetamerCandidate, s: CandidateScore) -> dict:
            return {"candidate": c.to_dict(), "score": s.to_dict()}

        return {
            "reference_wavelengths": self.reference.wavelengths.tolist(),
            "reference_values": self.reference.values.tolist(),
            "reference_provenance_source": self.reference.provenance.source,
            "config": self.config.to_dict(),
            "all_scored": [_pair(c, s) for c, s in self.all_scored],
            "pareto_scored": [_pair(c, s) for c, s in self.pareto_scored],
            "selected": [_pair(c, s) for c, s in self.selected],
            "pool_diagnostics": self.pool_diagnostics.to_dict(),
        }


# ---------------------------------------------------------------------------
# End-to-end search
# ---------------------------------------------------------------------------


def run_search(
    reference: Spectrum,
    config: SearchConfig,
) -> SearchResult:
    """Run a complete end-to-end metamer search.

    Steps:
    1. Generate bounded null-space partner candidates (Phase C).
    2. Score every candidate under all reveal conditions (Phase D).
    3. Rank and filter by conceal threshold and optional roughness limit.
    4. Compute the Pareto front over (weighted_reveal_score,
       minimum_reveal_delta_e00, roughness, boundary_saturation).
    5. Apply greedy farthest-point diversity selection from the Pareto front.
    6. Summarise the full pool for diagnostics.

    Args:
        reference: Reference reflectance spectrum (domain=REFLECTANCE).
        config: Full search configuration.

    Returns:
        SearchResult with all scored candidates, Pareto front, and
        diversity-selected subset.
    """
    candidates = generate_directional_metamers(
        reference, config.conceal_condition, config.generation
    )

    reveal_weights = (
        list(config.reveal_weights) if config.reveal_weights is not None else None
    )
    all_scored: list[_ScoredPair] = [
        (
            c,
            score_candidate(
                c,
                list(config.reveal_conditions),
                reveal_weights=reveal_weights,
            ),
        )
        for c in candidates
    ]

    ranked = rank_scored_candidates(all_scored, config.ranking)
    pareto = pareto_front(ranked)
    selected = diversity_select(pareto, config.n_diverse)
    diag = summarise_pool(all_scored, conceal_limit=config.ranking.conceal_limit)

    return SearchResult(
        reference=reference,
        config=config,
        all_scored=all_scored,
        pareto_scored=pareto,
        selected=selected,
        pool_diagnostics=diag,
    )
