"""Conceal/reveal scoring filters and ranking utilities."""

from __future__ import annotations

from metamerism.core import Spectrum
from metamerism.search.metamer import (
    MetamerScore,
)

DEFAULT_CONCEAL_DELTA_E00_LIMIT = 1.0
# One just-noticeable difference. Pairs within this threshold are imperceptible
# under the conceal illuminant. Tighten to 0.5 for stricter searches.


def passes_conceal_threshold(
    score: MetamerScore,
    limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT,
) -> bool:
    """Return True if the conceal ΔE00 is within the allowed limit."""
    return score.conceal.delta_e00 <= limit


def rank_candidates(
    candidates: list[tuple[Spectrum, MetamerScore]],
    *,
    conceal_limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT,
) -> list[tuple[Spectrum, MetamerScore]]:
    """Filter and rank candidates by reveal strength subject to conceal threshold.

    Candidates failing the conceal threshold are excluded. The remainder are
    sorted descending by minimum_reveal_delta_e00 (highest reveal mismatch first).

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
