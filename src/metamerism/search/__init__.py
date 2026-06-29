"""Metameric search and scoring."""

from metamerism.search.metamer import (
    ConditionResult,
    MetamerScore,
    ViewingCondition,
    score_condition,
    score_metameric_pair,
)
from metamerism.search.objectives import (
    DEFAULT_CONCEAL_DELTA_E00_LIMIT,
    passes_conceal_threshold,
    rank_candidates,
)

__all__ = [
    "DEFAULT_CONCEAL_DELTA_E00_LIMIT",
    "ConditionResult",
    "MetamerScore",
    "ViewingCondition",
    "passes_conceal_threshold",
    "rank_candidates",
    "score_condition",
    "score_metameric_pair",
]
