"""Metameric search and scoring."""

from metamerism.search.feasibility import (
    DEFAULT_REFLECTANCE_BOUND_TOLERANCE,
    feasible_step_interval,
    is_feasible,
    max_bound_violation,
)
from metamerism.search.generate import (
    DEFAULT_ENDPOINT_FRACTION,
    CandidateGenerationConfig,
    MetamerCandidate,
    generate_directional_metamers,
    spectral_roughness,
)
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
from metamerism.search.projection import (
    DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE,
    DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE,
    NullSpace,
    XYZProjection,
    build_xyz_projection,
    compute_null_space,
    projection_xyz,
)

__all__ = [
    "DEFAULT_CONCEAL_DELTA_E00_LIMIT",
    "DEFAULT_ENDPOINT_FRACTION",
    "DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE",
    "DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE",
    "DEFAULT_REFLECTANCE_BOUND_TOLERANCE",
    "CandidateGenerationConfig",
    "ConditionResult",
    "MetamerCandidate",
    "MetamerScore",
    "NullSpace",
    "ViewingCondition",
    "XYZProjection",
    "build_xyz_projection",
    "compute_null_space",
    "feasible_step_interval",
    "generate_directional_metamers",
    "is_feasible",
    "max_bound_violation",
    "passes_conceal_threshold",
    "projection_xyz",
    "rank_candidates",
    "score_condition",
    "score_metameric_pair",
    "spectral_roughness",
]
