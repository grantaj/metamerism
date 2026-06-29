"""Metameric search and scoring."""

from metamerism.search.diagnostics import (
    PoolDiagnostics,
    ProjectionDiagnostics,
    candidate_report,
    check_projection,
    summarise_pool,
)
from metamerism.search.feasibility import (
    DEFAULT_REFLECTANCE_BOUND_TOLERANCE,
    feasible_step_interval,
    is_feasible,
    max_bound_violation,
)
from metamerism.search.generate import (
    DEFAULT_ENDPOINT_FRACTION,
    DEFAULT_N_SMOOTH_DIRECTIONS,
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
    DEFAULT_ROUGHNESS_LIMIT,
    CandidateScore,
    RankingConfig,
    boundary_saturation,
    passes_conceal_threshold,
    passes_roughness_limit,
    rank_candidates,
    rank_scored_candidates,
    score_candidate,
)
from metamerism.search.pool import (
    SearchConfig,
    SearchResult,
    diversity_select,
    pareto_front,
    run_search,
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
from metamerism.search.realisability import (
    DEFAULT_CONCEAL_DELTA_E00_LIMIT as DEFAULT_REALISABILITY_CONCEAL_LIMIT,
)
from metamerism.search.realisability import (
    MODEL_NAME_LINEAR,
    PaintRealisation,
    RealisabilityConfig,
    find_metameric_mixture,
)

__all__ = [
    "DEFAULT_CONCEAL_DELTA_E00_LIMIT",
    "DEFAULT_ENDPOINT_FRACTION",
    "DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE",
    "DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE",
    "DEFAULT_N_SMOOTH_DIRECTIONS",
    "DEFAULT_REALISABILITY_CONCEAL_LIMIT",
    "DEFAULT_REFLECTANCE_BOUND_TOLERANCE",
    "DEFAULT_ROUGHNESS_LIMIT",
    "MODEL_NAME_LINEAR",
    "CandidateGenerationConfig",
    "CandidateScore",
    "ConditionResult",
    "MetamerCandidate",
    "MetamerScore",
    "NullSpace",
    "PaintRealisation",
    "PoolDiagnostics",
    "ProjectionDiagnostics",
    "RankingConfig",
    "RealisabilityConfig",
    "SearchConfig",
    "SearchResult",
    "ViewingCondition",
    "XYZProjection",
    "boundary_saturation",
    "build_xyz_projection",
    "candidate_report",
    "check_projection",
    "compute_null_space",
    "diversity_select",
    "feasible_step_interval",
    "find_metameric_mixture",
    "generate_directional_metamers",
    "is_feasible",
    "max_bound_violation",
    "pareto_front",
    "passes_conceal_threshold",
    "passes_roughness_limit",
    "projection_xyz",
    "rank_candidates",
    "rank_scored_candidates",
    "run_search",
    "score_candidate",
    "score_condition",
    "score_metameric_pair",
    "spectral_roughness",
    "summarise_pool",
]
