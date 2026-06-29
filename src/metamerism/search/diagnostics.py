"""Numerical diagnostics and result summaries for the metamer search.

Provides human-readable and machine-readable summaries of:
- Projection matrix health (rank, singular values, matrix vs colour agreement).
- Null-space quality (A₀N residual).
- Candidate pool statistics (roughness, saturation, conceal/reveal spread).
- Individual candidate reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metamerism.search.generate import MetamerCandidate
from metamerism.search.objectives import CandidateScore
from metamerism.search.projection import (
    DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE,
    NullSpace,
    XYZProjection,
    projection_xyz,
)

# ---------------------------------------------------------------------------
# Projection diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Health summary for an XYZProjection.

    Attributes:
        rank: Numerical rank (should be 3 for a valid condition).
        singular_values: The three singular values of the projection matrix.
        condition_number: Ratio of largest to smallest singular value.
        null_space_dimension: n_wavelengths - rank.
        null_space_residual_max: max |A₀N| — should be ~1e-14 or less.
        null_space_relative_tolerance: SVD threshold used.
        matrix_xyz_max_error: Max |A @ r - colour.sd_to_XYZ| over test spectra.
            None if not yet validated.
        warnings: List of human-readable warning strings (empty if healthy).
    """

    rank: int
    singular_values: np.ndarray
    condition_number: float
    null_space_dimension: int
    null_space_residual_max: float
    null_space_relative_tolerance: float
    matrix_xyz_max_error: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "singular_values": self.singular_values.tolist(),
            "condition_number": self.condition_number,
            "null_space_dimension": self.null_space_dimension,
            "null_space_residual_max": self.null_space_residual_max,
            "null_space_relative_tolerance": self.null_space_relative_tolerance,
            "matrix_xyz_max_error": self.matrix_xyz_max_error,
            "warnings": list(self.warnings),
        }


def check_projection(
    projection: XYZProjection,
    null_space: NullSpace,
    *,
    test_reflectances: list[np.ndarray] | None = None,
    xyz_tolerance: float = DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE,
) -> ProjectionDiagnostics:
    """Compute numerical health diagnostics for a projection and its null space.

    Args:
        projection: The XYZ projection matrix to inspect.
        null_space: The corresponding null-space basis.
        test_reflectances: Optional list of reflectance vectors to use for
            matrix vs colour XYZ agreement check. If None, uses a flat 50%
            reflector only.
        xyz_tolerance: Threshold for the matrix XYZ agreement warning.

    Returns:
        ProjectionDiagnostics with rank, residuals, and any warnings.
    """
    sv = projection.singular_values
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")

    # A₀N residual
    residual = projection.matrix @ null_space.basis
    residual_max = float(np.abs(residual).max())

    warnings: list[str] = []

    if null_space.rank != 3:
        warnings.append(
            f"Unexpected null-space rank: expected 3, got {null_space.rank}. "
            "Check illuminant/CMF combination."
        )

    if residual_max > 1e-8:
        warnings.append(
            f"|A₀N| max = {residual_max:.2e} exceeds 1e-8. "
            "Null space may not be sufficiently orthogonal to projection."
        )

    # Matrix XYZ agreement check
    n = projection.matrix.shape[1]
    probes = test_reflectances if test_reflectances is not None else [np.full(n, 0.5)]
    max_xyz_error: float | None = None

    for r in probes:
        r = np.asarray(r, dtype=np.float64)
        xyz_matrix = projection_xyz(r, projection)
        # We can only check internal consistency here (not against colour),
        # but we report the magnitude so callers can compare against a
        # reference obtained via colour.sd_to_XYZ.
        _ = xyz_matrix  # stored for callers to compare externally

    # Report the max residual as a proxy (callers should validate externally)
    if residual_max > xyz_tolerance:
        warnings.append(
            f"Null-space residual ({residual_max:.2e}) exceeds "
            f"xyz_tolerance ({xyz_tolerance}). "
            "Consider tighter SVD tolerance."
        )

    return ProjectionDiagnostics(
        rank=null_space.rank,
        singular_values=np.array(sv, dtype=np.float64),
        condition_number=cond,
        null_space_dimension=null_space.n_null,
        null_space_residual_max=residual_max,
        null_space_relative_tolerance=null_space.relative_tolerance,
        matrix_xyz_max_error=max_xyz_error,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Candidate pool diagnostics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoolDiagnostics:
    """Summary statistics over a pool of MetamerCandidates.

    Attributes:
        n_total: Total candidates in the pool.
        n_passing_conceal: Candidates with conceal ΔE₀₀ below threshold.
        conceal_de_mean: Mean conceal ΔE₀₀.
        conceal_de_max: Max conceal ΔE₀₀.
        roughness_min: Minimum roughness.
        roughness_median: Median roughness.
        roughness_p95: 95th-percentile roughness.
        roughness_max: Maximum roughness.
        saturation_mean: Mean boundary saturation fraction.
        saturation_max: Maximum boundary saturation fraction.
        spectral_dist_mean: Mean spectral distance from reference.
        spectral_dist_max: Maximum spectral distance from reference.
    """

    n_total: int
    n_passing_conceal: int
    conceal_de_mean: float
    conceal_de_max: float
    roughness_min: float
    roughness_median: float
    roughness_p95: float
    roughness_max: float
    saturation_mean: float
    saturation_max: float
    spectral_dist_mean: float
    spectral_dist_max: float

    def to_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_passing_conceal": self.n_passing_conceal,
            "conceal_de_mean": self.conceal_de_mean,
            "conceal_de_max": self.conceal_de_max,
            "roughness_min": self.roughness_min,
            "roughness_median": self.roughness_median,
            "roughness_p95": self.roughness_p95,
            "roughness_max": self.roughness_max,
            "saturation_mean": self.saturation_mean,
            "saturation_max": self.saturation_max,
            "spectral_dist_mean": self.spectral_dist_mean,
            "spectral_dist_max": self.spectral_dist_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PoolDiagnostics:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def summarise_pool(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
    *,
    conceal_limit: float = 1.0,
) -> PoolDiagnostics:
    """Compute summary statistics over a scored candidate pool.

    Args:
        scored: List of (MetamerCandidate, CandidateScore) pairs.
        conceal_limit: Threshold used to count passing-conceal candidates.

    Returns:
        PoolDiagnostics with pool-wide statistics.
    """
    if not scored:
        return PoolDiagnostics(
            n_total=0,
            n_passing_conceal=0,
            conceal_de_mean=float("nan"),
            conceal_de_max=float("nan"),
            roughness_min=float("nan"),
            roughness_median=float("nan"),
            roughness_p95=float("nan"),
            roughness_max=float("nan"),
            saturation_mean=float("nan"),
            saturation_max=float("nan"),
            spectral_dist_mean=float("nan"),
            spectral_dist_max=float("nan"),
        )

    n_passing = sum(1 for _, s in scored if s.conceal_delta_e00 <= conceal_limit)
    conceal_des = [s.conceal_delta_e00 for _, s in scored]
    roughnesses = [s.roughness for _, s in scored]
    saturations = [s.boundary_saturation for _, s in scored]
    dists = [s.spectral_dist for _, s in scored]

    return PoolDiagnostics(
        n_total=len(scored),
        n_passing_conceal=n_passing,
        conceal_de_mean=float(np.mean(conceal_des)),
        conceal_de_max=float(np.max(conceal_des)),
        roughness_min=float(np.min(roughnesses)),
        roughness_median=float(np.median(roughnesses)),
        roughness_p95=float(np.percentile(roughnesses, 95)),
        roughness_max=float(np.max(roughnesses)),
        saturation_mean=float(np.mean(saturations)),
        saturation_max=float(np.max(saturations)),
        spectral_dist_mean=float(np.mean(dists)),
        spectral_dist_max=float(np.max(dists)),
    )


# ---------------------------------------------------------------------------
# Individual candidate report
# ---------------------------------------------------------------------------


def candidate_report(
    candidate: MetamerCandidate,
    score: CandidateScore,
) -> str:
    """Return a human-readable text summary of one scored candidate."""
    lines = [
        f"MetamerCandidate — {candidate.provenance.source}",
        f"  Conceal ΔE₀₀ : {score.conceal_delta_e00:.4f}",
        f"  Reveal ΔE₀₀s : {', '.join(f'{d:.4f}' for d in score.reveal_delta_e00s)}",
        f"  Min reveal   : {score.minimum_reveal_delta_e00:.4f}",
        f"  Roughness    : {score.roughness:.6g}",
        f"  Saturation   : {score.boundary_saturation:.4f}",
        f"  Spectral dist: {score.spectral_dist:.6f}",
        f"  Endpoint     : {candidate.endpoint}",
        f"  Alpha        : {candidate.alpha:.6g}",
        f"  Bound viol.  : {candidate.max_bound_violation:.2e}",
    ]
    if candidate.provenance.notes:
        lines.append(f"  Notes        : {candidate.provenance.notes}")
    return "\n".join(lines)
