"""Paint realisability for metameric pairs.

Uses a linear-in-reflectance (baseline) forward model to find paint mixtures
from the Golden library that match a reference paint under the conceal illuminant
while maximising reveal ΔE₀₀.

The optimisation targets perceptual constraints (XYZ / Lab under each illuminant)
directly — not spectral closeness to an ideal candidate spectrum.  An ideal
candidate may be supplied as a warm start but does not appear in the objective.

Model assumptions (linear-in-reflectance baseline)
-------------------------------------------------
R_mix(x) = Σ xᵢ Rᵢ

where Rᵢ are measured Golden drawdown reflectances aligned to the project grid.
This is a convex mixing model: the XYZ under any illuminant is linear in x,
making both the conceal constraint and the reveal objective smooth functions
of the mixture fractions.

Limitations: this model ignores optical interactions between pigments.  It is
a transparent baseline, not a physical prediction.  Phase H replaces it with
Kubelka-Munk arithmetic mixing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import colour
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, nnls

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain
from metamerism.pigments.golden import GoldenPaintSpectrum
from metamerism.search.generate import MetamerCandidate
from metamerism.search.metamer import ConditionResult, ViewingCondition, score_condition
from metamerism.search.projection import build_xyz_projection

MODEL_NAME_LINEAR = "linear-in-reflectance"
MODEL_NAME_KS = "kubelka-munk-arithmetic"

DEFAULT_CONCEAL_DELTA_E00_LIMIT = 1.0
DEFAULT_N_RESTARTS = 8
DEFAULT_SEED = 0
DEFAULT_MAX_COMPONENTS = 4
MAX_COMPONENTS_LIMIT = 6
_KS_EPS = 1e-9  # reflectance clip before K/S conversion


@dataclass(frozen=True)
class RealisabilityConfig:
    """Configuration for paint mixture optimisation.

    Attributes:
        conceal_delta_e00_limit: Maximum ΔE₀₀ between the mixture and reference
            under the conceal illuminant.  Must be > 0.
        n_restarts: Number of random restarts for the nonlinear solver.
            More restarts reduce the risk of local optima.
        seed: Random seed for reproducible restarts.
        mixing_mode: Forward model label.  Only "reflectance" (linear baseline)
            is implemented in Phase G.
    """

    conceal_delta_e00_limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT
    n_restarts: int = DEFAULT_N_RESTARTS
    seed: int = DEFAULT_SEED
    mixing_mode: str = "reflectance"

    def __post_init__(self) -> None:
        if self.conceal_delta_e00_limit <= 0.0:
            raise ValueError(
                f"conceal_delta_e00_limit must be > 0, "
                f"got {self.conceal_delta_e00_limit}"
            )
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {self.n_restarts}")
        if self.mixing_mode not in ("reflectance", "ks"):
            raise ValueError(
                f"unsupported mixing_mode {self.mixing_mode!r}; "
                f"must be 'reflectance' or 'ks'"
            )

    def to_dict(self) -> dict:
        return {
            "conceal_delta_e00_limit": self.conceal_delta_e00_limit,
            "n_restarts": self.n_restarts,
            "seed": self.seed,
            "mixing_mode": self.mixing_mode,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RealisabilityConfig:
        return cls(
            conceal_delta_e00_limit=d.get(
                "conceal_delta_e00_limit", DEFAULT_CONCEAL_DELTA_E00_LIMIT
            ),
            n_restarts=d.get("n_restarts", DEFAULT_N_RESTARTS),
            seed=d.get("seed", DEFAULT_SEED),
            mixing_mode=d.get("mixing_mode", "reflectance"),
        )


@dataclass(frozen=True)
class PaintRealisation:
    """A paint mixture realisation of a metameric pair.

    Attributes:
        reference: The fixed reference paint reflectance.
        realised: The optimised mixture reflectance.
        fractions: Paint name → mixture fraction (sums to 1).
        conceal_delta_e00: ΔE₀₀ between mixture and reference under conceal.
            Should be ≤ conceal_delta_e00_limit if optimisation succeeded.
        reveal_results: Per-illuminant ConditionResult objects (mixture vs reference).
        ideal_candidate: The ideal-search candidate used as warm start, if any.
        ideal_reveal_delta_e00: The ideal candidate's reveal ΔE₀₀ (upper-bound
            diagnostic showing how much of the ceiling the mixture achieves).
        model_name: Label for the forward model used.
        model_assumptions: Key-value record of modelling assumptions.
        optimisation_success: True if the solver reported convergence.
    """

    reference: Spectrum
    realised: Spectrum
    fractions: Mapping[str, float]
    conceal_delta_e00: float
    reveal_results: tuple[ConditionResult, ...]
    ideal_candidate: Spectrum | None
    ideal_reveal_delta_e00: float | None
    model_name: str
    model_assumptions: Mapping[str, str]
    optimisation_success: bool

    @property
    def minimum_reveal_delta_e00(self) -> float:
        """Minimum reveal ΔE₀₀ across all reveal conditions."""
        if not self.reveal_results:
            return 0.0
        return min(r.delta_e00 for r in self.reveal_results)

    def to_dict(self) -> dict:
        return {
            "reference_wavelengths": self.reference.wavelengths.tolist(),
            "reference_values": self.reference.values.tolist(),
            "realised_wavelengths": self.realised.wavelengths.tolist(),
            "realised_values": self.realised.values.tolist(),
            "fractions": dict(self.fractions),
            "conceal_delta_e00": self.conceal_delta_e00,
            "reveal_results": [r.to_dict() for r in self.reveal_results],
            "ideal_candidate_values": (
                self.ideal_candidate.values.tolist()
                if self.ideal_candidate is not None
                else None
            ),
            "ideal_reveal_delta_e00": self.ideal_reveal_delta_e00,
            "model_name": self.model_name,
            "model_assumptions": dict(self.model_assumptions),
            "optimisation_success": self.optimisation_success,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PaintRealisation:
        ref = Spectrum.from_arrays(
            np.array(d["reference_wavelengths"], dtype=np.float64),
            np.array(d["reference_values"], dtype=np.float64),
            domain=SpectrumDomain.REFLECTANCE,
            provenance=Provenance(source="restored_reference"),
            validate=False,
        )
        realised = Spectrum.from_arrays(
            np.array(d["realised_wavelengths"], dtype=np.float64),
            np.array(d["realised_values"], dtype=np.float64),
            domain=SpectrumDomain.REFLECTANCE,
            provenance=Provenance(source="restored_realised"),
            validate=False,
        )
        ideal_cand: Spectrum | None = None
        if d.get("ideal_candidate_values") is not None:
            ideal_cand = Spectrum.from_arrays(
                np.array(d["realised_wavelengths"], dtype=np.float64),
                np.array(d["ideal_candidate_values"], dtype=np.float64),
                domain=SpectrumDomain.REFLECTANCE,
                provenance=Provenance(source="restored_ideal_candidate"),
                validate=False,
            )
        return cls(
            reference=ref,
            realised=realised,
            fractions=d["fractions"],
            conceal_delta_e00=d["conceal_delta_e00"],
            reveal_results=tuple(
                ConditionResult.from_dict(r) for r in d["reveal_results"]
            ),
            ideal_candidate=ideal_cand,
            ideal_reveal_delta_e00=d.get("ideal_reveal_delta_e00"),
            model_name=d["model_name"],
            model_assumptions=d["model_assumptions"],
            optimisation_success=d.get("optimisation_success", False),
        )


# ── internal helpers ───────────────────────────────────────────────────────────


def _align_paint_matrix(
    paints: list[GoldenPaintSpectrum],
) -> tuple[list[str], NDArray[np.float64]]:
    """Align all paint reflectances to the project grid; return (names, matrix).

    Returns:
        names: Paint names in the same order as matrix columns.
        matrix: Shape (n_wavelengths, n_paints).  Each column is the aligned
            reflectance of one paint.
    """
    names: list[str] = []
    columns: list[NDArray[np.float64]] = []
    for paint in paints:
        sd = paint.reflectance.sd.copy()
        if sd.shape != PROJECT_SHAPE:
            sd = sd.align(PROJECT_SHAPE)
        names.append(paint.paint_name)
        columns.append(np.asarray(sd.values, dtype=np.float64))
    return names, np.stack(columns, axis=1)


def _align_paint_ks_matrix(
    paints: list[GoldenPaintSpectrum],
) -> tuple[list[str], NDArray[np.float64]]:
    """Align all paint K/S spectra to the project grid; return (names, matrix).

    Returns:
        names: Paint names in the same order as matrix columns.
        matrix: Shape (n_wavelengths, n_paints).  Each column is the K/S ratio
            of one paint derived from its white-substrate drawdown reflectance.
    """
    names: list[str] = []
    columns: list[NDArray[np.float64]] = []
    for paint in paints:
        sd = paint.ks.sd.copy()
        if sd.shape != PROJECT_SHAPE:
            sd = sd.align(PROJECT_SHAPE)
        names.append(paint.paint_name)
        columns.append(np.asarray(sd.values, dtype=np.float64))
    return names, np.stack(columns, axis=1)


def _white_xyz_unit(projection_matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return XYZ of a perfect reflector under the illuminant, Y=1 scale.

    Used as the reference white for XYZ_to_Lab, matching the convention in
    score_condition (which computes xyz_white via colour.sd_to_XYZ).
    """
    n_wl = projection_matrix.shape[1]
    return (projection_matrix @ np.ones(n_wl)) / 100.0


def _de00_fast(
    xyz_a: NDArray[np.float64],
    xyz_b: NDArray[np.float64],
    white_xyz_unit: NDArray[np.float64],
) -> float:
    """CIEDE2000 from projection XYZ (Y=100 scale), using a precomputed white."""
    lab_a = colour.XYZ_to_Lab(xyz_a / 100.0, illuminant=white_xyz_unit)
    lab_b = colour.XYZ_to_Lab(xyz_b / 100.0, illuminant=white_xyz_unit)
    return float(colour.delta_E(lab_a, lab_b, method="CIE 2000"))


def _warm_start_from_target(
    R_matrix: NDArray[np.float64],
    r_target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Project r_target onto the mixture simplex via NNLS; normalise to sum=1."""
    x, _ = nnls(R_matrix, r_target)
    total = float(np.sum(x))
    if total < 1e-12:
        n = R_matrix.shape[1]
        return np.ones(n) / n
    return x / total


# ── forward-model factories ────────────────────────────────────────────────────


def _make_reflectance_forward(
    R_sub: NDArray[np.float64],
) -> Callable[[NDArray], NDArray[np.float64]]:
    def fwd(x: NDArray) -> NDArray[np.float64]:
        return R_sub @ x

    return fwd


def _make_ks_forward(
    KS_sub: NDArray[np.float64],
) -> Callable[[NDArray], NDArray[np.float64]]:
    def fwd(x: NDArray) -> NDArray[np.float64]:
        ks_mix = np.clip(KS_sub @ x, 0.0, None)
        return 1.0 + ks_mix - np.sqrt(ks_mix * ks_mix + 2.0 * ks_mix)

    return fwd


# ── shared SLSQP core ──────────────────────────────────────────────────────────


def _run_slsqp(
    forward: Callable[[NDArray], NDArray[np.float64]],
    n_vars: int,
    A_reveals: list[NDArray[np.float64]],
    ref_xyz_reveals: list[NDArray[np.float64]],
    white_reveals: list[NDArray[np.float64]],
    A0: NDArray[np.float64],
    ref_xyz_conceal: NDArray[np.float64],
    white_conceal: NDArray[np.float64],
    conceal_limit: float,
    starts: list[NDArray[np.float64]],
) -> tuple[NDArray[np.float64], float, bool]:
    """Multi-start SLSQP: maximise sum reveal ΔE₀₀ subject to conceal constraint.

    Returns (best_x, best_neg_objective, any_success).  Negative objective
    means a positive reveal sum was achieved; the caller negates to recover it.
    """

    def neg_sum_reveal(x: NDArray) -> float:
        r_mix = forward(x)
        return -sum(
            _de00_fast(A @ r_mix, ref_xyz, white)
            for A, ref_xyz, white in zip(
                A_reveals, ref_xyz_reveals, white_reveals, strict=True
            )
        )

    def conceal_de00(x: NDArray) -> float:
        return _de00_fast(A0 @ forward(x), ref_xyz_conceal, white_conceal)

    constraints = [
        {"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0},
        {"type": "ineq", "fun": lambda x: conceal_limit - conceal_de00(x)},
    ]
    bounds = [(0.0, 1.0)] * n_vars

    best_x = starts[0].copy()
    best_obj = float("inf")
    best_feasible = False
    any_success = False

    for x0 in starts:
        result = minimize(
            neg_sum_reveal,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        x_cand = np.asarray(result.x, dtype=np.float64)
        is_feasible = conceal_de00(x_cand) <= conceal_limit
        better = (is_feasible and not best_feasible) or (
            is_feasible == best_feasible and result.fun < best_obj
        )
        if better:
            best_obj = float(result.fun)
            best_x = x_cand
            best_feasible = is_feasible
            if result.success:
                any_success = True

    best_x = np.clip(best_x, 0.0, None)
    total = float(np.sum(best_x))
    if total < 1e-12:
        best_x = np.ones(n_vars, dtype=np.float64) / n_vars
    else:
        best_x /= total

    return best_x, best_obj, any_success


# ── OMP ordering ───────────────────────────────────────────────────────────────


def _omp_paint_order(
    paint_matrix: NDArray[np.float64],
    target: NDArray[np.float64],
) -> list[int]:
    """Rank paint columns by absolute cosine similarity with target (highest first)."""
    col_norms = np.linalg.norm(paint_matrix, axis=0)
    col_norms = np.where(col_norms < 1e-12, 1e-12, col_norms)
    target_norm = float(np.linalg.norm(target))
    if target_norm < 1e-12:
        return list(range(paint_matrix.shape[1]))
    correlations = (paint_matrix.T @ target) / (col_norms * target_norm)
    return list(np.argsort(-np.abs(correlations)))


# ── SparseRealisabilityConfig ──────────────────────────────────────────────────


@dataclass(frozen=True)
class SparseRealisabilityConfig:
    """Configuration for sparse paint mixture optimisation.

    Uses greedy forward selection to build a mixture of at most max_components
    paints.  At each step the paint that most improves reveal ΔE₀₀ (subject to
    the conceal constraint) is added to the active set; selection stops when no
    remaining paint improves the objective.

    If use_omp_warmstart is True and an ideal_candidate is supplied to
    find_sparse_metameric_mixture, paints are ranked by K/S (or reflectance)
    correlation with the candidate before the greedy loop, so the most
    relevant paints are tried first.

    Attributes:
        conceal_delta_e00_limit: Maximum ΔE₀₀ under the conceal illuminant.
        max_components: Maximum number of paints in the mixture
            (1-MAX_COMPONENTS_LIMIT).
        n_restarts: Random restarts per greedy step per paint candidate.
        seed: Random seed for reproducible restarts.
        mixing_mode: "reflectance" (linear baseline) or "ks" (K-M arithmetic).
        use_omp_warmstart: Order paints by spectral correlation with the ideal
            candidate before greedy selection.
    """

    conceal_delta_e00_limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT
    max_components: int = DEFAULT_MAX_COMPONENTS
    n_restarts: int = DEFAULT_N_RESTARTS
    seed: int = DEFAULT_SEED
    mixing_mode: str = "ks"
    use_omp_warmstart: bool = True

    def __post_init__(self) -> None:
        if self.conceal_delta_e00_limit <= 0.0:
            raise ValueError(
                "conceal_delta_e00_limit must be > 0, "
                f"got {self.conceal_delta_e00_limit}"
            )
        if not (1 <= self.max_components <= MAX_COMPONENTS_LIMIT):
            raise ValueError(
                f"max_components must be between 1 and {MAX_COMPONENTS_LIMIT}, "
                f"got {self.max_components}"
            )
        if self.n_restarts < 1:
            raise ValueError(f"n_restarts must be >= 1, got {self.n_restarts}")
        if self.mixing_mode not in ("reflectance", "ks"):
            raise ValueError(
                f"unsupported mixing_mode {self.mixing_mode!r}; "
                f"must be 'reflectance' or 'ks'"
            )

    def to_dict(self) -> dict:
        return {
            "conceal_delta_e00_limit": self.conceal_delta_e00_limit,
            "max_components": self.max_components,
            "n_restarts": self.n_restarts,
            "seed": self.seed,
            "mixing_mode": self.mixing_mode,
            "use_omp_warmstart": self.use_omp_warmstart,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SparseRealisabilityConfig:
        return cls(
            conceal_delta_e00_limit=d.get(
                "conceal_delta_e00_limit", DEFAULT_CONCEAL_DELTA_E00_LIMIT
            ),
            max_components=d.get("max_components", DEFAULT_MAX_COMPONENTS),
            n_restarts=d.get("n_restarts", DEFAULT_N_RESTARTS),
            seed=d.get("seed", DEFAULT_SEED),
            mixing_mode=d.get("mixing_mode", "ks"),
            use_omp_warmstart=d.get("use_omp_warmstart", True),
        )


# ── main public function ───────────────────────────────────────────────────────


def find_metameric_mixture(
    reference: Spectrum,
    paint_library: list[GoldenPaintSpectrum],
    conceal_condition: ViewingCondition,
    reveal_conditions: list[ViewingCondition],
    config: RealisabilityConfig | None = None,
    *,
    ideal_candidate: MetamerCandidate | None = None,
) -> PaintRealisation:
    """Find a paint mixture metameric with the reference.

    Optimises mixture fractions x such that:
        - R_mix(x) = Σ xᵢ Rᵢ (linear-in-reflectance baseline)
        - ΔE₀₀_conceal(R_mix(x), reference) ≤ conceal_delta_e00_limit
        - ΔE₀₀_reveal is maximised (summed across all reveal conditions)

    The objective is the perceptual difference under reveal, not spectral
    closeness to an ideal candidate.  If an ideal_candidate is supplied it is
    used only as a warm start for the nonlinear solver.

    Args:
        reference: Fixed reference reflectance (domain=REFLECTANCE).
        paint_library: Paints available for mixing.  May include the reference
            paint itself.
        conceal_condition: Illuminant under which the mixture must match reference.
        reveal_conditions: One or more illuminants under which the mixture should
            differ from reference.
        config: Optimisation settings; defaults to RealisabilityConfig().
        ideal_candidate: Optional ideal-search candidate for warm starting and
            diagnostic ceiling.

    Returns:
        PaintRealisation with the best mixture found, its perceptual scores, and
        the ideal-candidate ceiling if supplied.
    """
    if reference.domain is not SpectrumDomain.REFLECTANCE:
        raise ValueError(
            f"reference must have domain REFLECTANCE, got {reference.domain}"
        )
    if not paint_library:
        raise ValueError("paint_library must contain at least one paint")
    if not reveal_conditions:
        raise ValueError("at least one reveal condition is required")

    cfg = config or RealisabilityConfig()

    # Align reference to project grid
    ref_sd = reference.sd.copy()
    if ref_sd.shape != PROJECT_SHAPE:
        ref_sd = ref_sd.align(PROJECT_SHAPE)
    r_ref = np.asarray(ref_sd.values, dtype=np.float64)
    wavelengths = np.asarray(ref_sd.wavelengths, dtype=np.float64)
    ref_aligned = Spectrum.from_arrays(
        wavelengths,
        r_ref,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=reference.provenance,
        validate=False,
    )

    # Build paint matrices and projection matrices
    paint_names, R_matrix = _align_paint_matrix(paint_library)
    n_paints = R_matrix.shape[1]

    if cfg.mixing_mode == "ks":
        _, KS_matrix = _align_paint_ks_matrix(paint_library)

        def _forward(x: NDArray) -> NDArray[np.float64]:
            ks_mix = np.clip(KS_matrix @ x, 0.0, None)
            return 1.0 + ks_mix - np.sqrt(ks_mix * ks_mix + 2.0 * ks_mix)
    else:
        def _forward(x: NDArray) -> NDArray[np.float64]:
            return R_matrix @ x

    proj_conceal = build_xyz_projection(conceal_condition)
    proj_reveals = [build_xyz_projection(rc) for rc in reveal_conditions]

    A0 = proj_conceal.matrix
    A_reveals = [p.matrix for p in proj_reveals]

    # Reference XYZ and Lab white points
    ref_xyz_conceal = A0 @ r_ref
    ref_xyz_reveals = [A @ r_ref for A in A_reveals]
    white_conceal = _white_xyz_unit(A0)
    white_reveals = [_white_xyz_unit(A) for A in A_reveals]

    # ── objective and constraint functions ─────────────────────────────────────

    def neg_sum_reveal_de00(x: NDArray) -> float:
        r_mix = _forward(x)
        total = sum(
            _de00_fast(A @ r_mix, ref_xyz, white)
            for A, ref_xyz, white in zip(
                A_reveals, ref_xyz_reveals, white_reveals, strict=True
            )
        )
        return -total

    def conceal_de00_value(x: NDArray) -> float:
        return _de00_fast(A0 @ _forward(x), ref_xyz_conceal, white_conceal)

    constraints = [
        {"type": "eq", "fun": lambda x: float(np.sum(x)) - 1.0},
        {
            "type": "ineq",
            "fun": lambda x: cfg.conceal_delta_e00_limit - conceal_de00_value(x),
        },
    ]
    bounds = [(0.0, 1.0)] * n_paints

    # ── starting points ────────────────────────────────────────────────────────

    starts: list[NDArray[np.float64]] = []

    if ideal_candidate is not None:
        ideal_sd = ideal_candidate.spectrum.sd.copy()
        if ideal_sd.shape != PROJECT_SHAPE:
            ideal_sd = ideal_sd.align(PROJECT_SHAPE)
        r_ideal = np.asarray(ideal_sd.values, dtype=np.float64)
        starts.append(_warm_start_from_target(R_matrix, r_ideal))

    # Uniform start — always included
    starts.append(np.ones(n_paints, dtype=np.float64) / n_paints)

    # Random Dirichlet starts
    rng = np.random.default_rng(cfg.seed)
    while len(starts) < cfg.n_restarts:
        starts.append(rng.dirichlet(np.ones(n_paints)))

    # ── multi-start optimisation ───────────────────────────────────────────────

    best_x: NDArray[np.float64] | None = None
    best_obj = float("inf")
    any_success = False

    for x0 in starts:
        result = minimize(
            neg_sum_reveal_de00,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_x = np.asarray(result.x, dtype=np.float64)
            if result.success:
                any_success = True

    # Fallback: uniform if nothing better was found
    if best_x is None:
        best_x = np.ones(n_paints, dtype=np.float64) / n_paints

    # Project onto simplex (clip negatives, renormalise)
    best_x = np.clip(best_x, 0.0, None)
    total = float(np.sum(best_x))
    if total < 1e-12:
        best_x = np.ones(n_paints, dtype=np.float64) / n_paints
    else:
        best_x /= total

    # ── build output ───────────────────────────────────────────────────────────

    r_mix = _forward(best_x)
    fractions = {
        name: float(w) for name, w in zip(paint_names, best_x, strict=True)
    }

    realised_prov = Provenance(
        source=f"paint_mixture/{cfg.mixing_mode}",
        notes=(
            f"conceal={conceal_condition.name}; "
            f"reveal={','.join(rc.name for rc in reveal_conditions)}; "
            f"n_restarts={cfg.n_restarts}; seed={cfg.seed}; "
            f"n_paints={n_paints}"
        ),
    )
    realised_spectrum = Spectrum.from_arrays(
        wavelengths,
        r_mix,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=realised_prov,
        validate=False,
    )

    # Authoritative scores via colour-science
    conceal_cr = score_condition(ref_aligned, realised_spectrum, conceal_condition)
    reveal_crs = tuple(
        score_condition(ref_aligned, realised_spectrum, rc) for rc in reveal_conditions
    )

    # Ideal candidate ceiling
    ideal_spectrum: Spectrum | None = None
    ideal_reveal_de00: float | None = None
    if ideal_candidate is not None:
        ideal_spectrum = ideal_candidate.spectrum
        # Sum of reveal ΔE₀₀ across conditions for comparison with realised
        ideal_reveal_de00 = sum(
            score_condition(ref_aligned, ideal_spectrum, rc).delta_e00
            for rc in reveal_conditions
        )

    if cfg.mixing_mode == "ks":
        active_model_name = MODEL_NAME_KS
        mixing_formula = "(K/S)_mix = sum(x_i * (K/S)_i); R_mix via K-M inversion"
        limitations = (
            "arithmetic mean of K/S ratios assumes equal scattering S across "
            "components — valid for organic pigments in the same vehicle; "
            "K and S cannot be separated from white-substrate drawdown data alone; "
            "drawdown reflectances measured under specific film thickness and substrate"
        )
    else:
        active_model_name = MODEL_NAME_LINEAR
        mixing_formula = "R_mix = sum(x_i * R_i)"
        limitations = (
            "ignores optical interactions between pigments; "
            "drawdown reflectances measured under specific film thickness and substrate"
        )

    model_assumptions: Mapping[str, str] = {
        "model": active_model_name,
        "mixing": mixing_formula,
        "paint_library_size": str(n_paints),
        "conceal_illuminant": conceal_condition.name,
        "reveal_illuminants": ",".join(rc.name for rc in reveal_conditions),
        "conceal_delta_e00_limit": str(cfg.conceal_delta_e00_limit),
        "limitations": limitations,
    }

    return PaintRealisation(
        reference=ref_aligned,
        realised=realised_spectrum,
        fractions=fractions,
        conceal_delta_e00=conceal_cr.delta_e00,
        reveal_results=reveal_crs,
        ideal_candidate=ideal_spectrum,
        ideal_reveal_delta_e00=ideal_reveal_de00,
        model_name=active_model_name,
        model_assumptions=model_assumptions,
        optimisation_success=any_success,
    )


def find_sparse_metameric_mixture(
    reference: Spectrum,
    paint_library: list[GoldenPaintSpectrum],
    conceal_condition: ViewingCondition,
    reveal_conditions: list[ViewingCondition],
    config: SparseRealisabilityConfig | None = None,
    *,
    ideal_candidate: MetamerCandidate | None = None,
) -> PaintRealisation:
    """Find a sparse paint mixture metameric with the reference.

    Uses greedy forward selection: at each step the paint that most improves
    the sum reveal ΔE₀₀ (subject to the conceal constraint) is added to the
    active set.  Selection stops when no remaining paint improves the objective
    or max_components is reached.

    Cost is O(n_paints * max_components) optimisations rather than the
    O(C(n, k)) full enumeration.

    If use_omp_warmstart=True and an ideal_candidate is supplied, paints are
    first ranked by spectral (K/S or reflectance) cosine similarity with the
    candidate, so the greedy loop tries the most relevant paints first.

    The fractions mapping in the returned PaintRealisation contains only the
    selected active paints (at most max_components entries).

    Args:
        reference: Fixed reference reflectance (domain=REFLECTANCE).
        paint_library: Available paints.
        conceal_condition: Illuminant under which mixture must match reference.
        reveal_conditions: Illuminants under which mixture should differ.
        config: Optimisation settings; defaults to SparseRealisabilityConfig().
        ideal_candidate: Optional ideal-search candidate for OMP ordering and
            diagnostic ceiling reporting.

    Returns:
        PaintRealisation with at most max_components non-zero fractions.
    """
    if reference.domain is not SpectrumDomain.REFLECTANCE:
        raise ValueError(
            f"reference must have domain REFLECTANCE, got {reference.domain}"
        )
    if not paint_library:
        raise ValueError("paint_library must contain at least one paint")
    if not reveal_conditions:
        raise ValueError("at least one reveal condition is required")

    cfg = config or SparseRealisabilityConfig()

    # Align reference to project grid
    ref_sd = reference.sd.copy()
    if ref_sd.shape != PROJECT_SHAPE:
        ref_sd = ref_sd.align(PROJECT_SHAPE)
    r_ref = np.asarray(ref_sd.values, dtype=np.float64)
    wavelengths = np.asarray(ref_sd.wavelengths, dtype=np.float64)
    ref_aligned = Spectrum.from_arrays(
        wavelengths,
        r_ref,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=reference.provenance,
        validate=False,
    )

    # Build full paint matrices and projection matrices
    paint_names, R_full = _align_paint_matrix(paint_library)
    n_paints = len(paint_names)
    KS_full: NDArray[np.float64] | None = None
    if cfg.mixing_mode == "ks":
        _, KS_full = _align_paint_ks_matrix(paint_library)

    proj_conceal = build_xyz_projection(conceal_condition)
    proj_reveals = [build_xyz_projection(rc) for rc in reveal_conditions]
    A0 = proj_conceal.matrix
    A_reveals = [p.matrix for p in proj_reveals]
    ref_xyz_conceal = A0 @ r_ref
    ref_xyz_reveals = [A @ r_ref for A in A_reveals]
    white_conceal = _white_xyz_unit(A0)
    white_reveals = [_white_xyz_unit(A) for A in A_reveals]

    # OMP paint ordering: rank by cosine similarity with ideal candidate
    if cfg.use_omp_warmstart and ideal_candidate is not None:
        ideal_sd = ideal_candidate.spectrum.sd.copy()
        if ideal_sd.shape != PROJECT_SHAPE:
            ideal_sd = ideal_sd.align(PROJECT_SHAPE)
        r_ideal = np.asarray(ideal_sd.values, dtype=np.float64)
        if cfg.mixing_mode == "ks" and KS_full is not None:
            r_clipped = np.clip(r_ideal, _KS_EPS, 1.0)
            ks_ideal = ((1.0 - r_clipped) ** 2) / (2.0 * r_clipped)
            paint_order = _omp_paint_order(KS_full, ks_ideal)
        else:
            paint_order = _omp_paint_order(R_full, r_ideal)
        ideal_spectrum_ref: Spectrum | None = ideal_candidate.spectrum
        ideal_reveal_de00: float | None = None
    else:
        paint_order = list(range(n_paints))
        ideal_spectrum_ref = None
        ideal_reveal_de00 = None

    # ── greedy forward selection ───────────────────────────────────────────────
    #
    # Acceptance uses feasibility-aware comparison: a feasible solution always
    # beats an infeasible one, regardless of the raw reveal objective.  This
    # prevents the greedy from locking onto a high-reveal but constraint-
    # violating single paint and then refusing to add a second paint that makes
    # the mixture feasible (at the cost of some reveal).

    rng = np.random.default_rng(cfg.seed)
    active_indices: list[int] = []
    best_overall_x: NDArray[np.float64] | None = None
    best_overall_obj = float("inf")
    best_overall_feasible = False
    best_overall_success = False

    for _step in range(cfg.max_components):
        remaining = [i for i in paint_order if i not in active_indices]
        if not remaining:
            break

        best_step_obj = float("inf")
        best_step_idx = -1
        best_step_x: NDArray[np.float64] | None = None
        best_step_feasible = False
        best_step_success = False

        for idx in remaining:
            trial_indices = [*active_indices, idx]
            k = len(trial_indices)

            if cfg.mixing_mode == "ks" and KS_full is not None:
                forward = _make_ks_forward(KS_full[:, trial_indices])
            else:
                forward = _make_reflectance_forward(R_full[:, trial_indices])

            # Warm start: extend previous best fractions, give new paint 10%
            starts: list[NDArray[np.float64]] = []
            if best_overall_x is not None and len(best_overall_x) == k - 1:
                x_ext = np.zeros(k, dtype=np.float64)
                x_ext[: k - 1] = best_overall_x * 0.9
                x_ext[k - 1] = 0.1
                starts.append(x_ext)
            starts.append(np.ones(k, dtype=np.float64) / k)
            while len(starts) < cfg.n_restarts:
                starts.append(rng.dirichlet(np.ones(k)).astype(np.float64))

            x, obj, success = _run_slsqp(
                forward,
                k,
                A_reveals,
                ref_xyz_reveals,
                white_reveals,
                A0,
                ref_xyz_conceal,
                white_conceal,
                cfg.conceal_delta_e00_limit,
                starts,
            )

            trial_conceal = _de00_fast(
                A0 @ forward(x), ref_xyz_conceal, white_conceal
            )
            feasible = trial_conceal <= cfg.conceal_delta_e00_limit

            # Feasibility-aware comparison within this step
            better = (feasible and not best_step_feasible) or (
                feasible == best_step_feasible and obj < best_step_obj
            )
            if better:
                best_step_obj = obj
                best_step_idx = idx
                best_step_x = x
                best_step_feasible = feasible
                best_step_success = success

        if best_step_idx < 0:
            break

        # Accept step: step 0 always accepted (need at least one paint);
        # subsequent steps require feasibility or objective improvement.
        if _step == 0:
            accept = True
        else:
            accept = (best_step_feasible and not best_overall_feasible) or (
                best_step_feasible == best_overall_feasible
                and best_step_obj < best_overall_obj
            )

        if accept:
            active_indices.append(best_step_idx)
            best_overall_x = best_step_x
            best_overall_obj = best_step_obj
            best_overall_feasible = best_step_feasible
            best_overall_success = best_step_success
        else:
            break  # no further improvement possible

    # Fallback: single uniform mixture if nothing converged
    if not active_indices or best_overall_x is None:
        active_indices = list(range(min(cfg.max_components, n_paints)))
        best_overall_x = np.ones(len(active_indices), dtype=np.float64) / len(
            active_indices
        )
        best_overall_success = False

    # ── build output ───────────────────────────────────────────────────────────

    active_names = [paint_names[i] for i in active_indices]
    n_active = len(active_indices)

    if cfg.mixing_mode == "ks" and KS_full is not None:
        r_mix = _make_ks_forward(KS_full[:, active_indices])(best_overall_x)
    else:
        r_mix = R_full[:, active_indices] @ best_overall_x

    fractions = {
        name: float(w)
        for name, w in zip(active_names, best_overall_x, strict=True)
    }

    realised_prov = Provenance(
        source=f"sparse_paint_mixture/{cfg.mixing_mode}",
        notes=(
            f"conceal={conceal_condition.name}; "
            f"reveal={','.join(rc.name for rc in reveal_conditions)}; "
            f"max_components={cfg.max_components}; "
            f"n_components={n_active}; "
            f"n_restarts={cfg.n_restarts}; seed={cfg.seed}"
        ),
    )
    realised_spectrum = Spectrum.from_arrays(
        wavelengths,
        r_mix,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=realised_prov,
        validate=False,
    )

    conceal_cr = score_condition(ref_aligned, realised_spectrum, conceal_condition)
    reveal_crs = tuple(
        score_condition(ref_aligned, realised_spectrum, rc) for rc in reveal_conditions
    )

    if ideal_spectrum_ref is not None:
        ideal_reveal_de00 = sum(
            score_condition(ref_aligned, ideal_spectrum_ref, rc).delta_e00
            for rc in reveal_conditions
        )

    if cfg.mixing_mode == "ks":
        active_model_name = MODEL_NAME_KS
        mixing_formula = "(K/S)_mix = sum(x_i * (K/S)_i); R_mix via K-M inversion"
        limitations = (
            "arithmetic mean of K/S ratios assumes equal scattering S across "
            "components — valid for organic pigments in the same vehicle; "
            "K and S cannot be separated from white-substrate drawdown data alone; "
            "drawdown reflectances measured under specific film thickness and substrate"
        )
    else:
        active_model_name = MODEL_NAME_LINEAR
        mixing_formula = "R_mix = sum(x_i * R_i)"
        limitations = (
            "ignores optical interactions between pigments; "
            "drawdown reflectances measured under specific film thickness and substrate"
        )

    model_assumptions: Mapping[str, str] = {
        "model": active_model_name,
        "mixing": mixing_formula,
        "algorithm": "greedy_forward_selection",
        "max_components": str(cfg.max_components),
        "n_components_used": str(n_active),
        "omp_warmstart": str(cfg.use_omp_warmstart),
        "paint_library_size": str(n_paints),
        "conceal_illuminant": conceal_condition.name,
        "reveal_illuminants": ",".join(rc.name for rc in reveal_conditions),
        "conceal_delta_e00_limit": str(cfg.conceal_delta_e00_limit),
        "limitations": limitations,
    }

    return PaintRealisation(
        reference=ref_aligned,
        realised=realised_spectrum,
        fractions=fractions,
        conceal_delta_e00=conceal_cr.delta_e00,
        reveal_results=reveal_crs,
        ideal_candidate=ideal_spectrum_ref,
        ideal_reveal_delta_e00=ideal_reveal_de00,
        model_name=active_model_name,
        model_assumptions=model_assumptions,
        optimisation_success=best_overall_success,
    )
