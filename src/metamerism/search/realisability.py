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

from collections.abc import Mapping
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

DEFAULT_CONCEAL_DELTA_E00_LIMIT = 1.0
DEFAULT_N_RESTARTS = 8
DEFAULT_SEED = 0


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
        if self.mixing_mode != "reflectance":
            raise ValueError(
                f"unsupported mixing_mode {self.mixing_mode!r}; "
                f"only 'reflectance' is implemented in Phase G"
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

    # Build paint matrix and projection matrices
    paint_names, R_matrix = _align_paint_matrix(paint_library)
    n_paints = R_matrix.shape[1]

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
        r_mix = R_matrix @ x
        total = sum(
            _de00_fast(A @ r_mix, ref_xyz, white)
            for A, ref_xyz, white in zip(
                A_reveals, ref_xyz_reveals, white_reveals, strict=True
            )
        )
        return -total

    def conceal_de00_value(x: NDArray) -> float:
        return _de00_fast(A0 @ (R_matrix @ x), ref_xyz_conceal, white_conceal)

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

    r_mix = R_matrix @ best_x
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

    model_assumptions: Mapping[str, str] = {
        "model": MODEL_NAME_LINEAR,
        "mixing": "R_mix = sum(x_i * R_i)",
        "paint_library_size": str(n_paints),
        "conceal_illuminant": conceal_condition.name,
        "reveal_illuminants": ",".join(rc.name for rc in reveal_conditions),
        "limitations": (
            "ignores optical interactions between pigments; "
            "drawdown reflectances measured under specific film thickness and substrate"
        ),
    }

    return PaintRealisation(
        reference=ref_aligned,
        realised=realised_spectrum,
        fractions=fractions,
        conceal_delta_e00=conceal_cr.delta_e00,
        reveal_results=reveal_crs,
        ideal_candidate=ideal_spectrum,
        ideal_reveal_delta_e00=ideal_reveal_de00,
        model_name=MODEL_NAME_LINEAR,
        model_assumptions=model_assumptions,
        optimisation_success=any_success,
    )
