"""Bounded ideal-metamer candidate generation.

Given a reference reflectance spectrum and a conceal viewing condition,
generates partner spectra that match the reference under the conceal illuminant
but may differ under reveal illuminants.

Algorithm (directional):
1. Align reference to the project spectral shape.
2. Build the conceal XYZ projection and null-space basis N.
3. Rank null-space basis vectors by second-difference roughness and
   weight random direction sampling toward smoother vectors.
4. Sample a unit direction d = N z / ||N z||.
5. Compute feasible interval [alpha_min, alpha_max] for r_c + alpha * d in [0,1].
6. Step to inset endpoints r_- and r_+ using endpoint_fraction f.
7. Compute roughness of each candidate immediately.
8. Wrap each candidate in a Spectrum with full provenance.
9. Verify conceal ΔE₀₀ via direct colour.sd_to_XYZ.
10. Retain only candidates below the conceal threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain
from metamerism.search.feasibility import (
    DEFAULT_REFLECTANCE_BOUND_TOLERANCE,
    feasible_step_interval,
    is_feasible,
    max_bound_violation,
)
from metamerism.search.metamer import (
    DEFAULT_CONCEAL_DELTA_E00_LIMIT,
    ViewingCondition,
    score_condition,
)
from metamerism.search.projection import (
    DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE,
    build_xyz_projection,
    compute_null_space,
)

DEFAULT_ENDPOINT_FRACTION = 0.95
# Step to 95% of the feasibility boundary. Avoids forcing wavelengths to
# exact 0/1 while still exploring close to the edges of the feasible region.

_ROUGHNESS_EPS = 1e-12
# Small constant added to roughness when computing sampling weights, so that
# perfectly smooth basis vectors (roughness=0) get finite weight.


def _second_difference_matrix(n: int) -> NDArray[np.float64]:
    """Return the (n-2) x n second-difference operator D2."""
    D2 = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D2[i, i] = 1.0
        D2[i, i + 1] = -2.0
        D2[i, i + 2] = 1.0
    return D2


def spectral_roughness(values: NDArray[np.float64]) -> float:
    """Return the second-difference roughness ||D2 r||_2^2.

    Lower values indicate smoother spectra. Physically plausible reflectance
    curves from real paint measurements have low roughness.
    """
    v = np.asarray(values, dtype=np.float64)
    d2 = np.diff(np.diff(v))
    return float(np.dot(d2, d2))


def _basis_roughnesses(basis: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return roughness of each column of basis (shape n_null,)."""
    n_null = basis.shape[1]
    return np.array([spectral_roughness(basis[:, j]) for j in range(n_null)])


def _sampling_weights(roughnesses: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return normalised sampling weights proportional to 1 / (roughness + eps)."""
    weights = 1.0 / (roughnesses + _ROUGHNESS_EPS)
    return weights / weights.sum()


@dataclass(frozen=True)
class CandidateGenerationConfig:
    """Configuration for directional metamer generation.

    Attributes:
        seed: Random seed for reproducibility.
        n_directions: Number of null-space directions to sample.
        endpoint_fraction: Boundary proximity f in (0, 1]. f=1 steps to the
            exact feasibility boundary; f=0.95 steps to 95% of the way from
            the centre to the boundary. Values closer to 0 stay near centre.
        conceal_delta_e00_limit: Maximum conceal ΔE₀₀ for a candidate to pass.
        reflectance_bound_tolerance: Maximum allowed reflectance bound
            violation before a candidate is flagged as infeasible.
        null_space_relative_tolerance: SVD threshold for null-space rank.
    """

    seed: int
    n_directions: int
    endpoint_fraction: float = DEFAULT_ENDPOINT_FRACTION
    conceal_delta_e00_limit: float = DEFAULT_CONCEAL_DELTA_E00_LIMIT
    reflectance_bound_tolerance: float = DEFAULT_REFLECTANCE_BOUND_TOLERANCE
    null_space_relative_tolerance: float = DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE

    def __post_init__(self) -> None:
        if not 0.0 < self.endpoint_fraction <= 1.0:
            raise ValueError(
                f"endpoint_fraction must be in (0, 1], got {self.endpoint_fraction}"
            )
        if self.n_directions < 1:
            raise ValueError(f"n_directions must be >= 1, got {self.n_directions}")
        if self.conceal_delta_e00_limit <= 0.0:
            raise ValueError(
                f"conceal_delta_e00_limit must be > 0, "
                f"got {self.conceal_delta_e00_limit}"
            )

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "n_directions": self.n_directions,
            "endpoint_fraction": self.endpoint_fraction,
            "conceal_delta_e00_limit": self.conceal_delta_e00_limit,
            "reflectance_bound_tolerance": self.reflectance_bound_tolerance,
            "null_space_relative_tolerance": self.null_space_relative_tolerance,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CandidateGenerationConfig:
        return cls(
            seed=d["seed"],
            n_directions=d["n_directions"],
            endpoint_fraction=d.get("endpoint_fraction", DEFAULT_ENDPOINT_FRACTION),
            conceal_delta_e00_limit=d.get(
                "conceal_delta_e00_limit", DEFAULT_CONCEAL_DELTA_E00_LIMIT
            ),
            reflectance_bound_tolerance=d.get(
                "reflectance_bound_tolerance", DEFAULT_REFLECTANCE_BOUND_TOLERANCE
            ),
            null_space_relative_tolerance=d.get(
                "null_space_relative_tolerance",
                DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE,
            ),
        )


@dataclass(frozen=True)
class MetamerCandidate:
    """A bounded reflectance candidate that matches a reference under conceal.

    Attributes:
        spectrum: The generated reflectance spectrum.
        source_reference: The reference spectrum used to generate this candidate.
        null_space_coordinates: Coordinates z in null-space (before direction
            normalisation), for provenance and diagnostics.
        endpoint: Which endpoint was taken: "minus" or "plus".
        alpha: The scalar step taken along the direction.
        max_bound_violation: Largest reflectance bound violation (ideally ~0).
        roughness: Second-difference roughness ||D2 r||_2^2.
        conceal_delta_e00: ΔE₀₀ under the conceal condition (verified by
            direct colour.sd_to_XYZ).
        provenance: Full generation provenance.
    """

    spectrum: Spectrum
    source_reference: Spectrum
    null_space_coordinates: NDArray[np.float64]
    endpoint: str
    alpha: float
    max_bound_violation: float
    roughness: float
    conceal_delta_e00: float
    provenance: Provenance

    def to_dict(self) -> dict:
        return {
            "spectrum_wavelengths": self.spectrum.wavelengths.tolist(),
            "spectrum_values": self.spectrum.values.tolist(),
            "reference_wavelengths": self.source_reference.wavelengths.tolist(),
            "reference_values": self.source_reference.values.tolist(),
            "null_space_coordinates": self.null_space_coordinates.tolist(),
            "endpoint": self.endpoint,
            "alpha": self.alpha,
            "max_bound_violation": self.max_bound_violation,
            "roughness": self.roughness,
            "conceal_delta_e00": self.conceal_delta_e00,
            "provenance_source": self.provenance.source,
            "provenance_notes": self.provenance.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MetamerCandidate:
        from metamerism.core import Provenance, Spectrum, SpectrumDomain

        prov = Provenance(
            source=d["provenance_source"],
            notes=d.get("provenance_notes"),
        )
        spectrum = Spectrum.from_arrays(
            np.array(d["spectrum_wavelengths"], dtype=np.float64),
            np.array(d["spectrum_values"], dtype=np.float64),
            domain=SpectrumDomain.REFLECTANCE,
            provenance=prov,
            validate=False,
        )
        source_reference = Spectrum.from_arrays(
            np.array(d["reference_wavelengths"], dtype=np.float64),
            np.array(d["reference_values"], dtype=np.float64),
            domain=SpectrumDomain.REFLECTANCE,
            provenance=Provenance(source="restored_reference"),
            validate=False,
        )
        return cls(
            spectrum=spectrum,
            source_reference=source_reference,
            null_space_coordinates=np.array(
                d["null_space_coordinates"], dtype=np.float64
            ),
            endpoint=d["endpoint"],
            alpha=d["alpha"],
            max_bound_violation=d["max_bound_violation"],
            roughness=d["roughness"],
            conceal_delta_e00=d["conceal_delta_e00"],
            provenance=prov,
        )


def generate_directional_metamers(
    reference: Spectrum,
    conceal_condition: ViewingCondition,
    config: CandidateGenerationConfig,
) -> list[MetamerCandidate]:
    """Generate bounded null-space partner spectra for a reference reflectance.

    Samples random directions in the null space of the conceal XYZ projection,
    computes the feasible step interval for each direction, steps to inset
    endpoints, and returns candidates verified by direct colour scoring.

    Args:
        reference: The reference reflectance spectrum (domain=REFLECTANCE).
        conceal_condition: Illuminant/observer under which pair should match.
        config: Generation parameters including seed, n_directions, and thresholds.

    Returns:
        List of MetamerCandidate objects passing the conceal ΔE₀₀ threshold,
        sorted by roughness (smoothest first). May be shorter than
        2 * n_directions if many directions have degenerate feasible intervals
        or fail the conceal threshold.
    """
    if reference.domain is not SpectrumDomain.REFLECTANCE:
        raise ValueError(
            f"reference must have domain REFLECTANCE, got {reference.domain}"
        )

    # Align reference to project grid
    ref_sd = reference.sd.copy()
    if ref_sd.shape != PROJECT_SHAPE:
        ref_sd = ref_sd.align(PROJECT_SHAPE)
    r_c = np.asarray(ref_sd.values, dtype=np.float64)
    wavelengths = np.asarray(ref_sd.wavelengths, dtype=np.float64)

    if not is_feasible(r_c, tolerance=config.reflectance_bound_tolerance):
        raise ValueError(
            f"Reference reflectance violates [0, 1] bounds "
            f"(max violation: {max_bound_violation(r_c):.4g})"
        )

    # Build projection and null space
    projection = build_xyz_projection(conceal_condition)
    null_space = compute_null_space(
        projection,
        relative_tolerance=config.null_space_relative_tolerance,
    )

    # Rank-weighted direction sampling: prefer smoother basis vectors
    roughnesses = _basis_roughnesses(null_space.basis)
    weights = _sampling_weights(roughnesses)

    rng = np.random.default_rng(config.seed)
    candidates: list[MetamerCandidate] = []

    ref_spectrum = Spectrum.from_arrays(
        wavelengths,
        r_c,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=reference.provenance,
        validate=False,
    )

    for _ in range(config.n_directions):
        # Sample a direction as a weighted combination of null-space basis vectors
        z = np.zeros(null_space.n_null, dtype=np.float64)
        # Draw basis indices proportional to smoothness weights and combine
        chosen = rng.choice(
            null_space.n_null, size=min(null_space.n_null, 8), replace=False, p=weights
        )
        coeffs = rng.standard_normal(len(chosen))
        for idx, coeff in zip(chosen, coeffs, strict=True):
            z[idx] = coeff

        direction = null_space.basis @ z
        norm = np.linalg.norm(direction)
        if norm < 1e-14:
            continue
        direction /= norm

        interval = feasible_step_interval(r_c, direction)
        if interval is None:
            continue

        alpha_min, alpha_max = interval
        delta = alpha_max - alpha_min
        # inset pulls candidates back from the boundary toward the centre.
        # f=1 steps to the exact boundary; f=0.95 steps to 95% of the boundary.
        inset = (1.0 - config.endpoint_fraction) * delta / 2.0

        for alpha, endpoint in (
            (alpha_min + inset, "minus"),
            (alpha_max - inset, "plus"),
        ):
            r_candidate = r_c + alpha * direction
            violation = max_bound_violation(r_candidate)
            roughness = spectral_roughness(r_candidate)

            prov = Provenance(
                source=f"directional_metamer/{reference.provenance.source}",
                notes=(
                    f"conceal={conceal_condition.name}; "
                    f"endpoint={endpoint}; "
                    f"alpha={alpha:.6g}; "
                    f"seed={config.seed}; "
                    f"endpoint_fraction={config.endpoint_fraction}; "
                    f"null_space_tol={config.null_space_relative_tolerance}"
                ),
            )
            candidate_spectrum = Spectrum.from_arrays(
                wavelengths,
                r_candidate,
                domain=SpectrumDomain.REFLECTANCE,
                provenance=prov,
                validate=False,
            )

            # Verify conceal ΔE₀₀ via direct colour
            result = score_condition(
                ref_spectrum, candidate_spectrum, conceal_condition
            )
            if result.delta_e00 > config.conceal_delta_e00_limit:
                continue

            candidates.append(
                MetamerCandidate(
                    spectrum=candidate_spectrum,
                    source_reference=reference,
                    null_space_coordinates=np.array(z, dtype=np.float64),
                    endpoint=endpoint,
                    alpha=float(alpha),
                    max_bound_violation=violation,
                    roughness=roughness,
                    conceal_delta_e00=result.delta_e00,
                    provenance=prov,
                )
            )

    candidates.sort(key=lambda c: c.roughness)
    return candidates
