"""Tests for Phase C: bounded directional metamer generation."""

from __future__ import annotations

import numpy as np
import pytest

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain
from metamerism.illuminants import get_illuminant
from metamerism.pigments.golden import load_golden_heavy_body_paints
from metamerism.search.feasibility import (
    DEFAULT_REFLECTANCE_BOUND_TOLERANCE,
    is_feasible,
)
from metamerism.search.generate import (
    CandidateGenerationConfig,
    MetamerCandidate,
    generate_directional_metamers,
    spectral_roughness,
)
from metamerism.search.metamer import ViewingCondition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WAVELENGTHS = np.asarray(PROJECT_SHAPE.wavelengths, dtype=np.float64)
N = len(WAVELENGTHS)


@pytest.fixture(scope="module")
def d65_condition() -> ViewingCondition:
    return ViewingCondition(name="D65", illuminant=get_illuminant("D65"))


@pytest.fixture(scope="module")
def default_config() -> CandidateGenerationConfig:
    return CandidateGenerationConfig(seed=42, n_directions=40)


@pytest.fixture(scope="module")
def flat_reference() -> Spectrum:
    return Spectrum.from_arrays(
        WAVELENGTHS,
        np.full(N, 0.5),
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source="flat_0.5"),
    )


@pytest.fixture(scope="module")
def smooth_reference() -> Spectrum:
    values = 0.15 + 0.55 * np.exp(-((WAVELENGTHS - 560.0) ** 2) / (2 * 70.0**2))
    return Spectrum.from_arrays(
        WAVELENGTHS,
        values,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source="smooth_green"),
    )


@pytest.fixture(scope="module")
def golden_reference() -> Spectrum:
    paints = load_golden_heavy_body_paints()
    return paints[0].reflectance


@pytest.fixture(scope="module")
def flat_candidates(
    flat_reference: Spectrum,
    d65_condition: ViewingCondition,
    default_config: CandidateGenerationConfig,
) -> list[MetamerCandidate]:
    return generate_directional_metamers(flat_reference, d65_condition, default_config)


@pytest.fixture(scope="module")
def smooth_candidates(
    smooth_reference: Spectrum,
    d65_condition: ViewingCondition,
    default_config: CandidateGenerationConfig,
) -> list[MetamerCandidate]:
    return generate_directional_metamers(
        smooth_reference, d65_condition, default_config
    )


# ---------------------------------------------------------------------------
# spectral_roughness
# ---------------------------------------------------------------------------


def test_roughness_flat_is_zero():
    values = np.full(N, 0.5)
    assert spectral_roughness(values) == pytest.approx(0.0, abs=1e-14)


def test_roughness_linear_is_zero():
    values = np.linspace(0.1, 0.9, N)
    assert spectral_roughness(values) == pytest.approx(0.0, abs=1e-12)


def test_roughness_smooth_is_small():
    values = 0.5 + 0.3 * np.sin(np.linspace(0, np.pi, N))
    assert spectral_roughness(values) < 0.01


def test_roughness_oscillating_is_large():
    values = 0.5 + 0.4 * np.sin(np.linspace(0, 40 * np.pi, N))
    assert spectral_roughness(values) > 1.0


def test_roughness_is_non_negative():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = rng.uniform(0.0, 1.0, N)
        assert spectral_roughness(v) >= 0.0


# ---------------------------------------------------------------------------
# CandidateGenerationConfig validation
# ---------------------------------------------------------------------------


def test_config_bad_endpoint_fraction():
    with pytest.raises(ValueError, match="endpoint_fraction"):
        CandidateGenerationConfig(seed=0, n_directions=10, endpoint_fraction=0.0)


def test_config_bad_endpoint_fraction_gt1():
    with pytest.raises(ValueError, match="endpoint_fraction"):
        CandidateGenerationConfig(seed=0, n_directions=10, endpoint_fraction=1.1)


def test_config_bad_n_directions():
    with pytest.raises(ValueError, match="n_directions"):
        CandidateGenerationConfig(seed=0, n_directions=0)


def test_config_bad_conceal_limit():
    with pytest.raises(ValueError, match="conceal_delta_e00_limit"):
        CandidateGenerationConfig(seed=0, n_directions=10, conceal_delta_e00_limit=-1.0)


def test_config_round_trip():
    cfg = CandidateGenerationConfig(
        seed=7, n_directions=20, endpoint_fraction=0.8, conceal_delta_e00_limit=2.0
    )
    d = cfg.to_dict()
    restored = CandidateGenerationConfig.from_dict(d)
    assert restored.seed == cfg.seed
    assert restored.n_directions == cfg.n_directions
    assert restored.endpoint_fraction == pytest.approx(cfg.endpoint_fraction)
    assert restored.conceal_delta_e00_limit == pytest.approx(
        cfg.conceal_delta_e00_limit
    )


# ---------------------------------------------------------------------------
# generate_directional_metamers: basic contracts
# ---------------------------------------------------------------------------


def test_generates_multiple_candidates(flat_candidates: list[MetamerCandidate]):
    assert len(flat_candidates) >= 2


def test_generates_both_endpoints(flat_candidates: list[MetamerCandidate]):
    endpoints = {c.endpoint for c in flat_candidates}
    assert "minus" in endpoints
    assert "plus" in endpoints


def test_candidates_have_reflectance_domain(flat_candidates: list[MetamerCandidate]):
    for c in flat_candidates:
        assert c.spectrum.domain is SpectrumDomain.REFLECTANCE


def test_candidates_on_project_grid(flat_candidates: list[MetamerCandidate]):
    for c in flat_candidates:
        assert c.spectrum.is_on_canonical_grid()


def test_candidates_within_bounds(flat_candidates: list[MetamerCandidate]):
    for c in flat_candidates:
        assert is_feasible(
            c.spectrum.values, tolerance=DEFAULT_REFLECTANCE_BOUND_TOLERANCE
        ), f"candidate violates bounds: max violation={c.max_bound_violation:.4g}"


def test_candidates_satisfy_conceal_threshold(
    flat_candidates: list[MetamerCandidate],
    default_config: CandidateGenerationConfig,
):
    for c in flat_candidates:
        assert c.conceal_delta_e00 <= default_config.conceal_delta_e00_limit


def test_candidates_distinct_from_reference(
    flat_candidates: list[MetamerCandidate],
    flat_reference: Spectrum,
):
    for c in flat_candidates:
        assert not np.allclose(c.spectrum.values, flat_reference.values)


def test_candidates_sorted_by_roughness(flat_candidates: list[MetamerCandidate]):
    roughnesses = [c.roughness for c in flat_candidates]
    assert roughnesses == sorted(roughnesses)


def test_roughness_stored_matches_computed(flat_candidates: list[MetamerCandidate]):
    for c in flat_candidates:
        computed = spectral_roughness(c.spectrum.values)
        assert c.roughness == pytest.approx(computed, rel=1e-10)


def test_max_bound_violation_stored(flat_candidates: list[MetamerCandidate]):
    for c in flat_candidates:
        assert c.max_bound_violation >= 0.0
        assert c.max_bound_violation <= DEFAULT_REFLECTANCE_BOUND_TOLERANCE


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_reproducible_with_same_seed(
    flat_reference: Spectrum,
    d65_condition: ViewingCondition,
    default_config: CandidateGenerationConfig,
):
    run1 = generate_directional_metamers(flat_reference, d65_condition, default_config)
    run2 = generate_directional_metamers(flat_reference, d65_condition, default_config)
    assert len(run1) == len(run2)
    for c1, c2 in zip(run1, run2, strict=True):
        assert np.allclose(c1.spectrum.values, c2.spectrum.values)


def test_different_seeds_give_different_results(
    flat_reference: Spectrum,
    d65_condition: ViewingCondition,
):
    cfg1 = CandidateGenerationConfig(seed=1, n_directions=20)
    cfg2 = CandidateGenerationConfig(seed=2, n_directions=20)
    run1 = generate_directional_metamers(flat_reference, d65_condition, cfg1)
    run2 = generate_directional_metamers(flat_reference, d65_condition, cfg2)
    # At least one candidate should differ
    if run1 and run2:
        assert not all(
            np.allclose(c1.spectrum.values, c2.spectrum.values)
            for c1, c2 in zip(run1, run2, strict=False)
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_candidate_provenance_records_conceal_condition(
    flat_candidates: list[MetamerCandidate],
    d65_condition: ViewingCondition,
):
    for c in flat_candidates:
        assert d65_condition.name in (c.provenance.notes or "")


def test_candidate_provenance_records_seed(
    flat_candidates: list[MetamerCandidate],
    default_config: CandidateGenerationConfig,
):
    for c in flat_candidates:
        assert str(default_config.seed) in (c.provenance.notes or "")


def test_candidate_source_reference_preserved(
    flat_candidates: list[MetamerCandidate],
    flat_reference: Spectrum,
):
    for c in flat_candidates:
        assert np.allclose(c.source_reference.values, flat_reference.values)


# ---------------------------------------------------------------------------
# Smooth reference
# ---------------------------------------------------------------------------


def test_smooth_reference_generates_candidates(
    smooth_candidates: list[MetamerCandidate],
):
    assert len(smooth_candidates) >= 2


def test_smooth_reference_candidates_within_bounds(
    smooth_candidates: list[MetamerCandidate],
):
    for c in smooth_candidates:
        assert is_feasible(
            c.spectrum.values, tolerance=DEFAULT_REFLECTANCE_BOUND_TOLERANCE
        )


def test_smooth_reference_conceal_verified(
    smooth_candidates: list[MetamerCandidate],
    default_config: CandidateGenerationConfig,
):
    for c in smooth_candidates:
        assert c.conceal_delta_e00 <= default_config.conceal_delta_e00_limit


# ---------------------------------------------------------------------------
# Golden paint reference
# ---------------------------------------------------------------------------


def test_golden_reference_generates_candidates(
    golden_reference: Spectrum,
    d65_condition: ViewingCondition,
):
    cfg = CandidateGenerationConfig(seed=0, n_directions=30)
    candidates = generate_directional_metamers(golden_reference, d65_condition, cfg)
    assert len(candidates) >= 1
    for c in candidates:
        assert is_feasible(
            c.spectrum.values, tolerance=DEFAULT_REFLECTANCE_BOUND_TOLERANCE
        )
        assert c.conceal_delta_e00 <= cfg.conceal_delta_e00_limit


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_rejects_non_reflectance_reference(d65_condition: ViewingCondition):
    ill = get_illuminant("D65")
    cfg = CandidateGenerationConfig(seed=0, n_directions=5)
    with pytest.raises(ValueError, match="REFLECTANCE"):
        generate_directional_metamers(ill, d65_condition, cfg)


def test_endpoint_fraction_affects_alpha(
    flat_reference: Spectrum,
    d65_condition: ViewingCondition,
):
    """Larger endpoint_fraction → larger |alpha| for corresponding endpoints."""
    # Use the same seed so the same directions are sampled; choose a tight
    # conceal limit so both runs survive comparably, and compare |alpha|.
    cfg_large = CandidateGenerationConfig(
        seed=42, n_directions=20, endpoint_fraction=0.9, conceal_delta_e00_limit=5.0
    )
    cfg_small = CandidateGenerationConfig(
        seed=42, n_directions=20, endpoint_fraction=0.1, conceal_delta_e00_limit=5.0
    )
    large = generate_directional_metamers(flat_reference, d65_condition, cfg_large)
    small = generate_directional_metamers(flat_reference, d65_condition, cfg_small)

    if large and small:
        avg_alpha_large = np.mean([abs(c.alpha) for c in large])
        avg_alpha_small = np.mean([abs(c.alpha) for c in small])
        assert avg_alpha_large > avg_alpha_small


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_candidate_to_dict(flat_candidates: list[MetamerCandidate]):
    c = flat_candidates[0]
    d = c.to_dict()
    assert "spectrum_values" in d
    assert "roughness" in d
    assert "conceal_delta_e00" in d
    assert "endpoint" in d
    vals = np.array(d["spectrum_values"])
    assert np.allclose(vals, c.spectrum.values)
