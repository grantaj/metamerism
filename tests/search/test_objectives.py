"""Tests for Phase D: smoothness metrics, diagnostics, and practical ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain
from metamerism.illuminants import get_illuminant
from metamerism.search.diagnostics import (
    PoolDiagnostics,
    candidate_report,
    check_projection,
    summarise_pool,
)
from metamerism.search.generate import (
    CandidateGenerationConfig,
    MetamerCandidate,
    generate_directional_metamers,
    spectral_roughness,
)
from metamerism.search.metamer import ViewingCondition
from metamerism.search.objectives import (
    CandidateScore,
    RankingConfig,
    boundary_saturation,
    passes_conceal_threshold,
    passes_roughness_limit,
    rank_scored_candidates,
    score_candidate,
)
from metamerism.search.projection import build_xyz_projection, compute_null_space

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WAVELENGTHS = np.asarray(PROJECT_SHAPE.wavelengths, dtype=np.float64)
N = len(WAVELENGTHS)


@pytest.fixture(scope="module")
def d65_condition() -> ViewingCondition:
    return ViewingCondition(name="D65", illuminant=get_illuminant("D65"))


@pytest.fixture(scope="module")
def a_condition() -> ViewingCondition:
    return ViewingCondition(name="A", illuminant=get_illuminant("A"))


@pytest.fixture(scope="module")
def fl11_condition() -> ViewingCondition:
    return ViewingCondition(name="FL11", illuminant=get_illuminant("FL11"))


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
def candidates(
    flat_reference: Spectrum,
    d65_condition: ViewingCondition,
) -> list[MetamerCandidate]:
    cfg = CandidateGenerationConfig(seed=42, n_directions=30)
    return generate_directional_metamers(flat_reference, d65_condition, cfg)


@pytest.fixture(scope="module")
def scored(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
) -> list[tuple[MetamerCandidate, CandidateScore]]:
    return [(c, score_candidate(c, [a_condition])) for c in candidates]


# ---------------------------------------------------------------------------
# boundary_saturation
# ---------------------------------------------------------------------------


def test_saturation_zero_for_interior():
    values = np.full(N, 0.5)
    assert boundary_saturation(values) == pytest.approx(0.0)


def test_saturation_one_all_at_zero():
    values = np.zeros(N)
    assert boundary_saturation(values) == pytest.approx(1.0)


def test_saturation_one_all_at_one():
    values = np.ones(N)
    assert boundary_saturation(values) == pytest.approx(1.0)


def test_saturation_half_at_boundary():
    values = np.concatenate([np.zeros(N // 2), np.full(N - N // 2, 0.5)])
    sat = boundary_saturation(values)
    assert 0.4 < sat < 0.6


def test_saturation_custom_near():
    values = np.full(N, 0.5)
    # With near=0.6 every sample at 0.5 is within 0.6 of the lower bound
    assert boundary_saturation(values, near=0.6) == pytest.approx(1.0)


def test_saturation_in_zero_one():
    rng = np.random.default_rng(0)
    values = rng.uniform(0.0, 1.0, N)
    sat = boundary_saturation(values)
    assert 0.0 <= sat <= 1.0


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


def test_score_candidate_returns_candidatescore(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    c = candidates[0]
    s = score_candidate(c, [a_condition])
    assert isinstance(s, CandidateScore)


def test_score_candidate_conceal_de_matches_stored(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    """Stored conceal ΔE₀₀ should match what score_candidate reports."""
    for c in candidates[:5]:
        s = score_candidate(c, [a_condition])
        assert s.conceal_delta_e00 == pytest.approx(c.conceal_delta_e00, abs=1e-10)


def test_score_candidate_reveal_positive(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    for c in candidates[:5]:
        s = score_candidate(c, [a_condition])
        assert s.minimum_reveal_delta_e00 >= 0.0
        assert all(de >= 0.0 for de in s.reveal_delta_e00s)


def test_score_candidate_multiple_reveal(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
):
    c = candidates[0]
    s = score_candidate(c, [a_condition, fl11_condition])
    assert len(s.reveal_delta_e00s) == 2
    assert s.minimum_reveal_delta_e00 == pytest.approx(
        min(s.reveal_delta_e00s), abs=1e-12
    )


def test_score_candidate_weighted_reveal(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
):
    c = candidates[0]
    weights = [2.0, 1.0]
    s = score_candidate(c, [a_condition, fl11_condition], reveal_weights=weights)
    expected_weighted = (
        weights[0] * s.reveal_delta_e00s[0] + weights[1] * s.reveal_delta_e00s[1]
    )
    assert s.weighted_reveal_score == pytest.approx(expected_weighted, rel=1e-10)


def test_score_candidate_roughness_matches(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    c = candidates[0]
    s = score_candidate(c, [a_condition])
    assert s.roughness == pytest.approx(
        spectral_roughness(c.spectrum.values), rel=1e-10
    )


def test_score_candidate_saturation_in_range(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    for c in candidates[:5]:
        s = score_candidate(c, [a_condition])
        assert 0.0 <= s.boundary_saturation <= 1.0


def test_score_candidate_spectral_dist_positive(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    for c in candidates[:5]:
        s = score_candidate(c, [a_condition])
        assert s.spectral_dist > 0.0


def test_score_candidate_requires_reveal(
    candidates: list[MetamerCandidate],
):
    with pytest.raises(ValueError, match="reveal condition"):
        score_candidate(candidates[0], [])


def test_score_candidate_weight_mismatch(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    with pytest.raises(ValueError, match="reveal_weights"):
        score_candidate(candidates[0], [a_condition], reveal_weights=[1.0, 2.0])


def test_score_candidate_round_trip(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
):
    s = score_candidate(candidates[0], [a_condition])
    d = s.to_dict()
    restored = CandidateScore.from_dict(d)
    assert restored.conceal_delta_e00 == pytest.approx(s.conceal_delta_e00)
    assert restored.minimum_reveal_delta_e00 == pytest.approx(
        s.minimum_reveal_delta_e00
    )
    assert restored.roughness == pytest.approx(s.roughness)


# ---------------------------------------------------------------------------
# passes_conceal_threshold / passes_roughness_limit
# ---------------------------------------------------------------------------


def _make_score(conceal: float, roughness: float = 0.5) -> CandidateScore:
    return CandidateScore(
        conceal_delta_e00=conceal,
        reveal_delta_e00s=(5.0,),
        minimum_reveal_delta_e00=5.0,
        weighted_reveal_score=5.0,
        roughness=roughness,
        boundary_saturation=0.0,
        spectral_dist=0.1,
    )


def test_passes_conceal_threshold_candidate_score():
    assert passes_conceal_threshold(_make_score(0.5), limit=1.0)
    assert not passes_conceal_threshold(_make_score(1.5), limit=1.0)


def test_passes_roughness_limit_none_always_passes():
    assert passes_roughness_limit(_make_score(0.0, roughness=1e6), limit=None)


def test_passes_roughness_limit_enforced():
    assert passes_roughness_limit(_make_score(0.0, roughness=1.0), limit=2.0)
    assert not passes_roughness_limit(_make_score(0.0, roughness=3.0), limit=2.0)


# ---------------------------------------------------------------------------
# RankingConfig validation
# ---------------------------------------------------------------------------


def test_ranking_config_bad_primary():
    with pytest.raises(ValueError, match="primary"):
        RankingConfig(primary="bad_key")


def test_ranking_config_round_trip():
    cfg = RankingConfig(
        conceal_limit=0.5,
        roughness_limit=10.0,
        roughness_weight=0.1,
        saturation_weight=0.2,
        primary="weighted_reveal",
    )
    d = cfg.to_dict()
    restored = RankingConfig.from_dict(d)
    assert restored.conceal_limit == pytest.approx(cfg.conceal_limit)
    assert restored.roughness_limit == pytest.approx(cfg.roughness_limit)
    assert restored.primary == cfg.primary


# ---------------------------------------------------------------------------
# rank_scored_candidates
# ---------------------------------------------------------------------------


def test_rank_scored_candidates_filters_conceal(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    cfg = RankingConfig(conceal_limit=0.0)  # reject everything
    ranked = rank_scored_candidates(scored, cfg)
    assert ranked == []


def test_rank_scored_candidates_passes_all_below_limit(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    cfg = RankingConfig(conceal_limit=100.0)
    ranked = rank_scored_candidates(scored, cfg)
    assert len(ranked) == len(scored)


def test_rank_scored_candidates_sorted_descending_reveal(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    ranked = rank_scored_candidates(scored)
    reveal_scores = [s.minimum_reveal_delta_e00 for _, s in ranked]
    assert reveal_scores == sorted(reveal_scores, reverse=True)


def test_rank_scored_candidates_roughness_filter(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    """A roughness hard limit should drop the roughest candidates."""
    roughnesses = sorted(s.roughness for _, s in scored)
    median_roughness = roughnesses[len(roughnesses) // 2]

    cfg_tight = RankingConfig(roughness_limit=median_roughness)
    ranked_tight = rank_scored_candidates(scored, cfg_tight)

    cfg_loose = RankingConfig(roughness_limit=None)
    ranked_loose = rank_scored_candidates(scored, cfg_loose)

    assert len(ranked_tight) < len(ranked_loose)
    for _, s in ranked_tight:
        assert s.roughness <= median_roughness


def test_rank_scored_candidates_roughness_penalty_changes_order(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    """Adding a roughness penalty should change ranking when roughness varies."""
    cfg_no_penalty = RankingConfig(roughness_weight=0.0)
    cfg_with_penalty = RankingConfig(roughness_weight=1000.0)

    ranked_no = rank_scored_candidates(scored, cfg_no_penalty)
    ranked_with = rank_scored_candidates(scored, cfg_with_penalty)

    if len(ranked_no) > 1 and len(ranked_with) > 1:
        # At least one position should differ
        order_no = [id(c) for c, _ in ranked_no]
        order_with = [id(c) for c, _ in ranked_with]
        assert order_no != order_with


def test_rank_scored_candidates_weighted_primary(
    candidates: list[MetamerCandidate],
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
):
    scored_multi = [
        (c, score_candidate(c, [a_condition, fl11_condition])) for c in candidates
    ]
    cfg = RankingConfig(primary="weighted_reveal")
    ranked = rank_scored_candidates(scored_multi, cfg)
    scores = [s.weighted_reveal_score for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_scored_candidates_empty():
    assert rank_scored_candidates([]) == []


# ---------------------------------------------------------------------------
# Roughness discriminates smooth vs rough
# ---------------------------------------------------------------------------


def test_rougher_candidate_ranked_lower_with_penalty(
    flat_reference: Spectrum,
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
):
    """A candidate with higher roughness should rank lower when penalty is high."""
    cfg_gen = CandidateGenerationConfig(seed=0, n_directions=60)
    cands = generate_directional_metamers(flat_reference, d65_condition, cfg_gen)
    scored_pairs = [(c, score_candidate(c, [a_condition])) for c in cands]

    if len(scored_pairs) < 2:
        pytest.skip("Need at least 2 candidates")

    roughnesses = [s.roughness for _, s in scored_pairs]
    if max(roughnesses) == min(roughnesses):
        pytest.skip("All candidates have identical roughness")

    cfg_rank = RankingConfig(roughness_weight=1e6)
    ranked = rank_scored_candidates(scored_pairs, cfg_rank)

    ranked_roughnesses = [s.roughness for _, s in ranked]
    # The top-ranked candidate should not be the roughest
    roughest = max(roughnesses)
    assert ranked_roughnesses[0] < roughest


# ---------------------------------------------------------------------------
# Diagnostics: check_projection
# ---------------------------------------------------------------------------


def test_check_projection_rank_3(d65_condition: ViewingCondition):
    proj = build_xyz_projection(d65_condition)
    ns = compute_null_space(proj)
    diag = check_projection(proj, ns)
    assert diag.rank == 3
    assert diag.null_space_dimension == N - 3


def test_check_projection_residual_small(d65_condition: ViewingCondition):
    proj = build_xyz_projection(d65_condition)
    ns = compute_null_space(proj)
    diag = check_projection(proj, ns)
    assert diag.null_space_residual_max < 1e-8


def test_check_projection_condition_number_finite(d65_condition: ViewingCondition):
    proj = build_xyz_projection(d65_condition)
    ns = compute_null_space(proj)
    diag = check_projection(proj, ns)
    assert math.isfinite(diag.condition_number)
    assert diag.condition_number > 1.0


def test_check_projection_no_warnings_healthy(d65_condition: ViewingCondition):
    proj = build_xyz_projection(d65_condition)
    ns = compute_null_space(proj)
    diag = check_projection(proj, ns)
    assert len(diag.warnings) == 0


def test_check_projection_to_dict(d65_condition: ViewingCondition):
    proj = build_xyz_projection(d65_condition)
    ns = compute_null_space(proj)
    diag = check_projection(proj, ns)
    d = diag.to_dict()
    assert d["rank"] == 3
    assert isinstance(d["warnings"], list)


# ---------------------------------------------------------------------------
# Diagnostics: summarise_pool
# ---------------------------------------------------------------------------


def test_summarise_pool_empty():
    diag = summarise_pool([])
    assert diag.n_total == 0
    assert math.isnan(diag.roughness_median)


def test_summarise_pool_counts(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    diag = summarise_pool(scored)
    assert diag.n_total == len(scored)
    assert 0 <= diag.n_passing_conceal <= diag.n_total


def test_summarise_pool_roughness_ordering(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    diag = summarise_pool(scored)
    assert diag.roughness_min <= diag.roughness_median
    assert diag.roughness_median <= diag.roughness_p95
    assert diag.roughness_p95 <= diag.roughness_max


def test_summarise_pool_saturation_range(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    diag = summarise_pool(scored)
    assert 0.0 <= diag.saturation_mean <= 1.0
    assert 0.0 <= diag.saturation_max <= 1.0


def test_summarise_pool_to_dict(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    diag = summarise_pool(scored)
    d = diag.to_dict()
    restored = PoolDiagnostics.from_dict(d)
    assert restored.n_total == diag.n_total
    assert restored.roughness_p95 == pytest.approx(diag.roughness_p95)


# ---------------------------------------------------------------------------
# candidate_report
# ---------------------------------------------------------------------------


def test_candidate_report_contains_key_fields(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    c, s = scored[0]
    report = candidate_report(c, s)
    assert "Conceal" in report
    assert "Reveal" in report
    assert "Roughness" in report
    assert "Saturation" in report


def test_candidate_report_is_string(
    scored: list[tuple[MetamerCandidate, CandidateScore]],
):
    c, s = scored[0]
    assert isinstance(candidate_report(c, s), str)
