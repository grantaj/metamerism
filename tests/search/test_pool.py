"""Tests for Phase E: Pareto filtering and diversity-based pool selection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from metamerism.core import Provenance, Spectrum, SpectrumDomain
from metamerism.illuminants import get_illuminant
from metamerism.search.generate import (
    CandidateGenerationConfig,
    MetamerCandidate,
)
from metamerism.search.metamer import ViewingCondition
from metamerism.search.objectives import (
    CandidateScore,
    RankingConfig,
)
from metamerism.search.pool import (
    SearchConfig,
    SearchResult,
    diversity_select,
    pareto_front,
    run_search,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


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
def reference() -> Spectrum:
    wl = np.linspace(380.0, 780.0, 81)
    vals = 0.3 + 0.3 * np.sin(np.linspace(0, np.pi, 81))
    return Spectrum.from_arrays(
        wl,
        vals,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source="test_smooth_reference"),
    )


@pytest.fixture(scope="module")
def small_gen_config() -> CandidateGenerationConfig:
    return CandidateGenerationConfig(seed=42, n_directions=30)


@pytest.fixture(scope="module")
def small_ranking() -> RankingConfig:
    return RankingConfig(conceal_limit=1.0)


@pytest.fixture(scope="module")
def search_config(
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
    small_gen_config: CandidateGenerationConfig,
    small_ranking: RankingConfig,
) -> SearchConfig:
    return SearchConfig(
        conceal_condition=d65_condition,
        reveal_conditions=(a_condition, fl11_condition),
        generation=small_gen_config,
        ranking=small_ranking,
        n_diverse=5,
    )


@pytest.fixture(scope="module")
def search_result(
    reference: Spectrum,
    search_config: SearchConfig,
) -> SearchResult:
    return run_search(reference, search_config)


# ---------------------------------------------------------------------------
# Helpers for unit-level Pareto / diversity tests
# ---------------------------------------------------------------------------


def _make_score(
    *,
    weighted_reveal: float = 5.0,
    min_reveal: float = 4.0,
    roughness: float = 0.01,
    saturation: float = 0.05,
) -> CandidateScore:
    return CandidateScore(
        conceal_delta_e00=0.3,
        reveal_delta_e00s=(4.0,),
        minimum_reveal_delta_e00=min_reveal,
        weighted_reveal_score=weighted_reveal,
        roughness=roughness,
        boundary_saturation=saturation,
        spectral_dist=0.1,
    )


def _make_candidate(values: np.ndarray, reference: Spectrum) -> MetamerCandidate:
    """Construct a minimal MetamerCandidate with given spectrum values."""
    wl = np.linspace(380.0, 780.0, 81)
    prov = Provenance(source="test_candidate")
    spectrum = Spectrum.from_arrays(
        wl, values, domain=SpectrumDomain.REFLECTANCE, provenance=prov, validate=False
    )
    return MetamerCandidate(
        spectrum=spectrum,
        source_reference=reference,
        null_space_coordinates=np.zeros(78),
        endpoint="plus",
        alpha=1.0,
        max_bound_violation=0.0,
        roughness=0.01,
        conceal_delta_e00=0.3,
        provenance=prov,
    )


# ---------------------------------------------------------------------------
# pareto_front
# ---------------------------------------------------------------------------


def test_pareto_front_empty_input() -> None:
    assert pareto_front([]) == []


def test_pareto_front_single_candidate(reference: Spectrum) -> None:
    c = _make_candidate(np.full(81, 0.5), reference)
    s = _make_score()
    result = pareto_front([(c, s)])
    assert len(result) == 1


def test_pareto_front_clear_domination(reference: Spectrum) -> None:
    c_good = _make_candidate(np.full(81, 0.5), reference)
    c_bad = _make_candidate(np.full(81, 0.4), reference)
    s_good = _make_score(weighted_reveal=8.0, min_reveal=7.0, roughness=0.005)
    s_bad = _make_score(weighted_reveal=3.0, min_reveal=2.0, roughness=0.02)
    front = pareto_front([(c_good, s_good), (c_bad, s_bad)])
    assert len(front) == 1
    assert front[0][0] is c_good


def test_pareto_front_dominated_does_not_dominate_dominator(
    reference: Spectrum,
) -> None:
    c_a = _make_candidate(np.full(81, 0.5), reference)
    c_b = _make_candidate(np.full(81, 0.4), reference)
    s_a = _make_score(weighted_reveal=8.0, min_reveal=7.0, roughness=0.005)
    s_b = _make_score(weighted_reveal=3.0, min_reveal=2.0, roughness=0.02)
    front = pareto_front([(c_b, s_b), (c_a, s_a)])
    assert len(front) == 1
    assert front[0][0] is c_a


def test_pareto_front_incomparable_both_in_front(reference: Spectrum) -> None:
    c_a = _make_candidate(np.full(81, 0.5), reference)
    c_b = _make_candidate(np.full(81, 0.4), reference)
    # a: high reveal, rough; b: low reveal, smooth — neither dominates
    s_a = _make_score(weighted_reveal=9.0, min_reveal=8.0, roughness=0.1)
    s_b = _make_score(weighted_reveal=3.0, min_reveal=2.0, roughness=0.001)
    front = pareto_front([(c_a, s_a), (c_b, s_b)])
    assert len(front) == 2


def test_pareto_front_chain(reference: Spectrum) -> None:
    # A dominates B dominates C → only A on front
    c_a = _make_candidate(np.full(81, 0.5), reference)
    c_b = _make_candidate(np.full(81, 0.4), reference)
    c_c = _make_candidate(np.full(81, 0.3), reference)
    s_a = _make_score(weighted_reveal=9.0, min_reveal=8.0, roughness=0.001)
    s_b = _make_score(weighted_reveal=5.0, min_reveal=4.0, roughness=0.01)
    s_c = _make_score(weighted_reveal=2.0, min_reveal=1.0, roughness=0.05)
    front = pareto_front([(c_a, s_a), (c_b, s_b), (c_c, s_c)])
    assert len(front) == 1
    assert front[0][0] is c_a


def test_pareto_front_preserves_relative_order(reference: Spectrum) -> None:
    # Three incomparable candidates — all on front; order should be preserved
    specs = [np.full(81, v) for v in [0.3, 0.5, 0.7]]
    scores = [
        _make_score(weighted_reveal=8.0, min_reveal=7.0, roughness=0.05),
        _make_score(weighted_reveal=5.0, min_reveal=5.0, roughness=0.005),
        _make_score(weighted_reveal=2.0, min_reveal=1.5, roughness=0.001),
    ]
    pairs = [
        (_make_candidate(s, reference), sc)
        for s, sc in zip(specs, scores, strict=True)
    ]
    front = pareto_front(pairs)
    assert len(front) == 3
    for original, found in zip(pairs, front, strict=True):
        assert original[0] is found[0]


def test_pareto_front_saturation_breaks_tie(reference: Spectrum) -> None:
    c_a = _make_candidate(np.full(81, 0.5), reference)
    c_b = _make_candidate(np.full(81, 0.4), reference)
    # Equal on reveal and roughness; a has lower saturation → a dominates
    s_a = _make_score(saturation=0.02)
    s_b = _make_score(saturation=0.10)
    front = pareto_front([(c_a, s_a), (c_b, s_b)])
    assert len(front) == 1
    assert front[0][0] is c_a


# ---------------------------------------------------------------------------
# diversity_select
# ---------------------------------------------------------------------------


def test_diversity_select_empty(reference: Spectrum) -> None:
    assert diversity_select([], 5) == []


def test_diversity_select_n_zero(reference: Spectrum) -> None:
    c = _make_candidate(np.full(81, 0.5), reference)
    s = _make_score()
    assert diversity_select([(c, s)], 0) == []


def test_diversity_select_n_larger_than_pool(reference: Spectrum) -> None:
    pairs = [
        (_make_candidate(np.full(81, v), reference), _make_score())
        for v in [0.3, 0.5, 0.7]
    ]
    result = diversity_select(pairs, 10)
    assert len(result) == 3


def test_diversity_select_returns_n_candidates(reference: Spectrum) -> None:
    pairs = [
        (_make_candidate(np.full(81, v), reference), _make_score())
        for v in np.linspace(0.1, 0.9, 20)
    ]
    result = diversity_select(pairs, 5)
    assert len(result) == 5


def test_diversity_select_first_is_first_input(reference: Spectrum) -> None:
    pairs = [
        (_make_candidate(np.full(81, v), reference), _make_score())
        for v in [0.5, 0.1, 0.9]
    ]
    result = diversity_select(pairs, 2)
    assert result[0][0] is pairs[0][0]


def test_diversity_select_spreads_spectra(reference: Spectrum) -> None:
    # Pool: 10 spectra clustered near 0.3 and 10 near 0.7
    pairs = [
        (_make_candidate(np.full(81, 0.3) + rng_val, reference), _make_score())
        for rng_val in np.random.default_rng(0).uniform(-0.01, 0.01, 10)
    ] + [
        (_make_candidate(np.full(81, 0.7) + rng_val, reference), _make_score())
        for rng_val in np.random.default_rng(1).uniform(-0.01, 0.01, 10)
    ]

    result = diversity_select(pairs, 4)
    values = np.stack([c.spectrum.values for c, _ in result])
    # Selected should span a wider range than if we picked 4 near-identical ones
    span = float(np.max(values) - np.min(values))
    assert span > 0.2


def test_diversity_select_no_duplicates(reference: Spectrum) -> None:
    pairs = [
        (_make_candidate(np.full(81, v), reference), _make_score())
        for v in np.linspace(0.1, 0.9, 15)
    ]
    result = diversity_select(pairs, 8)
    ids = [id(c) for c, _ in result]
    assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# SearchConfig validation and round-trip
# ---------------------------------------------------------------------------


def test_search_config_rejects_empty_reveal(
    d65_condition: ViewingCondition,
    small_gen_config: CandidateGenerationConfig,
    small_ranking: RankingConfig,
) -> None:
    with pytest.raises(ValueError, match="reveal condition"):
        SearchConfig(
            conceal_condition=d65_condition,
            reveal_conditions=(),
            generation=small_gen_config,
            ranking=small_ranking,
        )


def test_search_config_rejects_weight_mismatch(
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    small_gen_config: CandidateGenerationConfig,
    small_ranking: RankingConfig,
) -> None:
    with pytest.raises(ValueError, match="length"):
        SearchConfig(
            conceal_condition=d65_condition,
            reveal_conditions=(a_condition,),
            generation=small_gen_config,
            ranking=small_ranking,
            reveal_weights=(1.0, 2.0),  # wrong length
        )


def test_search_config_rejects_nonpositive_weight(
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    small_gen_config: CandidateGenerationConfig,
    small_ranking: RankingConfig,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        SearchConfig(
            conceal_condition=d65_condition,
            reveal_conditions=(a_condition,),
            generation=small_gen_config,
            ranking=small_ranking,
            reveal_weights=(0.0,),
        )


def test_search_config_rejects_n_diverse_zero(
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    small_gen_config: CandidateGenerationConfig,
    small_ranking: RankingConfig,
) -> None:
    with pytest.raises(ValueError, match="n_diverse"):
        SearchConfig(
            conceal_condition=d65_condition,
            reveal_conditions=(a_condition,),
            generation=small_gen_config,
            ranking=small_ranking,
            n_diverse=0,
        )


def test_search_config_round_trip_no_weights(
    search_config: SearchConfig,
) -> None:
    d = search_config.to_dict()
    restored = SearchConfig.from_dict(d)
    assert restored.conceal_condition.name == search_config.conceal_condition.name
    assert len(restored.reveal_conditions) == len(search_config.reveal_conditions)
    assert restored.generation.seed == search_config.generation.seed
    assert restored.n_diverse == search_config.n_diverse
    assert restored.reveal_weights is None


def test_search_config_round_trip_with_weights(
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
    small_gen_config: CandidateGenerationConfig,
    small_ranking: RankingConfig,
) -> None:
    cfg = SearchConfig(
        conceal_condition=d65_condition,
        reveal_conditions=(a_condition, fl11_condition),
        generation=small_gen_config,
        ranking=small_ranking,
        reveal_weights=(2.0, 1.0),
    )
    d = cfg.to_dict()
    restored = SearchConfig.from_dict(d)
    assert restored.reveal_weights == pytest.approx((2.0, 1.0))


def test_search_config_to_dict_is_json_serialisable(
    search_config: SearchConfig,
) -> None:
    d = search_config.to_dict()
    json_str = json.dumps(d)
    assert isinstance(json_str, str)


def test_search_config_illuminant_round_trip_preserves_values(
    search_config: SearchConfig,
) -> None:
    d = search_config.to_dict()
    restored = SearchConfig.from_dict(d)
    orig_vals = search_config.conceal_condition.illuminant.values
    rest_vals = restored.conceal_condition.illuminant.values
    assert np.allclose(orig_vals, rest_vals, atol=1e-12)


# ---------------------------------------------------------------------------
# run_search integration
# ---------------------------------------------------------------------------


def test_run_search_returns_search_result(
    reference: Spectrum,
    search_config: SearchConfig,
    search_result: SearchResult,
) -> None:
    assert isinstance(search_result, SearchResult)


def test_run_search_is_reproducible(
    reference: Spectrum,
    search_config: SearchConfig,
) -> None:
    r1 = run_search(reference, search_config)
    r2 = run_search(reference, search_config)
    assert len(r1.all_scored) == len(r2.all_scored)
    assert len(r1.selected) == len(r2.selected)
    if r1.selected and r2.selected:
        v1 = r1.selected[0][0].spectrum.values
        v2 = r2.selected[0][0].spectrum.values
        assert np.allclose(v1, v2, atol=1e-12)


def test_run_search_pareto_subset_of_ranked(
    search_result: SearchResult,
) -> None:
    ranked_ids = {id(c) for c, _ in search_result.all_scored}
    for c, _ in search_result.pareto_scored:
        assert id(c) in ranked_ids


def test_run_search_selected_subset_of_pareto(
    search_result: SearchResult,
) -> None:
    pareto_ids = {id(c) for c, _ in search_result.pareto_scored}
    for c, _ in search_result.selected:
        assert id(c) in pareto_ids


def test_run_search_selected_count_at_most_n_diverse(
    search_result: SearchResult,
    search_config: SearchConfig,
) -> None:
    assert len(search_result.selected) <= search_config.n_diverse


def test_run_search_candidates_meet_conceal_threshold(
    search_result: SearchResult,
    search_config: SearchConfig,
) -> None:
    limit = search_config.ranking.conceal_limit
    for c, _ in search_result.pareto_scored:
        assert c.conceal_delta_e00 <= limit


def test_run_search_pareto_is_non_dominated(
    search_result: SearchResult,
) -> None:
    pareto = search_result.pareto_scored
    if len(pareto) < 2:
        return
    from metamerism.search.pool import _dominates

    for i, (_, si) in enumerate(pareto):
        for j, (_, sj) in enumerate(pareto):
            if i != j:
                assert not _dominates(sj, si), (
                    f"Candidate {j} dominates candidate {i} in the Pareto front"
                )


def test_run_search_multiple_reveal_conditions(
    reference: Spectrum,
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
) -> None:
    cfg = SearchConfig(
        conceal_condition=d65_condition,
        reveal_conditions=(a_condition, fl11_condition),
        generation=CandidateGenerationConfig(seed=7, n_directions=20),
        ranking=RankingConfig(conceal_limit=1.0),
        n_diverse=3,
    )
    result = run_search(reference, cfg)
    for _, score in result.all_scored:
        assert len(score.reveal_delta_e00s) == 2


def test_run_search_with_reveal_weights(
    reference: Spectrum,
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
    fl11_condition: ViewingCondition,
) -> None:
    cfg = SearchConfig(
        conceal_condition=d65_condition,
        reveal_conditions=(a_condition, fl11_condition),
        generation=CandidateGenerationConfig(seed=7, n_directions=20),
        ranking=RankingConfig(
            conceal_limit=1.0, primary="weighted_reveal"
        ),
        reveal_weights=(2.0, 1.0),
        n_diverse=3,
    )
    result = run_search(reference, cfg)
    for _, score in result.all_scored:
        expected = 2.0 * score.reveal_delta_e00s[0] + 1.0 * score.reveal_delta_e00s[1]
        assert score.weighted_reveal_score == pytest.approx(expected, rel=1e-10)


def test_run_search_to_dict_is_json_serialisable(
    search_result: SearchResult,
) -> None:
    d = search_result.to_dict()
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert len(json_str) > 100


def test_run_search_result_has_pool_diagnostics(
    search_result: SearchResult,
) -> None:
    diag = search_result.pool_diagnostics
    assert diag.n_total == len(search_result.all_scored)
    assert 0 <= diag.n_passing_conceal <= diag.n_total


def test_run_search_empty_when_all_fail_conceal(
    reference: Spectrum,
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
) -> None:
    # Conceal limit of 0.0 should reject everything
    cfg = SearchConfig(
        conceal_condition=d65_condition,
        reveal_conditions=(a_condition,),
        generation=CandidateGenerationConfig(seed=1, n_directions=10),
        ranking=RankingConfig(conceal_limit=0.0),
        n_diverse=3,
    )
    result = run_search(reference, cfg)
    assert result.pareto_scored == []
    assert result.selected == []


# ---------------------------------------------------------------------------
# Regression fixture
# ---------------------------------------------------------------------------


def test_run_search_regression(
    reference: Spectrum,
    d65_condition: ViewingCondition,
    a_condition: ViewingCondition,
) -> None:
    """Regression: fixed seed produces stable pool size and top-candidate metrics."""
    cfg = SearchConfig(
        conceal_condition=d65_condition,
        reveal_conditions=(a_condition,),
        generation=CandidateGenerationConfig(seed=42, n_directions=50),
        ranking=RankingConfig(conceal_limit=1.0),
        n_diverse=3,
    )
    result = run_search(reference, cfg)

    # Pool should be non-empty
    assert len(result.all_scored) > 0
    # Pareto front should be non-empty
    assert len(result.pareto_scored) > 0
    # Top selection should satisfy conceal threshold
    for c, _ in result.selected:
        assert c.conceal_delta_e00 <= 1.0
    # Top candidate should have a meaningful reveal score
    if result.selected:
        _, top_score = result.selected[0]
        assert top_score.minimum_reveal_delta_e00 > 0.0
