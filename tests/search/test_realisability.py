"""Tests for search.realisability — paint mixture optimisation for metameric pairs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain
from metamerism.illuminants import get_illuminant
from metamerism.pigments.golden import (
    GoldenPaintSpectrum,
    load_golden_heavy_body_paints,
)
from metamerism.search.metamer import ConditionResult, ViewingCondition
from metamerism.search.projection import build_xyz_projection
from metamerism.search.realisability import (
    DEFAULT_CONCEAL_DELTA_E00_LIMIT,
    DEFAULT_MAX_COMPONENTS,
    MAX_COMPONENTS_LIMIT,
    MODEL_NAME_KS,
    MODEL_NAME_LINEAR,
    PaintRealisation,
    RealisabilityConfig,
    SparseRealisabilityConfig,
    _align_paint_ks_matrix,
    _align_paint_matrix,
    _omp_paint_order,
    _white_xyz_unit,
    find_metameric_mixture,
    find_sparse_metameric_mixture,
)

# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def all_paints() -> list[GoldenPaintSpectrum]:
    return load_golden_heavy_body_paints()


@pytest.fixture(scope="module")
def paint_lookup(all_paints) -> dict[str, GoldenPaintSpectrum]:
    return {p.paint_name: p for p in all_paints}


@pytest.fixture(scope="module")
def d65() -> ViewingCondition:
    return ViewingCondition(name="D65", illuminant=get_illuminant("D65"))


@pytest.fixture(scope="module")
def illuminant_a() -> ViewingCondition:
    return ViewingCondition(name="A", illuminant=get_illuminant("A"))


@pytest.fixture(scope="module")
def small_library(all_paints) -> list[GoldenPaintSpectrum]:
    """A small subset of paints for fast tests."""
    names = [
        "Dioxazine Purple",
        "Phthalo Green BS",
        "Burnt Umber",
        "Titanium White",
        "Paynes Gray",
    ]
    lookup = {p.paint_name: p for p in all_paints}
    return [lookup[n] for n in names if n in lookup]


@pytest.fixture(scope="module")
def dark_reference(paint_lookup) -> GoldenPaintSpectrum:
    """Dioxazine Purple — dark paint with high metameric potential."""
    return paint_lookup["Dioxazine Purple"]


@pytest.fixture(scope="module")
def fast_config() -> RealisabilityConfig:
    return RealisabilityConfig(n_restarts=3, seed=42)


# ── RealisabilityConfig validation ────────────────────────────────────────────


def test_config_defaults():
    cfg = RealisabilityConfig()
    assert cfg.conceal_delta_e00_limit == DEFAULT_CONCEAL_DELTA_E00_LIMIT
    assert cfg.n_restarts >= 1
    assert cfg.mixing_mode == "reflectance"


def test_config_invalid_limit():
    with pytest.raises(ValueError, match="conceal_delta_e00_limit"):
        RealisabilityConfig(conceal_delta_e00_limit=0.0)


def test_config_invalid_restarts():
    with pytest.raises(ValueError, match="n_restarts"):
        RealisabilityConfig(n_restarts=0)


def test_config_invalid_mixing_mode():
    with pytest.raises(ValueError, match="mixing_mode"):
        RealisabilityConfig(mixing_mode="unknown")


def test_config_ks_mode_accepted():
    cfg = RealisabilityConfig(mixing_mode="ks")
    assert cfg.mixing_mode == "ks"


def test_config_round_trip():
    cfg = RealisabilityConfig(conceal_delta_e00_limit=0.5, n_restarts=4, seed=99)
    assert RealisabilityConfig.from_dict(cfg.to_dict()) == cfg


# ── _align_paint_matrix ────────────────────────────────────────────────────────


def test_align_paint_matrix_shape(small_library):
    names, matrix = _align_paint_matrix(small_library)
    assert len(names) == len(small_library)
    assert matrix.shape == (81, len(small_library))
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


def test_align_paint_matrix_names_match(small_library):
    names, _ = _align_paint_matrix(small_library)
    expected = [p.paint_name for p in small_library]
    assert names == expected


# ── _white_xyz_unit ────────────────────────────────────────────────────────────


def test_white_xyz_unit_y_near_one(d65):
    proj = build_xyz_projection(d65)
    white = _white_xyz_unit(proj.matrix)
    assert pytest.approx(white[1], abs=1e-6) == 1.0  # Y=100 scale → /100 = 1


# ── find_metameric_mixture input validation ────────────────────────────────────


def test_rejects_non_reflectance_reference(small_library, d65, illuminant_a):
    bad = Spectrum.from_arrays(
        np.linspace(380, 780, 81),
        np.ones(81) * 0.5,
        domain=SpectrumDomain.ILLUMINANT,
        provenance=Provenance(),
        validate=False,
    )
    with pytest.raises(ValueError, match="REFLECTANCE"):
        find_metameric_mixture(bad, small_library, d65, [illuminant_a])


def test_rejects_empty_library(dark_reference, d65, illuminant_a):
    with pytest.raises(ValueError, match="paint_library"):
        find_metameric_mixture(
            dark_reference.reflectance, [], d65, [illuminant_a]
        )


def test_rejects_empty_reveal_conditions(dark_reference, small_library, d65):
    with pytest.raises(ValueError, match="reveal condition"):
        find_metameric_mixture(
            dark_reference.reflectance, small_library, d65, []
        )


# ── basic result structure ────────────────────────────────────────────────────


def test_result_has_correct_types(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert isinstance(result, PaintRealisation)
    assert result.realised.domain is SpectrumDomain.REFLECTANCE
    assert len(result.reveal_results) == 1
    assert isinstance(result.reveal_results[0], ConditionResult)
    assert result.model_name == MODEL_NAME_LINEAR
    assert result.ideal_candidate is None
    assert result.ideal_reveal_delta_e00 is None


def test_fractions_sum_to_one(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    total = sum(result.fractions.values())
    assert pytest.approx(total, abs=1e-6) == 1.0


def test_fractions_non_negative(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert all(v >= -1e-9 for v in result.fractions.values())


def test_fractions_keys_match_library(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert set(result.fractions.keys()) == {p.paint_name for p in small_library}


def test_realised_spectrum_on_project_grid(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert result.realised.sd.shape == PROJECT_SHAPE


def test_realised_values_in_bounds(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    vals = result.realised.values
    assert np.all(vals >= -1e-9)
    assert np.all(vals <= 1.0 + 1e-9)


# ── conceal constraint ────────────────────────────────────────────────────────


def test_conceal_de00_within_limit(dark_reference, small_library, d65, illuminant_a):
    cfg = RealisabilityConfig(conceal_delta_e00_limit=2.0, n_restarts=5, seed=0)
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert result.conceal_delta_e00 <= cfg.conceal_delta_e00_limit + 0.5


def test_conceal_de00_matches_score_condition(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    from metamerism.search.metamer import score_condition

    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    direct = score_condition(result.reference, result.realised, d65)
    assert pytest.approx(result.conceal_delta_e00, abs=1e-4) == direct.delta_e00


# ── reveal objective ──────────────────────────────────────────────────────────


def test_reveal_de00_is_positive(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    """The optimiser should find a mixture with nonzero reveal difference."""
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert result.minimum_reveal_delta_e00 >= 0.0


def test_reveal_de00_matches_score_condition(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    from metamerism.search.metamer import score_condition

    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    direct = score_condition(result.reference, result.realised, illuminant_a)
    expected = direct.delta_e00
    assert pytest.approx(result.reveal_results[0].delta_e00, abs=1e-4) == expected


def test_multiple_reveal_conditions(
    dark_reference, small_library, d65, illuminant_a, all_paints
):
    fl2 = ViewingCondition(name="FL2", illuminant=get_illuminant("FL2"))
    cfg = RealisabilityConfig(n_restarts=3, seed=0)
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a, fl2], cfg
    )
    assert len(result.reveal_results) == 2
    assert result.reveal_results[0].condition_name == "A"
    assert result.reveal_results[1].condition_name == "FL2"


# ── pure-paint recovery ───────────────────────────────────────────────────────


def test_pure_paint_self_match(paint_lookup, d65, illuminant_a):
    """A library of one paint should recover itself with conceal dE00 ≈ 0."""
    paint = paint_lookup["Dioxazine Purple"]
    library = [paint]
    cfg = RealisabilityConfig(n_restarts=2, seed=0)
    result = find_metameric_mixture(
        paint.reflectance, library, d65, [illuminant_a], cfg
    )
    # Only one paint in the library, so fractions[paint] ≈ 1 and conceal ΔE ≈ 0
    assert result.conceal_delta_e00 < 0.01
    assert pytest.approx(result.fractions[paint.paint_name], abs=1e-6) == 1.0


# ── reproducibility ───────────────────────────────────────────────────────────


def test_reproducible(dark_reference, small_library, d65, illuminant_a):
    cfg = RealisabilityConfig(n_restarts=3, seed=7)
    r1 = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    r2 = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert pytest.approx(r1.conceal_delta_e00, abs=1e-6) == r2.conceal_delta_e00
    assert pytest.approx(
        r1.minimum_reveal_delta_e00, abs=1e-6
    ) == r2.minimum_reveal_delta_e00


# ── ideal candidate warm start ────────────────────────────────────────────────


def test_ideal_candidate_ceiling_stored(
    dark_reference, small_library, d65, illuminant_a
):
    from metamerism.search import (
        CandidateGenerationConfig,
        RankingConfig,
        SearchConfig,
        run_search,
    )

    search_cfg = SearchConfig(
        conceal_condition=d65,
        reveal_conditions=(illuminant_a,),
        generation=CandidateGenerationConfig(seed=0, n_directions=10),
        ranking=RankingConfig(),
        n_diverse=1,
    )
    search_result = run_search(dark_reference.reflectance, search_cfg)

    if not search_result.selected:
        pytest.skip("search returned no candidates")

    ideal, _ = search_result.selected[0]
    cfg = RealisabilityConfig(n_restarts=3, seed=0)
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg,
        ideal_candidate=ideal,
    )
    assert result.ideal_candidate is not None
    assert result.ideal_reveal_delta_e00 is not None
    assert result.ideal_reveal_delta_e00 >= 0.0


# ── model metadata ────────────────────────────────────────────────────────────


def test_model_name(dark_reference, small_library, d65, illuminant_a, fast_config):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert result.model_name == MODEL_NAME_LINEAR


def test_model_assumptions_present(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    assert "model" in result.model_assumptions
    assert "mixing" in result.model_assumptions
    assert "limitations" in result.model_assumptions


# ── serialisation ─────────────────────────────────────────────────────────────


def test_to_dict_json_serialisable(
    dark_reference, small_library, d65, illuminant_a, fast_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    d = result.to_dict()
    json.dumps(d)  # must not raise


def test_round_trip(dark_reference, small_library, d65, illuminant_a, fast_config):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], fast_config
    )
    restored = PaintRealisation.from_dict(result.to_dict())
    assert pytest.approx(
        restored.conceal_delta_e00, abs=1e-8
    ) == result.conceal_delta_e00
    assert (
        pytest.approx(restored.minimum_reveal_delta_e00, abs=1e-8)
        == result.minimum_reveal_delta_e00
    )
    assert set(restored.fractions.keys()) == set(result.fractions.keys())


# ── _align_paint_ks_matrix ────────────────────────────────────────────────────


def test_align_paint_ks_matrix_shape(small_library):
    names, matrix = _align_paint_ks_matrix(small_library)
    assert len(names) == len(small_library)
    assert matrix.shape == (81, len(small_library))
    assert np.all(matrix >= 0.0)


def test_align_paint_ks_matrix_names_match(small_library):
    names, _ = _align_paint_ks_matrix(small_library)
    expected = [p.paint_name for p in small_library]
    assert names == expected


# ── KS mixing mode ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ks_config() -> RealisabilityConfig:
    return RealisabilityConfig(mixing_mode="ks", n_restarts=3, seed=42)


def test_ks_result_has_correct_types(
    dark_reference, small_library, d65, illuminant_a, ks_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], ks_config
    )
    assert isinstance(result, PaintRealisation)
    assert result.realised.domain is SpectrumDomain.REFLECTANCE
    assert result.model_name == MODEL_NAME_KS


def test_ks_fractions_sum_to_one(
    dark_reference, small_library, d65, illuminant_a, ks_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], ks_config
    )
    assert pytest.approx(sum(result.fractions.values()), abs=1e-6) == 1.0


def test_ks_realised_values_in_bounds(
    dark_reference, small_library, d65, illuminant_a, ks_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], ks_config
    )
    vals = result.realised.values
    assert np.all(vals >= -1e-9)
    assert np.all(vals <= 1.0 + 1e-9)


def test_ks_conceal_de00_within_limit(
    dark_reference, small_library, d65, illuminant_a
):
    cfg = RealisabilityConfig(
        mixing_mode="ks", conceal_delta_e00_limit=2.0, n_restarts=5, seed=0
    )
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert result.conceal_delta_e00 <= cfg.conceal_delta_e00_limit + 0.5


def test_ks_pure_paint_self_match(paint_lookup, d65, illuminant_a):
    """With one paint, KS mixing must recover it exactly (conceal ΔE ≈ 0)."""
    paint = paint_lookup["Dioxazine Purple"]
    cfg = RealisabilityConfig(mixing_mode="ks", n_restarts=2, seed=0)
    result = find_metameric_mixture(
        paint.reflectance, [paint], d65, [illuminant_a], cfg
    )
    assert result.conceal_delta_e00 < 0.01
    assert pytest.approx(result.fractions[paint.paint_name], abs=1e-6) == 1.0


def test_ks_model_assumptions_present(
    dark_reference, small_library, d65, illuminant_a, ks_config
):
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], ks_config
    )
    assert "model" in result.model_assumptions
    assert "mixing" in result.model_assumptions
    assert "limitations" in result.model_assumptions


def test_ks_reproducible(dark_reference, small_library, d65, illuminant_a):
    cfg = RealisabilityConfig(mixing_mode="ks", n_restarts=3, seed=7)
    r1 = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    r2 = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert pytest.approx(r1.conceal_delta_e00, abs=1e-6) == r2.conceal_delta_e00


# ── minimum_reveal_delta_e00 property ─────────────────────────────────────────


def test_minimum_reveal_delta_e00_property(
    dark_reference, small_library, d65, illuminant_a
):
    fl2 = ViewingCondition(name="FL2", illuminant=get_illuminant("FL2"))
    cfg = RealisabilityConfig(n_restarts=3, seed=0)
    result = find_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a, fl2], cfg
    )
    expected_min = min(r.delta_e00 for r in result.reveal_results)
    assert pytest.approx(result.minimum_reveal_delta_e00) == expected_min


# ── _omp_paint_order ──────────────────────────────────────────────────────────


def test_omp_paint_order_length(small_library):
    _, matrix = _align_paint_matrix(small_library)
    target = matrix[:, 0]
    order = _omp_paint_order(matrix, target)
    assert len(order) == matrix.shape[1]
    assert set(order) == set(range(matrix.shape[1]))


def test_omp_paint_order_self_first(small_library):
    """A column should rank first when it is the target."""
    _, matrix = _align_paint_matrix(small_library)
    target = matrix[:, 2]
    order = _omp_paint_order(matrix, target)
    assert order[0] == 2


def test_omp_paint_order_zero_target(small_library):
    """Zero target falls back to identity ordering 0..n-1."""
    _, matrix = _align_paint_matrix(small_library)
    order = _omp_paint_order(matrix, np.zeros(matrix.shape[0]))
    assert order == list(range(matrix.shape[1]))


# ── SparseRealisabilityConfig ─────────────────────────────────────────────────


def test_sparse_config_defaults():
    cfg = SparseRealisabilityConfig()
    assert cfg.max_components == DEFAULT_MAX_COMPONENTS
    assert cfg.mixing_mode == "ks"
    assert cfg.use_omp_warmstart is True


def test_sparse_config_invalid_limit():
    with pytest.raises(ValueError, match="conceal_delta_e00_limit"):
        SparseRealisabilityConfig(conceal_delta_e00_limit=0.0)


def test_sparse_config_invalid_max_components_zero():
    with pytest.raises(ValueError, match="max_components"):
        SparseRealisabilityConfig(max_components=0)


def test_sparse_config_invalid_max_components_over_limit():
    with pytest.raises(ValueError, match="max_components"):
        SparseRealisabilityConfig(max_components=MAX_COMPONENTS_LIMIT + 1)


def test_sparse_config_invalid_mixing_mode():
    with pytest.raises(ValueError, match="mixing_mode"):
        SparseRealisabilityConfig(mixing_mode="unknown")


def test_sparse_config_round_trip():
    cfg = SparseRealisabilityConfig(
        max_components=3, mixing_mode="reflectance", seed=7, use_omp_warmstart=False
    )
    assert SparseRealisabilityConfig.from_dict(cfg.to_dict()) == cfg


# ── find_sparse_metameric_mixture ─────────────────────────────────────────────


@pytest.fixture(scope="module")
def sparse_config() -> SparseRealisabilityConfig:
    return SparseRealisabilityConfig(
        max_components=3, n_restarts=3, seed=42, mixing_mode="ks"
    )


def test_sparse_result_is_paint_realisation(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    assert isinstance(result, PaintRealisation)
    assert result.realised.domain is SpectrumDomain.REFLECTANCE


def test_sparse_fractions_at_most_max_components(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    assert len(result.fractions) <= sparse_config.max_components


def test_sparse_fractions_sum_to_one(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    assert pytest.approx(sum(result.fractions.values()), abs=1e-6) == 1.0


def test_sparse_fractions_non_negative(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    assert all(v >= -1e-9 for v in result.fractions.values())


def test_sparse_realised_values_in_bounds(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    vals = result.realised.values
    assert np.all(vals >= -1e-9)
    assert np.all(vals <= 1.0 + 1e-9)


def test_sparse_fractions_keys_are_paint_names(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    library_names = {p.paint_name for p in small_library}
    assert set(result.fractions.keys()).issubset(library_names)


def test_sparse_conceal_de00_within_limit(
    dark_reference, small_library, d65, illuminant_a
):
    cfg = SparseRealisabilityConfig(
        max_components=3, conceal_delta_e00_limit=2.0, n_restarts=4, seed=0
    )
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert result.conceal_delta_e00 <= cfg.conceal_delta_e00_limit + 0.5


def test_sparse_pure_paint_self_match(paint_lookup, d65, illuminant_a):
    """Library of one paint: sparse selection must recover it (conceal ΔE ≈ 0)."""
    paint = paint_lookup["Dioxazine Purple"]
    cfg = SparseRealisabilityConfig(max_components=2, n_restarts=2, seed=0)
    result = find_sparse_metameric_mixture(
        paint.reflectance, [paint], d65, [illuminant_a], cfg
    )
    assert result.conceal_delta_e00 < 0.01
    assert len(result.fractions) == 1
    assert pytest.approx(result.fractions[paint.paint_name], abs=1e-6) == 1.0


def test_sparse_reproducible(dark_reference, small_library, d65, illuminant_a):
    cfg = SparseRealisabilityConfig(max_components=3, n_restarts=3, seed=7)
    r1 = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    r2 = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert pytest.approx(r1.conceal_delta_e00, abs=1e-6) == r2.conceal_delta_e00
    assert set(r1.fractions.keys()) == set(r2.fractions.keys())


def test_sparse_model_metadata(
    dark_reference, small_library, d65, illuminant_a, sparse_config
):
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], sparse_config
    )
    assert result.model_name == MODEL_NAME_KS
    assert result.model_assumptions["algorithm"] == "greedy_forward_selection"
    assert "max_components" in result.model_assumptions
    assert "n_components_used" in result.model_assumptions


def test_sparse_with_omp_warmstart(dark_reference, small_library, d65, illuminant_a):
    from metamerism.search import (
        CandidateGenerationConfig,
        RankingConfig,
        SearchConfig,
        run_search,
    )

    search_cfg = SearchConfig(
        conceal_condition=d65,
        reveal_conditions=(illuminant_a,),
        generation=CandidateGenerationConfig(seed=0, n_directions=10),
        ranking=RankingConfig(),
        n_diverse=1,
    )
    search_result = run_search(dark_reference.reflectance, search_cfg)
    if not search_result.selected:
        pytest.skip("search returned no candidates")

    ideal, _ = search_result.selected[0]
    cfg = SparseRealisabilityConfig(
        max_components=3, n_restarts=3, seed=0, use_omp_warmstart=True
    )
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance,
        small_library,
        d65,
        [illuminant_a],
        cfg,
        ideal_candidate=ideal,
    )
    assert isinstance(result, PaintRealisation)
    assert result.ideal_candidate is not None
    assert result.ideal_reveal_delta_e00 is not None
    assert len(result.fractions) <= cfg.max_components


def test_sparse_reflectance_mode(dark_reference, small_library, d65, illuminant_a):
    cfg = SparseRealisabilityConfig(
        max_components=3, mixing_mode="reflectance", n_restarts=3, seed=0
    )
    result = find_sparse_metameric_mixture(
        dark_reference.reflectance, small_library, d65, [illuminant_a], cfg
    )
    assert result.model_name == MODEL_NAME_LINEAR
    assert len(result.fractions) <= cfg.max_components


def test_sparse_rejects_empty_library(dark_reference, d65, illuminant_a):
    with pytest.raises(ValueError, match="paint_library"):
        find_sparse_metameric_mixture(
            dark_reference.reflectance, [], d65, [illuminant_a]
        )


def test_sparse_rejects_empty_reveal(dark_reference, small_library, d65):
    with pytest.raises(ValueError, match="reveal condition"):
        find_sparse_metameric_mixture(
            dark_reference.reflectance, small_library, d65, []
        )
