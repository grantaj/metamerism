"""Tests for Phase A: conceal/reveal scoring vertical slice."""

from __future__ import annotations

import numpy as np
import pytest

from metamerism.core import Provenance, Spectrum, SpectrumDomain
from metamerism.illuminants import get_illuminant
from metamerism.search.metamer import (
    ConditionResult,
    MetamerScore,
    ViewingCondition,
    score_condition,
    score_metameric_pair,
)
from metamerism.search.objectives import (
    passes_conceal_threshold,
    rank_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WAVELENGTHS = np.linspace(380.0, 780.0, 81)


def _make_flat_reflectance(value: float, *, name: str = "flat") -> Spectrum:
    return Spectrum.from_arrays(
        WAVELENGTHS,
        np.full(81, value),
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source=name),
    )


def _make_smooth_reflectance(*, name: str = "smooth") -> Spectrum:
    """A smooth bell-shaped spectrum with values in [0.1, 0.8]."""
    center = 580.0
    width = 80.0
    values = 0.1 + 0.7 * np.exp(-((WAVELENGTHS - center) ** 2) / (2 * width**2))
    return Spectrum.from_arrays(
        WAVELENGTHS,
        values,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source=name),
    )


@pytest.fixture
def d65_condition() -> ViewingCondition:
    return ViewingCondition(name="D65_conceal", illuminant=get_illuminant("D65"))


@pytest.fixture
def a_condition() -> ViewingCondition:
    return ViewingCondition(name="A_reveal", illuminant=get_illuminant("A"))


@pytest.fixture
def f11_condition() -> ViewingCondition:
    return ViewingCondition(name="FL11_reveal", illuminant=get_illuminant("FL11"))


# ---------------------------------------------------------------------------
# Illuminant factory tests
# ---------------------------------------------------------------------------


def test_get_illuminant_d65_domain():
    ill = get_illuminant("D65")
    assert ill.domain is SpectrumDomain.ILLUMINANT


def test_get_illuminant_on_project_grid():
    ill = get_illuminant("D65")
    assert ill.is_on_canonical_grid()


def test_get_illuminant_non_negative():
    for name in ("D65", "D50", "A", "FL11"):
        ill = get_illuminant(name)
        assert np.all(ill.values >= 0.0), f"{name} has negative values"


def test_get_illuminant_unknown_raises():
    with pytest.raises(ValueError, match="Unknown illuminant"):
        get_illuminant("NOTREAL")


def test_get_illuminant_provenance():
    ill = get_illuminant("D50")
    assert "D50" in ill.provenance.source


# ---------------------------------------------------------------------------
# ViewingCondition validation
# ---------------------------------------------------------------------------


def test_viewing_condition_rejects_non_illuminant():
    refl = _make_flat_reflectance(0.5)
    with pytest.raises(ValueError, match="ILLUMINANT"):
        ViewingCondition(name="bad", illuminant=refl)


def test_viewing_condition_accepts_illuminant():
    ill = get_illuminant("D65")
    vc = ViewingCondition(name="test", illuminant=ill)
    assert vc.name == "test"
    assert vc.observer_name == "CIE 1931 2 Degree Standard Observer"


# ---------------------------------------------------------------------------
# score_condition: identical spectra → ΔE ≈ 0
# ---------------------------------------------------------------------------


def test_score_condition_identical_spectra_zero_delta_e(d65_condition):
    flat = _make_flat_reflectance(0.5)
    result = score_condition(flat, flat, d65_condition)
    assert result.delta_e00 == pytest.approx(0.0, abs=1e-6)


def test_score_condition_identical_smooth_zero_delta_e(d65_condition):
    smooth = _make_smooth_reflectance()
    result = score_condition(smooth, smooth, d65_condition)
    assert result.delta_e00 == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# score_condition: different spectra → ΔE > 0
# ---------------------------------------------------------------------------


def test_score_condition_different_spectra_nonzero(d65_condition):
    dark = _make_flat_reflectance(0.1, name="dark")
    bright = _make_flat_reflectance(0.9, name="bright")
    result = score_condition(dark, bright, d65_condition)
    assert result.delta_e00 > 1.0


def test_score_condition_result_fields(d65_condition):
    a = _make_flat_reflectance(0.3, name="a")
    b = _make_flat_reflectance(0.7, name="b")
    result = score_condition(a, b, d65_condition)
    assert result.condition_name == "D65_conceal"
    assert result.xyz_first.shape == (3,)
    assert result.xyz_second.shape == (3,)
    assert result.lab_first.shape == (3,)
    assert result.lab_second.shape == (3,)
    assert result.delta_e00 > 0.0


# ---------------------------------------------------------------------------
# score_condition: direct colour agreement
# ---------------------------------------------------------------------------


def test_score_condition_xyz_non_negative(d65_condition):
    sp = _make_smooth_reflectance()
    result = score_condition(sp, sp, d65_condition)
    assert np.all(result.xyz_first >= 0.0)


def test_perfect_reflector_y_approx_100(d65_condition):
    """A flat 100% reflector should have Y ≈ 100 under D65."""
    perfect = _make_flat_reflectance(1.0, name="perfect_white")
    result = score_condition(perfect, perfect, d65_condition)
    assert result.xyz_first[1] == pytest.approx(100.0, abs=0.5)


# ---------------------------------------------------------------------------
# score_metameric_pair
# ---------------------------------------------------------------------------


def test_score_metameric_pair_identical(d65_condition, a_condition):
    flat = _make_flat_reflectance(0.5)
    score = score_metameric_pair(flat, flat, d65_condition, [a_condition])
    assert score.conceal.delta_e00 == pytest.approx(0.0, abs=1e-6)
    assert score.reveal[0].delta_e00 == pytest.approx(0.0, abs=1e-6)
    assert score.minimum_reveal_delta_e00 == pytest.approx(0.0, abs=1e-6)


def test_score_metameric_pair_different(d65_condition, a_condition):
    dark = _make_flat_reflectance(0.1, name="dark")
    bright = _make_flat_reflectance(0.9, name="bright")
    score = score_metameric_pair(dark, bright, d65_condition, [a_condition])
    assert score.conceal.delta_e00 > 1.0
    assert score.minimum_reveal_delta_e00 > 0.0


def test_score_metameric_pair_multiple_reveal(
    d65_condition, a_condition, f11_condition
):
    flat = _make_flat_reflectance(0.5)
    score = score_metameric_pair(
        flat, flat, d65_condition, [a_condition, f11_condition]
    )
    assert len(score.reveal) == 2
    assert score.minimum_reveal_delta_e00 == pytest.approx(0.0, abs=1e-6)


def test_score_metameric_pair_minimum_reveal_is_min(
    d65_condition, a_condition, f11_condition
):
    dark = _make_flat_reflectance(0.1, name="dark")
    bright = _make_flat_reflectance(0.9, name="bright")
    score = score_metameric_pair(
        dark, bright, d65_condition, [a_condition, f11_condition]
    )
    expected_min = min(r.delta_e00 for r in score.reveal)
    assert score.minimum_reveal_delta_e00 == pytest.approx(expected_min, abs=1e-9)


def test_score_metameric_pair_rejects_non_reflectance(d65_condition, a_condition):
    ill = get_illuminant("D65")
    flat = _make_flat_reflectance(0.5)
    with pytest.raises(ValueError, match="REFLECTANCE"):
        score_metameric_pair(ill, flat, d65_condition, [a_condition])


def test_score_metameric_pair_requires_reveal(d65_condition):
    flat = _make_flat_reflectance(0.5)
    with pytest.raises(ValueError, match="reveal condition"):
        score_metameric_pair(flat, flat, d65_condition, [])


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_condition_result_round_trip(d65_condition):
    a = _make_flat_reflectance(0.3)
    b = _make_flat_reflectance(0.7)
    result = score_condition(a, b, d65_condition)
    d = result.to_dict()
    restored = ConditionResult.from_dict(d)
    assert restored.condition_name == result.condition_name
    assert np.allclose(restored.xyz_first, result.xyz_first)
    assert restored.delta_e00 == pytest.approx(result.delta_e00)


def test_metamer_score_round_trip(d65_condition, a_condition):
    dark = _make_flat_reflectance(0.1)
    bright = _make_flat_reflectance(0.9)
    score = score_metameric_pair(dark, bright, d65_condition, [a_condition])
    d = score.to_dict()
    restored = MetamerScore.from_dict(d)
    assert restored.conceal.delta_e00 == pytest.approx(score.conceal.delta_e00)
    assert restored.minimum_reveal_delta_e00 == pytest.approx(
        score.minimum_reveal_delta_e00
    )


# ---------------------------------------------------------------------------
# objectives: filtering and ranking
# ---------------------------------------------------------------------------


def test_passes_conceal_threshold_true():
    result = ConditionResult(
        "c", np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.5
    )
    score = MetamerScore(conceal=result, reveal=(result,), minimum_reveal_delta_e00=5.0)
    assert passes_conceal_threshold(score, limit=1.0)


def test_passes_conceal_threshold_false():
    result = ConditionResult(
        "c", np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 2.0
    )
    score = MetamerScore(conceal=result, reveal=(result,), minimum_reveal_delta_e00=5.0)
    assert not passes_conceal_threshold(score, limit=1.0)


def test_rank_candidates_filters_and_sorts(d65_condition, a_condition):
    spectra = [
        _make_flat_reflectance(v, name=f"r{i}")
        for i, v in enumerate([0.1, 0.3, 0.6, 0.9])
    ]

    def _make_score(conceal_de: float, reveal_de: float) -> MetamerScore:
        c = ConditionResult(
            "conceal", np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), conceal_de
        )
        r = ConditionResult(
            "reveal", np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), reveal_de
        )
        return MetamerScore(conceal=c, reveal=(r,), minimum_reveal_delta_e00=reveal_de)

    candidates = [
        (spectra[0], _make_score(0.2, 10.0)),  # passes, high reveal
        (spectra[1], _make_score(1.5, 20.0)),  # fails conceal
        (spectra[2], _make_score(0.8, 5.0)),  # passes, low reveal
        (spectra[3], _make_score(0.5, 15.0)),  # passes, medium reveal
    ]

    ranked = rank_candidates(candidates, conceal_limit=1.0)
    assert len(ranked) == 3
    reveal_scores = [sc.minimum_reveal_delta_e00 for _, sc in ranked]
    assert reveal_scores == sorted(reveal_scores, reverse=True)
    assert reveal_scores[0] == pytest.approx(15.0)


def test_rank_candidates_empty():
    assert rank_candidates([]) == []
