"""Tests for Phase B: reflectance-bound feasibility geometry."""

from __future__ import annotations

import numpy as np
import pytest

from metamerism.core import PROJECT_SHAPE
from metamerism.illuminants import get_illuminant
from metamerism.search.feasibility import (
    DEFAULT_REFLECTANCE_BOUND_TOLERANCE,
    feasible_step_interval,
    is_feasible,
    max_bound_violation,
)
from metamerism.search.metamer import ViewingCondition
from metamerism.search.projection import (
    build_xyz_projection,
    compute_null_space,
)

N = len(list(PROJECT_SHAPE.wavelengths))


# ---------------------------------------------------------------------------
# feasible_step_interval: basic geometry
# ---------------------------------------------------------------------------


def test_zero_direction_returns_none():
    c = np.full(5, 0.5)
    d = np.zeros(5)
    assert feasible_step_interval(c, d) is None


def test_positive_direction_upper_limited():
    # centre at 0.8, direction +1 → can step at most 0.2 up
    c = np.array([0.8])
    d = np.array([1.0])
    result = feasible_step_interval(c, d)
    assert result is not None
    alpha_min, alpha_max = result
    assert alpha_max == pytest.approx(0.2, abs=1e-12)
    assert alpha_min == pytest.approx(-0.8, abs=1e-12)


def test_negative_direction_lower_limited():
    # centre at 0.2, direction -1 → can step at most 0.2 down
    c = np.array([0.2])
    d = np.array([-1.0])
    result = feasible_step_interval(c, d)
    assert result is not None
    alpha_min, alpha_max = result
    assert alpha_max == pytest.approx(0.2, abs=1e-12)
    assert alpha_min == pytest.approx(-0.8, abs=1e-12)


def test_multi_sample_tightest_bound_governs():
    # Two samples: centre [0.3, 0.8], direction [1, 1]
    # Sample 0: step to upper 0.7, lower -0.3
    # Sample 1: step to upper 0.2, lower -0.8
    # alpha_max = min(0.7, 0.2) = 0.2
    # alpha_min = max(-0.3, -0.8) = -0.3
    c = np.array([0.3, 0.8])
    d = np.array([1.0, 1.0])
    result = feasible_step_interval(c, d)
    assert result is not None
    alpha_min, alpha_max = result
    assert alpha_max == pytest.approx(0.2, abs=1e-12)
    assert alpha_min == pytest.approx(-0.3, abs=1e-12)


def test_candidates_satisfy_bounds_after_step():
    """Points at alpha_min and alpha_max should stay within [0, 1]."""
    rng = np.random.default_rng(7)
    for _ in range(50):
        c = rng.uniform(0.1, 0.9, size=20)
        d = rng.standard_normal(20)
        result = feasible_step_interval(c, d)
        if result is None:
            continue
        alpha_min, alpha_max = result
        for alpha in (alpha_min, alpha_max):
            candidate = c + alpha * d
            assert np.all(candidate >= -1e-10)
            assert np.all(candidate <= 1.0 + 1e-10)


def test_feasible_interval_positive_width():
    """For a generic interior centre and random direction, interval width > 0."""
    rng = np.random.default_rng(99)
    c = np.full(81, 0.5)
    d = rng.standard_normal(81)
    result = feasible_step_interval(c, d)
    assert result is not None
    alpha_min, alpha_max = result
    assert alpha_max > alpha_min


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        feasible_step_interval(np.ones(3), np.ones(4))


def test_degenerate_interval_returns_none():
    # Both samples at lower boundary, directions pulling in opposite ways:
    # Sample 0 (c=0, d=+1): alpha >= 0 (lower bound)
    # Sample 1 (c=0, d=-1): alpha <= 0 (lower bound)
    # Combined: alpha_min=0, alpha_max=0 → zero width → None
    c = np.array([0.0, 0.0])
    d = np.array([1.0, -1.0])
    result = feasible_step_interval(c, d)
    assert result is None


# ---------------------------------------------------------------------------
# max_bound_violation and is_feasible
# ---------------------------------------------------------------------------


def test_max_bound_violation_zero_for_feasible():
    r = np.full(10, 0.5)
    assert max_bound_violation(r) == pytest.approx(0.0)


def test_max_bound_violation_detects_low():
    r = np.array([0.5, -0.05])
    assert max_bound_violation(r) == pytest.approx(0.05, abs=1e-12)


def test_max_bound_violation_detects_high():
    r = np.array([0.5, 1.03])
    assert max_bound_violation(r) == pytest.approx(0.03, abs=1e-12)


def test_is_feasible_true_within_tolerance():
    r = np.full(5, 0.5)
    assert is_feasible(r)


def test_is_feasible_false_exceeds_tolerance():
    r = np.array([0.5, -0.5])
    assert not is_feasible(r)


def test_is_feasible_respects_custom_tolerance():
    r = np.array([0.5, -0.00005])
    # Violation is 5e-5, below default tolerance 1e-4 → passes
    assert is_feasible(r, tolerance=DEFAULT_REFLECTANCE_BOUND_TOLERANCE)
    # Same violation exceeds tighter tolerance 1e-6 → fails
    assert not is_feasible(r, tolerance=1e-6)


# ---------------------------------------------------------------------------
# Integration: null-space steps stay feasible
# ---------------------------------------------------------------------------


def test_null_space_steps_satisfy_bounds():
    """Endpoints computed via feasible_step_interval should be within bounds."""
    cond = ViewingCondition(name="D65", illuminant=get_illuminant("D65"))
    proj = build_xyz_projection(cond)
    ns = compute_null_space(proj)

    rng = np.random.default_rng(42)
    centre = np.full(N, 0.5)

    for i in range(30):
        # Pick a random null-space direction
        z = rng.standard_normal(ns.n_null)
        z /= np.linalg.norm(z)
        direction = ns.basis @ z

        result = feasible_step_interval(centre, direction)
        if result is None:
            continue
        alpha_min, alpha_max = result

        for alpha, label in ((alpha_min, "alpha_min"), (alpha_max, "alpha_max")):
            candidate = centre + alpha * direction
            violation = max_bound_violation(candidate)
            assert violation <= DEFAULT_REFLECTANCE_BOUND_TOLERANCE, (
                f"direction {i}, {label}: violation={violation:.2e}"
            )
