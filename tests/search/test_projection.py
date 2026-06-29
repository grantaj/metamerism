"""Tests for Phase B: XYZ projection matrix and null-space geometry."""

from __future__ import annotations

import colour
import numpy as np
import pytest

from metamerism.core import PROJECT_SHAPE, Provenance, Spectrum, SpectrumDomain
from metamerism.illuminants import get_illuminant
from metamerism.pigments.golden import load_golden_heavy_body_paints
from metamerism.search.metamer import ViewingCondition
from metamerism.search.projection import (
    DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE,
    DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE,
    NullSpace,
    XYZProjection,
    build_xyz_projection,
    compute_null_space,
    projection_xyz,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WAVELENGTHS = np.asarray(PROJECT_SHAPE.wavelengths, dtype=np.float64)
N = len(WAVELENGTHS)  # 81


@pytest.fixture(scope="module")
def d65_condition() -> ViewingCondition:
    return ViewingCondition(name="D65", illuminant=get_illuminant("D65"))


@pytest.fixture(scope="module")
def a_condition() -> ViewingCondition:
    return ViewingCondition(name="A", illuminant=get_illuminant("A"))


@pytest.fixture(scope="module")
def d65_projection(d65_condition: ViewingCondition) -> XYZProjection:
    return build_xyz_projection(d65_condition)


@pytest.fixture(scope="module")
def d65_null(d65_projection: XYZProjection) -> NullSpace:
    return compute_null_space(d65_projection)


@pytest.fixture(scope="module")
def golden_spectrum() -> Spectrum:
    paints = load_golden_heavy_body_paints()
    return paints[0].reflectance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_reflectance(value: float) -> Spectrum:
    return Spectrum.from_arrays(
        WAVELENGTHS,
        np.full(N, value),
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source=f"flat_{value}"),
    )


def _smooth_reflectance(center: float = 580.0, width: float = 80.0) -> Spectrum:
    values = 0.15 + 0.6 * np.exp(
        -((WAVELENGTHS - center) ** 2) / (2 * width**2)
    )
    return Spectrum.from_arrays(
        WAVELENGTHS,
        values,
        domain=SpectrumDomain.REFLECTANCE,
        provenance=Provenance(source="smooth"),
    )


def _direct_xyz(
    spectrum: Spectrum, condition: ViewingCondition
) -> np.ndarray:
    """XYZ via direct colour call — the authoritative reference."""
    cmfs = colour.colorimetry.MSDS_CMFS[condition.observer_name]
    ill_sd = condition.illuminant.sd.copy().align(PROJECT_SHAPE)
    sd = spectrum.sd.copy().align(PROJECT_SHAPE)
    return np.array(
        colour.sd_to_XYZ(sd, cmfs=cmfs, illuminant=ill_sd, shape=PROJECT_SHAPE),
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# build_xyz_projection: shape and rank
# ---------------------------------------------------------------------------


def test_projection_matrix_shape(d65_projection: XYZProjection):
    assert d65_projection.matrix.shape == (3, N)


def test_projection_wavelengths_match_project_shape(d65_projection: XYZProjection):
    assert np.allclose(d65_projection.wavelengths_nm, WAVELENGTHS)


def test_projection_has_three_singular_values(d65_projection: XYZProjection):
    assert d65_projection.singular_values.shape == (3,)


def test_projection_rank_3_d65(d65_projection: XYZProjection, d65_null: NullSpace):
    assert d65_null.rank == 3


def test_projection_rank_3_a(a_condition: ViewingCondition):
    proj = build_xyz_projection(a_condition)
    ns = compute_null_space(proj)
    assert ns.rank == 3


def test_projection_null_dimension(d65_null: NullSpace):
    assert d65_null.n_null == N - 3
    assert d65_null.basis.shape == (N, N - 3)


def test_projection_observer_name_recorded(d65_projection: XYZProjection):
    assert d65_projection.observer_name == "CIE 1931 2 Degree Standard Observer"


def test_projection_illuminant_name_recorded(d65_projection: XYZProjection):
    assert "D65" in d65_projection.illuminant_name


# ---------------------------------------------------------------------------
# Matrix XYZ vs direct colour — Y=100 normalisation agreement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,value",
    [("perfect", 1.0), ("half", 0.5), ("dark", 0.1), ("bright", 0.9)],
)
def test_matrix_xyz_flat_agrees_with_colour(
    label: str,
    value: float,
    d65_condition: ViewingCondition,
    d65_projection: XYZProjection,
):
    sp = _flat_reflectance(value)
    r = sp.values
    xyz_matrix = projection_xyz(r, d65_projection)
    xyz_direct = _direct_xyz(sp, d65_condition)
    assert np.allclose(
        xyz_matrix, xyz_direct, atol=DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE
    ), f"flat {label}: matrix {xyz_matrix} vs colour {xyz_direct}"


def test_matrix_xyz_smooth_agrees_with_colour(
    d65_condition: ViewingCondition, d65_projection: XYZProjection
):
    sp = _smooth_reflectance()
    xyz_matrix = projection_xyz(sp.values, d65_projection)
    xyz_direct = _direct_xyz(sp, d65_condition)
    tol = DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE
    assert np.allclose(xyz_matrix, xyz_direct, atol=tol)


def test_matrix_xyz_smooth_blue_agrees_with_colour(
    d65_condition: ViewingCondition, d65_projection: XYZProjection
):
    sp = _smooth_reflectance(center=460.0, width=50.0)
    xyz_matrix = projection_xyz(sp.values, d65_projection)
    xyz_direct = _direct_xyz(sp, d65_condition)
    tol = DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE
    assert np.allclose(xyz_matrix, xyz_direct, atol=tol)


def test_matrix_xyz_smooth_red_agrees_with_colour(
    d65_condition: ViewingCondition, d65_projection: XYZProjection
):
    sp = _smooth_reflectance(center=680.0, width=40.0)
    xyz_matrix = projection_xyz(sp.values, d65_projection)
    xyz_direct = _direct_xyz(sp, d65_condition)
    tol = DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE
    assert np.allclose(xyz_matrix, xyz_direct, atol=tol)


def test_matrix_xyz_golden_spectrum_agrees_with_colour(
    golden_spectrum: Spectrum,
    d65_condition: ViewingCondition,
    d65_projection: XYZProjection,
):
    sp = golden_spectrum
    sd_aligned = sp.sd.copy().align(PROJECT_SHAPE)
    r = np.asarray(sd_aligned.values, dtype=np.float64)
    xyz_matrix = projection_xyz(r, d65_projection)
    xyz_direct = _direct_xyz(sp, d65_condition)
    tol = DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE
    assert np.allclose(xyz_matrix, xyz_direct, atol=tol)


def test_perfect_reflector_y_100(d65_projection: XYZProjection):
    r = np.ones(N)
    xyz = projection_xyz(r, d65_projection)
    assert xyz[1] == pytest.approx(100.0, abs=0.1)


# ---------------------------------------------------------------------------
# Null-space invariants
# ---------------------------------------------------------------------------


def test_null_space_product_near_zero(
    d65_projection: XYZProjection, d65_null: NullSpace
):
    """A₀N should be numerically zero."""
    product = d65_projection.matrix @ d65_null.basis
    assert np.allclose(product, 0.0, atol=1e-8), (
        f"max |A₀N| = {np.abs(product).max():.2e}"
    )


def test_null_space_basis_orthonormal(d65_null: NullSpace):
    NtN = d65_null.basis.T @ d65_null.basis
    assert np.allclose(NtN, np.eye(d65_null.n_null), atol=1e-10)


def test_null_space_perturbation_preserves_xyz(
    d65_condition: ViewingCondition,
    d65_projection: XYZProjection,
    d65_null: NullSpace,
):
    """Adding a null-space vector to a reflectance preserves its projected XYZ."""
    rng = np.random.default_rng(42)
    centre = _flat_reflectance(0.5).values
    xyz_centre = projection_xyz(centre, d65_projection)

    # Try 20 random null-space perturbations (small enough to stay in [0,1])
    for _ in range(20):
        z = rng.standard_normal(d65_null.n_null) * 0.01
        delta = d65_null.basis @ z
        r_perturbed = centre + delta
        xyz_perturbed = projection_xyz(r_perturbed, d65_projection)
        assert np.allclose(xyz_perturbed, xyz_centre, atol=1e-8), (
            f"XYZ shifted by {np.abs(xyz_perturbed - xyz_centre).max():.2e}"
        )


def test_null_space_tolerance_recorded(d65_null: NullSpace):
    assert d65_null.relative_tolerance == DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE


def test_null_space_custom_tolerance():
    cond = ViewingCondition(name="D50", illuminant=get_illuminant("D50"))
    proj = build_xyz_projection(cond)
    ns = compute_null_space(proj, relative_tolerance=1e-8)
    assert ns.relative_tolerance == 1e-8
    assert ns.rank == 3


# ---------------------------------------------------------------------------
# projection_xyz: input validation
# ---------------------------------------------------------------------------


def test_projection_xyz_wrong_length(d65_projection: XYZProjection):
    with pytest.raises(ValueError, match="length"):
        projection_xyz(np.ones(10), d65_projection)


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------


def test_xyz_projection_round_trip(d65_projection: XYZProjection):
    d = d65_projection.to_dict()
    restored = XYZProjection.from_dict(d)
    assert np.allclose(restored.matrix, d65_projection.matrix)
    assert np.allclose(restored.wavelengths_nm, d65_projection.wavelengths_nm)
    assert restored.observer_name == d65_projection.observer_name
    assert restored.illuminant_name == d65_projection.illuminant_name


def test_null_space_round_trip(d65_null: NullSpace):
    d = d65_null.to_dict()
    restored = NullSpace.from_dict(d)
    assert np.allclose(restored.basis, d65_null.basis)
    assert restored.rank == d65_null.rank
    assert restored.n_null == d65_null.n_null
    assert restored.relative_tolerance == d65_null.relative_tolerance
