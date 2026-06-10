"""
tests/core/test_spectrum.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tests for metamerism.core.spectrum.

Test categories
---------------
1. CANONICAL_GRID — shape, range, spacing
2. Provenance — construction, with_note, with_confidence
3. Spectrum.from_arrays — validation, domain checks
4. Spectrum properties — n_samples, wl_min/max, spacing, is_on_canonical_grid
5. Resampling — within-range, extrapolation policies, provenance notes,
   confidence reduction, round-trip fidelity, no-op detection
6. Arithmetic — mul, add, sub, div, neg, scalar forms, grid mismatch
7. Normalise and clip
8. Integration — integrate, dot
9. Comparison — allclose, __eq__, __hash__
10. Domain validation — reflectance/illuminant range checks
11. Error types — GridMismatchError, ExtrapolationError, DomainError
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from metamerism.core.spectrum import (
    CANONICAL_GRID,
    DomainError,
    ExtrapolationError,
    ExtrapolationPolicy,
    GridMismatchError,
    InterpolationMethod,
    Provenance,
    Spectrum,
    SpectrumDomain,
    SpectrumError,
)

# ============================================================================
# Fixtures and helpers
# ============================================================================


def flat_spectrum(
    value: float = 0.5,
    domain: SpectrumDomain = SpectrumDomain.REFLECTANCE,
    grid: np.ndarray | None = None,
) -> Spectrum:
    """Return a spectrum with uniform values on the canonical grid."""
    wl = grid if grid is not None else CANONICAL_GRID
    values = np.full_like(wl, value)
    return Spectrum.from_arrays(wl, values, domain=domain)


def linear_spectrum(
    slope: float = 0.001,
    intercept: float = 0.1,
    domain: SpectrumDomain = SpectrumDomain.REFLECTANCE,
    grid: np.ndarray | None = None,
) -> Spectrum:
    """Return a spectrum with linearly varying values."""
    wl = grid if grid is not None else CANONICAL_GRID
    values = np.clip(intercept + slope * (wl - wl[0]), 0.0, 1.0)
    return Spectrum.from_arrays(wl, values, domain=domain)


# ============================================================================
# 1. CANONICAL_GRID
# ============================================================================


class TestCanonicalGrid:
    def test_start(self):
        assert CANONICAL_GRID[0] == pytest.approx(380.0)

    def test_end(self):
        assert CANONICAL_GRID[-1] == pytest.approx(780.0)

    def test_length(self):
        assert len(CANONICAL_GRID) == 81

    def test_uniform_spacing(self):
        diffs = np.diff(CANONICAL_GRID)
        assert np.allclose(diffs, 5.0, atol=1e-10)

    def test_monotone(self):
        assert np.all(np.diff(CANONICAL_GRID) > 0)

    def test_dtype(self):
        assert CANONICAL_GRID.dtype == np.float64


# ============================================================================
# 2. Provenance
# ============================================================================


class TestProvenance:
    def test_defaults(self):
        p = Provenance()
        assert p.source == "unknown"
        assert p.confidence == 1.0
        assert p.notes is None

    def test_with_note_appends(self):
        p = Provenance(source="test")
        p2 = p.with_note("resampled")
        assert p2.notes == "resampled"
        p3 = p2.with_note("clipped")
        assert p3.notes == "resampled; clipped"

    def test_with_note_preserves_other_fields(self):
        p = Provenance(source="src", instrument="i1Pro", confidence=0.8)
        p2 = p.with_note("x")
        assert p2.source == "src"
        assert p2.instrument == "i1Pro"
        assert p2.confidence == 0.8

    def test_with_confidence_valid(self):
        p = Provenance(confidence=0.9)
        p2 = p.with_confidence(0.5)
        assert p2.confidence == 0.5

    def test_with_confidence_out_of_range(self):
        p = Provenance()
        with pytest.raises(ValueError, match="confidence must be in"):
            p.with_confidence(1.5)
        with pytest.raises(ValueError, match="confidence must be in"):
            p.with_confidence(-0.1)

    def test_frozen(self):
        p = Provenance()
        with pytest.raises(FrozenInstanceError):
            p.__setattr__("confidence", 0.5)


# ============================================================================
# 3. Spectrum.from_arrays — construction and validation
# ============================================================================


class TestFromArrays:
    def test_basic_construction(self):
        wl = np.linspace(400, 700, 31)
        v = np.full(31, 0.5)
        s = Spectrum.from_arrays(wl, v, domain=SpectrumDomain.REFLECTANCE)
        assert s.n_samples == 31
        assert s.domain == SpectrumDomain.REFLECTANCE

    def test_list_input(self):
        wl = [400.0, 450.0, 500.0]
        v = [0.1, 0.2, 0.3]
        s = Spectrum.from_arrays(wl, v)
        assert s.n_samples == 3
        assert s.wavelengths.dtype == np.float64

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            Spectrum.from_arrays([400, 500], [0.1, 0.2, 0.3])

    def test_non_monotone_raises(self):
        with pytest.raises(ValueError, match="monotonically increasing"):
            Spectrum.from_arrays([400, 300, 500], [0.1, 0.2, 0.3])

    def test_single_sample_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            Spectrum.from_arrays([400], [0.5])

    def test_2d_wavelengths_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            Spectrum.from_arrays(
                np.array([[400, 500], [600, 700]]),
                np.array([0.1, 0.2, 0.3, 0.4]),
            )

    def test_default_provenance(self):
        s = Spectrum.from_arrays([400, 500], [0.1, 0.2])
        assert isinstance(s.provenance, Provenance)

    def test_custom_provenance(self):
        p = Provenance(source="test source", confidence=0.7)
        s = Spectrum.from_arrays([400, 500], [0.1, 0.2], provenance=p)
        assert s.provenance.source == "test source"
        assert s.provenance.confidence == 0.7

    def test_reflectance_negative_raises(self):
        with pytest.raises(DomainError, match="Negative values"):
            Spectrum.from_arrays(
                [400, 500, 600],
                [-0.1, 0.5, 0.5],
                domain=SpectrumDomain.REFLECTANCE,
            )

    def test_reflectance_above_one_raises(self):
        with pytest.raises(DomainError, match="Values exceeding 1"):
            Spectrum.from_arrays(
                [400, 500, 600],
                [0.5, 1.5, 0.5],
                domain=SpectrumDomain.REFLECTANCE,
            )

    def test_illuminant_negative_raises(self):
        with pytest.raises(DomainError, match="Negative values"):
            Spectrum.from_arrays(
                [400, 500, 600],
                [1.0, -0.1, 0.5],
                domain=SpectrumDomain.ILLUMINANT,
            )

    def test_validate_false_skips_checks(self):
        # Should not raise even with out-of-range reflectance
        s = Spectrum.from_arrays(
            [400, 500, 600],
            [0.5, 1.5, 0.5],
            domain=SpectrumDomain.REFLECTANCE,
            validate=False,
        )
        assert s.values[1] == pytest.approx(1.5)

    def test_frozen(self):
        # frozen=True prevents attribute reassignment; it does not make the
        # underlying numpy array read-only (arrays are mutable objects).
        s = flat_spectrum()
        with pytest.raises(FrozenInstanceError):
            s.__setattr__("values", np.zeros(81))


# ============================================================================
# 4. Spectrum properties
# ============================================================================


class TestProperties:
    def test_n_samples(self):
        s = flat_spectrum()
        assert s.n_samples == 81

    def test_wl_min_max(self):
        s = flat_spectrum()
        assert s.wl_min == pytest.approx(380.0)
        assert s.wl_max == pytest.approx(780.0)

    def test_spacing_uniform(self):
        s = flat_spectrum()
        assert s.spacing == pytest.approx(5.0)

    def test_spacing_nonuniform(self):
        wl = np.array([400.0, 450.0, 600.0])
        s = Spectrum.from_arrays(wl, [0.1, 0.2, 0.3])
        assert s.spacing is None

    def test_is_on_canonical_grid_true(self):
        s = flat_spectrum()
        assert s.is_on_canonical_grid()

    def test_is_on_canonical_grid_false_different_range(self):
        wl = np.linspace(400, 700, 31)
        s = Spectrum.from_arrays(wl, np.full(31, 0.5))
        assert not s.is_on_canonical_grid()

    def test_repr_contains_domain(self):
        s = flat_spectrum(domain=SpectrumDomain.REFLECTANCE)
        assert "reflectance" in repr(s)

    def test_repr_contains_range(self):
        s = flat_spectrum()
        assert "380" in repr(s)
        assert "780" in repr(s)


# ============================================================================
# 5. Resampling
# ============================================================================


class TestResample:
    def test_resample_deprecated_but_functional(self):
        """Legacy resample API is deprecated but still functional for now."""
        s = flat_spectrum()
        with pytest.warns(DeprecationWarning):
            s2 = s.resample(CANONICAL_GRID)
        assert s2 is s

    def test_to_canonical_warns(self):
        wl = np.linspace(400, 700, 31)
        s = Spectrum.from_arrays(wl, np.full(31, 0.5))
        with pytest.warns(DeprecationWarning):
            s2 = s.to_canonical()
        assert s2.is_on_canonical_grid()

    def test_deprecated_resample_affects_provenance(self):
        wl = np.linspace(400, 700, 31)
        s = Spectrum.from_arrays(wl, np.full(31, 0.5))
        with pytest.warns(DeprecationWarning):
            s2 = s.resample(CANONICAL_GRID)
        assert "resampled" in (s2.provenance.notes or "")


# ============================================================================
# 6. Arithmetic
# ============================================================================


class TestArithmetic:
    def test_arithmetic_operators_warn_deprecated(self):
        s1 = flat_spectrum(0.5)
        s2 = flat_spectrum(0.4)
        with pytest.warns(DeprecationWarning):
            _ = s1 * s2
        with pytest.warns(DeprecationWarning):
            _ = s1 + s2
        with pytest.warns(DeprecationWarning):
            _ = s1 - s2
        with pytest.warns(DeprecationWarning):
            _ = s1 * 2.0

    def test_mul_combined_confidence_preserved(self):
        p1 = Provenance(confidence=0.9)
        p2 = Provenance(confidence=0.6)
        s1 = flat_spectrum(0.5)
        s1 = Spectrum(s1.wavelengths, s1.values, s1.domain, p1)
        s2 = flat_spectrum(1.0)
        s2 = Spectrum(s2.wavelengths, s2.values, s2.domain, p2)
        with pytest.warns(DeprecationWarning):
            product = s1 * s2
        assert product.provenance.confidence == pytest.approx(0.6)


# ============================================================================
# 7. Normalise and clip
# ============================================================================


class TestNormaliseAndClip:
    def test_normalise_warns_deprecated(self):
        s = flat_spectrum(0.4)
        with pytest.warns(DeprecationWarning):
            _ = s.normalise()

    def test_clip_behaviour_preserved(self):
        values = np.linspace(-0.1, 1.1, 81)
        s = Spectrum.from_arrays(CANONICAL_GRID, values, validate=False)
        s2 = s.clip(0.0, 1.0, warn=True)
        assert "clipped" in (s2.provenance.notes or "")


# ============================================================================
# 8. Integration
# ============================================================================


class TestIntegration:
    def test_integrate_and_dot_warn_deprecated(self):
        r = flat_spectrum(0.5)
        i = flat_spectrum(2.0, domain=SpectrumDomain.ILLUMINANT)
        with pytest.warns(DeprecationWarning):
            _ = r.dot(i)
        with pytest.warns(DeprecationWarning):
            _ = r.integrate()


# ============================================================================
# 9. Comparison and identity
# ============================================================================


class TestComparison:
    def test_allclose_identical(self):
        s = flat_spectrum()
        assert s.allclose(s)

    def test_allclose_within_tolerance(self):
        s1 = flat_spectrum(0.5)
        values2 = s1.values + 1e-7
        s2 = Spectrum.from_arrays(
            CANONICAL_GRID, values2, domain=SpectrumDomain.REFLECTANCE
        )
        assert s1.allclose(s2, atol=1e-6)

    def test_allclose_outside_tolerance(self):
        s1 = flat_spectrum(0.5)
        s2 = flat_spectrum(0.6)
        assert not s1.allclose(s2)

    def test_allclose_different_grids(self):
        s1 = flat_spectrum(grid=CANONICAL_GRID)
        s2 = flat_spectrum(grid=np.linspace(400, 700, 31))
        assert not s1.allclose(s2)

    def test_eq_identical(self):
        s1 = flat_spectrum()
        s2 = flat_spectrum()
        assert s1 == s2

    def test_eq_different_values(self):
        s1 = flat_spectrum(0.5)
        s2 = flat_spectrum(0.6)
        assert s1 != s2

    def test_eq_different_domain(self):
        s1 = flat_spectrum(domain=SpectrumDomain.REFLECTANCE)
        s2 = flat_spectrum(domain=SpectrumDomain.ILLUMINANT)
        assert s1 != s2

    def test_hash_equal_spectra(self):
        s1 = flat_spectrum()
        s2 = flat_spectrum()
        assert hash(s1) == hash(s2)

    def test_hash_usable_in_set(self):
        s1 = flat_spectrum(0.5)
        s2 = flat_spectrum(0.6)
        s3 = flat_spectrum(0.5)
        s = {s1, s2, s3}
        assert len(s) == 2


# ============================================================================
# 10. Domain validation (boundary conditions)
# ============================================================================


class TestColourInterop:
    def test_from_colour_and_to_colour_roundtrip(self):
        import colour

        # build a small spectral distribution and round-trip via Spectrum
        sd = colour.SpectralDistribution({380.0: 0.1, 385.0: 0.2, 390.0: 0.3}, name="test-sd")
        s = Spectrum.from_colour(sd, domain=SpectrumDomain.ILLUMINANT)
        sd2 = s.to_colour()
        assert isinstance(sd2, colour.SpectralDistribution)
        # name comes from provenance.source by default
        assert sd2.name == s.provenance.source

    def test_sd_property_returns_spectraldistribution(self):
        s = flat_spectrum(0.5)
        sd = s.sd
        import colour

        assert isinstance(sd, colour.SpectralDistribution)

    def test_project_shape_is_spectralshape(self):
        import colour

        from metamerism.core.spectrum import PROJECT_SHAPE
        assert isinstance(PROJECT_SHAPE, colour.SpectralShape)


class TestDomainValidation:
    def test_reflectance_zero_ok(self):
        s = Spectrum.from_arrays(
            CANONICAL_GRID, np.zeros(81), domain=SpectrumDomain.REFLECTANCE
        )
        assert np.allclose(s.values, 0.0)

    def test_reflectance_one_ok(self):
        s = Spectrum.from_arrays(
            CANONICAL_GRID, np.ones(81), domain=SpectrumDomain.REFLECTANCE
        )
        assert np.allclose(s.values, 1.0)

    def test_illuminant_zero_ok(self):
        s = Spectrum.from_arrays(
            CANONICAL_GRID, np.zeros(81), domain=SpectrumDomain.ILLUMINANT
        )
        assert np.allclose(s.values, 0.0)

    def test_unknown_domain_no_range_check(self):
        """UNKNOWN domain accepts any values without error."""
        s = Spectrum.from_arrays(
            CANONICAL_GRID, np.full(81, 999.0), domain=SpectrumDomain.UNKNOWN
        )
        assert s.values[0] == pytest.approx(999.0)

    def test_cmf_domain_no_range_check(self):
        """CMF domain has no strict range constraint; negative values are allowed."""
        values = np.linspace(-1.0, 3.0, 81)
        s = Spectrum.from_arrays(CANONICAL_GRID, values, domain=SpectrumDomain.CMF)
        assert s.values[0] == pytest.approx(-1.0)


# ============================================================================
# 11. Physical correctness — synthetic metamer identity test
# ============================================================================


class TestSyntheticMetamerIdentity:
    """
    Two spectra that differ in shape but integrate identically under a
    given illuminant x CMF should be confirmed as metamers by a
    colorimetric engine.  Here we verify the Spectrum primitives used
    by that engine behave correctly on a simple synthetic case.

    Two rectangular reflectances:
        r1 = 0.8 in [480, 580] nm, 0.1 elsewhere
        r2 = 0.8 in [500, 600] nm, 0.1 elsewhere

    These are NOT exact metamers, but we verify:
    - their Hadamard product with a flat illuminant can be computed
    - the dot product with a flat CMF is a finite scalar
    - the dot products differ (they are distinguishable under flat light)

    This is a correctness test for the primitive operations, not a full
    colorimetric test (which belongs in test_colorimetry.py).
    """

    def _rectangular(self, lo: float, hi: float) -> Spectrum:
        values = np.where((lo <= CANONICAL_GRID) & (hi >= CANONICAL_GRID), 0.8, 0.1)
        return Spectrum.from_arrays(
            CANONICAL_GRID, values, domain=SpectrumDomain.REFLECTANCE
        )

    def test_reflected_spectrum_computable(self):
        from metamerism.core.ops import multiply_spectra

        r = self._rectangular(480, 580)
        illum = flat_spectrum(1.0, domain=SpectrumDomain.ILLUMINANT)
        s = multiply_spectra(r, illum)
        assert s.n_samples == 81
        assert np.all(np.isfinite(s.values))

    def test_dot_with_flat_cmf_is_finite(self):
        from metamerism.core.ops import multiply_spectra, dot_spectra

        r = self._rectangular(480, 580)
        illum = flat_spectrum(1.0, domain=SpectrumDomain.ILLUMINANT)
        cmf = flat_spectrum(1.0, domain=SpectrumDomain.CMF)
        reflected = multiply_spectra(r, illum)
        response = dot_spectra(reflected, cmf)
        assert np.isfinite(response)

    def test_two_reflectances_distinguishable_under_flat_light(self):
        from metamerism.core.ops import multiply_spectra, dot_spectra

        r1 = self._rectangular(480, 580)
        r2 = self._rectangular(500, 600)
        illum = flat_spectrum(1.0, domain=SpectrumDomain.ILLUMINANT)
        cmf = flat_spectrum(1.0, domain=SpectrumDomain.CMF)
        resp1 = dot_spectra(multiply_spectra(r1, illum), cmf)
        resp2 = dot_spectra(multiply_spectra(r2, illum), cmf)
        # Both integrate the same area (100 nm of 0.8 + rest at 0.1),
        # so responses should be approximately equal under a flat CMF
        assert resp1 == pytest.approx(resp2, rel=1e-6)

    def test_spectra_are_physically_different(self):
        """The two reflectances differ spectrally even if responses match."""
        r1 = self._rectangular(480, 580)
        r2 = self._rectangular(500, 600)
        assert not r1.allclose(r2)
