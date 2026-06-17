"""Tests for the project Spectrum metadata boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import colour
import numpy as np
import pytest

from metamerism.core.spectrum import (
    CANONICAL_GRID,
    PROJECT_SHAPE,
    DomainError,
    Provenance,
    Spectrum,
    SpectrumDomain,
)


def flat_spectrum(
    value: float = 0.5,
    domain: SpectrumDomain = SpectrumDomain.REFLECTANCE,
    grid: np.ndarray | None = None,
) -> Spectrum:
    wavelengths = CANONICAL_GRID if grid is None else grid
    values = np.full_like(wavelengths, value, dtype=np.float64)
    return Spectrum.from_arrays(wavelengths, values, domain=domain)


class TestProjectShape:
    def test_canonical_grid(self):
        assert CANONICAL_GRID[0] == pytest.approx(380.0)
        assert CANONICAL_GRID[-1] == pytest.approx(780.0)
        assert len(CANONICAL_GRID) == 81
        assert np.allclose(np.diff(CANONICAL_GRID), 5.0)

    def test_project_shape_is_colour_shape(self):
        assert isinstance(PROJECT_SHAPE, colour.SpectralShape)
        assert PROJECT_SHAPE.start == pytest.approx(380.0)
        assert PROJECT_SHAPE.end == pytest.approx(780.0)
        assert PROJECT_SHAPE.interval == pytest.approx(5.0)


class TestProvenance:
    def test_defaults(self):
        provenance = Provenance()
        assert provenance.source == "unknown"
        assert provenance.confidence == 1.0
        assert provenance.notes is None

    def test_with_note_appends(self):
        provenance = Provenance(source="test").with_note("loaded")
        updated = provenance.with_note("aligned")
        assert updated.notes == "loaded; aligned"

    def test_with_confidence_validates_range(self):
        provenance = Provenance().with_confidence(0.5)
        assert provenance.confidence == 0.5

        with pytest.raises(ValueError, match="confidence must be in"):
            provenance.with_confidence(1.5)

    def test_frozen(self):
        provenance = Provenance()
        with pytest.raises(FrozenInstanceError):
            provenance.__setattr__("confidence", 0.5)


class TestSpectrumConstruction:
    def test_from_arrays_stores_colour_spectral_distribution(self):
        spectrum = Spectrum.from_arrays(
            [400.0, 410.0, 420.0],
            [0.1, 0.2, 0.3],
            domain=SpectrumDomain.REFLECTANCE,
            provenance=Provenance(source="sample"),
        )

        assert isinstance(spectrum.sd, colour.SpectralDistribution)
        assert spectrum.sd.name == "sample"
        assert spectrum.domain is SpectrumDomain.REFLECTANCE
        assert spectrum.wavelengths.tolist() == [400.0, 410.0, 420.0]
        assert spectrum.values.tolist() == pytest.approx([0.1, 0.2, 0.3])

    def test_from_colour_copies_distribution_and_preserves_metadata(self):
        sd = colour.SpectralDistribution(
            {400.0: 0.1, 410.0: 0.2},
            name="colour source",
        )
        provenance = Provenance(source="project source")
        spectrum = Spectrum.from_colour(
            sd,
            domain=SpectrumDomain.ILLUMINANT,
            provenance=provenance,
        )

        assert spectrum.sd is not sd
        assert spectrum.sd.name == "project source"
        assert spectrum.provenance is provenance
        assert spectrum.domain is SpectrumDomain.ILLUMINANT

    def test_input_validation(self):
        with pytest.raises(ValueError, match="same length"):
            Spectrum.from_arrays([400.0, 410.0], [0.1])

        with pytest.raises(ValueError, match="monotonically increasing"):
            Spectrum.from_arrays([410.0, 400.0], [0.1, 0.2])

        with pytest.raises(ValueError, match="at least two"):
            Spectrum.from_arrays([400.0], [0.1])

    def test_domain_validation(self):
        with pytest.raises(DomainError, match="Negative values"):
            Spectrum.from_arrays(
                [400.0, 410.0],
                [-0.1, 0.1],
                domain=SpectrumDomain.REFLECTANCE,
            )

        with pytest.raises(DomainError, match="Values exceeding 1"):
            Spectrum.from_arrays(
                [400.0, 410.0],
                [0.1, 1.1],
                domain=SpectrumDomain.REFLECTANCE,
            )

        with pytest.raises(DomainError, match="Negative values"):
            Spectrum.from_arrays(
                [400.0, 410.0],
                [1.0, -0.1],
                domain=SpectrumDomain.ILLUMINANT,
            )

    def test_validate_false_skips_domain_validation(self):
        spectrum = Spectrum.from_arrays(
            [400.0, 410.0],
            [0.1, 1.5],
            domain=SpectrumDomain.REFLECTANCE,
            validate=False,
        )
        assert spectrum.values[1] == pytest.approx(1.5)


class TestSpectrumProperties:
    def test_sampling_properties(self):
        spectrum = flat_spectrum()
        assert spectrum.n_samples == 81
        assert spectrum.wl_min == pytest.approx(380.0)
        assert spectrum.wl_max == pytest.approx(780.0)
        assert spectrum.spacing == pytest.approx(5.0)
        assert spectrum.is_on_canonical_grid()

    def test_nonuniform_spacing(self):
        spectrum = Spectrum.from_arrays([400.0, 410.0, 435.0], [0.1, 0.2, 0.3])
        assert spectrum.spacing is None

    def test_repr_contains_metadata(self):
        spectrum = flat_spectrum(domain=SpectrumDomain.REFLECTANCE)
        text = repr(spectrum)
        assert "reflectance" in text
        assert "380" in text
        assert "780" in text


class TestColourScienceUsage:
    def test_alignment_uses_colour_directly(self):
        spectrum = Spectrum.from_arrays(
            np.linspace(400.0, 700.0, 31),
            np.full(31, 0.5),
        )

        aligned_sd = spectrum.sd.copy().align(PROJECT_SHAPE)

        assert isinstance(aligned_sd, colour.SpectralDistribution)
        assert np.asarray(aligned_sd.wavelengths).tolist() == pytest.approx(
            CANONICAL_GRID.tolist()
        )

    def test_colour_sd_to_xyz_direct_usage(self):
        reflectance = flat_spectrum(0.5)
        illuminant = colour.SDS_ILLUMINANTS["D65"].copy().align(PROJECT_SHAPE)
        cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].copy().align(
            PROJECT_SHAPE
        )
        reflected_sd = colour.SpectralDistribution(
            {
                float(wavelength): float(value)
                for wavelength, value in zip(
                    reflectance.wavelengths,
                    reflectance.values * illuminant.values,
                    strict=True,
                )
            }
        )

        xyz = colour.sd_to_XYZ(reflected_sd, cmfs=cmfs)

        assert xyz.shape == (3,)
        assert np.all(np.isfinite(xyz))


class TestComparison:
    def test_allclose(self):
        spectrum_a = flat_spectrum(0.5)
        spectrum_b = flat_spectrum(0.5 + 1e-7)
        spectrum_c = flat_spectrum(0.6)

        assert spectrum_a.allclose(spectrum_b, atol=1e-6)
        assert not spectrum_a.allclose(spectrum_c)

    def test_equality_and_hash(self):
        spectrum_a = flat_spectrum(0.5)
        spectrum_b = flat_spectrum(0.5)
        spectrum_c = flat_spectrum(0.6)

        assert spectrum_a == spectrum_b
        assert spectrum_a != spectrum_c
        assert len({spectrum_a, spectrum_b, spectrum_c}) == 2
