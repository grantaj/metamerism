import numpy as np

from metamerism.colorimetry.core import reflected_spectrum
from metamerism.spectra.core import SpectralCurve


def test_spectral_curve_shape_check() -> None:
    wavelengths = np.array([400.0, 500.0, 600.0])
    values = np.array([0.1, 0.2, 0.3])

    curve = SpectralCurve(wavelengths_nm=wavelengths, values=values)

    assert curve.wavelengths_nm.shape == curve.values.shape


def test_reflected_spectrum() -> None:
    reflectance = np.array([0.5, 0.25])
    illuminant = np.array([2.0, 4.0])

    result = reflected_spectrum(reflectance, illuminant)

    np.testing.assert_allclose(result, np.array([1.0, 1.0]))
