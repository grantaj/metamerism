"""Sampled XYZ projection matrix and null-space basis.

The projection matrix A (shape 3 x n_wavelengths) maps a reflectance vector
r to approximate XYZ tristimulus values:

    XYZ ≈ A @ r

It is built to match colour.sd_to_XYZ exactly (Y=100 normalisation).
The null space of A spans all reflectance perturbations that are invisible
under the given illuminant/observer pair.

This module is an internal optimisation and geometry tool.
All public scoring must still use colour.sd_to_XYZ directly (see metamer.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import colour
import numpy as np
from numpy.typing import NDArray

from metamerism.core import PROJECT_SHAPE
from metamerism.search.metamer import ViewingCondition

# Named tolerance constants — centralised here so diagnostics can report them.

DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE = 1e-10
# SVD relative tolerance for determining numerical rank of the 3x81 projection
# matrix. Rank should always be 3 for a valid CMF/illuminant combination; a
# tighter value flags ill-conditioned conditions early.

DEFAULT_MATRIX_VS_COLOUR_XYZ_TOLERANCE = 0.1
# Maximum acceptable absolute difference between projection-matrix XYZ and
# direct colour.sd_to_XYZ, in XYZ units (Y_white = 100).


@dataclass(frozen=True)
class XYZProjection:
    """Sampled projection matrix for one viewing condition.

    Attributes:
        matrix: Shape (3, n_wavelengths). Row order: X, Y, Z.
        wavelengths_nm: 1-D array of wavelength sample positions.
        observer_name: Name of the CIE colour matching functions used.
        illuminant_name: Provenance name of the illuminant used.
        normalisation: Description of the Y=100 normalisation convention.
        singular_values: SVD singular values of the matrix (length 3).
    """

    matrix: NDArray[np.float64]
    wavelengths_nm: NDArray[np.float64]
    observer_name: str
    illuminant_name: str
    normalisation: str
    singular_values: NDArray[np.float64]

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "wavelengths_nm": self.wavelengths_nm.tolist(),
            "observer_name": self.observer_name,
            "illuminant_name": self.illuminant_name,
            "normalisation": self.normalisation,
            "singular_values": self.singular_values.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> XYZProjection:
        return cls(
            matrix=np.array(d["matrix"], dtype=np.float64),
            wavelengths_nm=np.array(d["wavelengths_nm"], dtype=np.float64),
            observer_name=d["observer_name"],
            illuminant_name=d["illuminant_name"],
            normalisation=d["normalisation"],
            singular_values=np.array(d["singular_values"], dtype=np.float64),
        )


@dataclass(frozen=True)
class NullSpace:
    """Null-space basis of a projection matrix.

    Attributes:
        basis: Shape (n_wavelengths, n_null). Columns are orthonormal basis
            vectors spanning ker(projection.matrix).
        relative_tolerance: SVD threshold used to determine numerical rank.
        rank: Numerical rank of the projection matrix (should be 3).
        n_null: Dimension of the null space (n_wavelengths - rank).
    """

    basis: NDArray[np.float64]
    relative_tolerance: float
    rank: int
    n_null: int

    def to_dict(self) -> dict:
        return {
            "basis": self.basis.tolist(),
            "relative_tolerance": self.relative_tolerance,
            "rank": self.rank,
            "n_null": self.n_null,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NullSpace:
        return cls(
            basis=np.array(d["basis"], dtype=np.float64),
            relative_tolerance=d["relative_tolerance"],
            rank=d["rank"],
            n_null=d["n_null"],
        )


def build_xyz_projection(
    condition: ViewingCondition,
    *,
    spectral_shape: colour.SpectralShape = PROJECT_SHAPE,
) -> XYZProjection:
    """Build the sampled XYZ projection matrix for a viewing condition.

    The matrix A satisfies XYZ ≈ A @ r for reflectance vector r sampled on
    spectral_shape, matching colour.sd_to_XYZ with Y=100 normalisation.

    Convention (Y=100 for perfect reflector):
        k  = 100 / sum_i( I(λ_i) * ybar(λ_i) * Δλ )
        A[row, i] = k * I(λ_i) * cbar_row(λ_i) * Δλ
    """
    wavelengths = np.asarray(spectral_shape.wavelengths, dtype=np.float64)
    delta_lambda = float(spectral_shape.interval)
    n = len(wavelengths)

    cmfs_msds = colour.colorimetry.MSDS_CMFS[condition.observer_name]
    cmfs_aligned = cmfs_msds.copy().align(spectral_shape)
    # shape (n, 3) — columns are X, Y, Z matching functions
    cmf_values = np.asarray(cmfs_aligned.values, dtype=np.float64)

    ill_sd = condition.illuminant.sd.copy()
    if ill_sd.shape != spectral_shape:
        ill_sd = ill_sd.align(spectral_shape)
    illuminant_values = np.asarray(ill_sd.values, dtype=np.float64)

    # Y=100 normalisation factor
    y_bar = cmf_values[:, 1]
    k = 100.0 / float(np.sum(illuminant_values * y_bar) * delta_lambda)

    # Build 3xn matrix; rows are X, Y, Z
    matrix = np.zeros((3, n), dtype=np.float64)
    for row in range(3):
        matrix[row] = k * illuminant_values * cmf_values[:, row] * delta_lambda

    # SVD for singular values (the null space is computed separately)
    _, sv, _ = np.linalg.svd(matrix, full_matrices=False)

    return XYZProjection(
        matrix=matrix,
        wavelengths_nm=wavelengths,
        observer_name=condition.observer_name,
        illuminant_name=condition.illuminant.provenance.source,
        normalisation="Y=100 for perfect reflector; "
        "k=100/sum(I*ybar*dlambda), A[row,i]=k*I*cbar_row*dlambda",
        singular_values=np.array(sv, dtype=np.float64),
    )


def compute_null_space(
    projection: XYZProjection,
    *,
    relative_tolerance: float | None = None,
) -> NullSpace:
    """Compute the null-space basis of a projection matrix via SVD.

    Args:
        projection: The XYZ projection to decompose.
        relative_tolerance: SVD threshold relative to the largest singular
            value. Defaults to DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE.

    Returns:
        NullSpace with orthonormal basis columns spanning ker(projection.matrix).
    """
    tol = (
        relative_tolerance
        if relative_tolerance is not None
        else DEFAULT_NULL_SPACE_RELATIVE_TOLERANCE
    )

    _, s, Vt = np.linalg.svd(projection.matrix, full_matrices=True)
    threshold = tol * s[0]
    rank = int(np.sum(s > threshold))
    # Null space: rows of Vt corresponding to near-zero singular values
    null_basis = Vt[rank:].T  # shape (n_wavelengths, n_null)

    return NullSpace(
        basis=np.array(null_basis, dtype=np.float64),
        relative_tolerance=tol,
        rank=rank,
        n_null=null_basis.shape[1],
    )


def projection_xyz(
    reflectance_values: NDArray[np.float64],
    projection: XYZProjection,
) -> NDArray[np.float64]:
    """Compute approximate XYZ using the projection matrix.

    For internal optimisation and numerical checks only.
    Authoritative XYZ must come from colour.sd_to_XYZ (see score_condition).

    Args:
        reflectance_values: 1-D array of length n_wavelengths.
        projection: The XYZ projection matrix.

    Returns:
        XYZ array of shape (3,).
    """
    r = np.asarray(reflectance_values, dtype=np.float64)
    if r.shape != (projection.matrix.shape[1],):
        raise ValueError(
            f"reflectance_values length {r.shape} does not match "
            f"projection n_wavelengths {projection.matrix.shape[1]}"
        )
    return np.array(projection.matrix @ r, dtype=np.float64)
