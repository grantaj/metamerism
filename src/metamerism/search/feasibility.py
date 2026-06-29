"""Reflectance-bound feasibility geometry.

Given a centre reflectance spectrum r_c and a direction d in the null space,
the feasible interval [alpha_min, alpha_max] is the largest scalar range such
that all samples of r_c + alpha * d remain in [lower, upper].
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

DEFAULT_REFLECTANCE_BOUND_TOLERANCE = 1e-4
# Allow numerical violations of [0, 1] bounds up to this amount before
# flagging a candidate as infeasible.


def feasible_step_interval(
    centre: NDArray[np.float64],
    direction: NDArray[np.float64],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> tuple[float, float] | None:
    """Compute the feasible scalar interval along a direction from a centre.

    Finds the largest [alpha_min, alpha_max] such that for all i:
        lower <= centre[i] + alpha * direction[i] <= upper

    For each sample i with direction[i] != 0 the active bound on alpha is:
        direction[i] > 0: alpha <= (upper - centre[i]) / direction[i]
                          alpha >= (lower - centre[i]) / direction[i]
        direction[i] < 0: alpha >= (upper - centre[i]) / direction[i]
                          alpha <= (lower - centre[i]) / direction[i]

    Args:
        centre: Base reflectance vector, shape (n,). Must satisfy bounds.
        direction: Step direction vector, shape (n,).
        lower: Lower reflectance bound (default 0.0).
        upper: Upper reflectance bound (default 1.0).

    Returns:
        (alpha_min, alpha_max) if the feasible interval has positive width,
        None if the direction is zero or the interval is degenerate.
    """
    centre = np.asarray(centre, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)

    if centre.shape != direction.shape:
        raise ValueError(
            f"centre and direction must have the same shape, "
            f"got {centre.shape} and {direction.shape}"
        )

    nonzero = direction != 0.0
    if not np.any(nonzero):
        return None

    d = direction[nonzero]
    c = centre[nonzero]

    # Upper-bound constraints: centre + alpha * d <= upper
    # Lower-bound constraints: centre + alpha * d >= lower
    alpha_upper = np.where(d > 0, (upper - c) / d, (lower - c) / d)
    alpha_lower = np.where(d > 0, (lower - c) / d, (upper - c) / d)

    alpha_max = float(np.min(alpha_upper))
    alpha_min = float(np.max(alpha_lower))

    if alpha_max <= alpha_min:
        return None

    return (alpha_min, alpha_max)


def max_bound_violation(
    reflectance: NDArray[np.float64],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> float:
    """Return the largest constraint violation in [lower, upper].

    A value of 0.0 means the reflectance is fully feasible.
    Positive values indicate the magnitude of the worst violation.
    """
    r = np.asarray(reflectance, dtype=np.float64)
    viol_low = np.maximum(0.0, lower - r)
    viol_high = np.maximum(0.0, r - upper)
    return float(np.max(np.maximum(viol_low, viol_high)))


def is_feasible(
    reflectance: NDArray[np.float64],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    tolerance: float = DEFAULT_REFLECTANCE_BOUND_TOLERANCE,
) -> bool:
    """Return True if all reflectance samples are within bounds plus tolerance."""
    return max_bound_violation(reflectance, lower=lower, upper=upper) <= tolerance
