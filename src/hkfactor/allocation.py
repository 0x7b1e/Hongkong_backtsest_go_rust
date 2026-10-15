"""Long-only signal allocations above point-in-time base-weight floors."""

import numpy as np


def target_weights(
    base_weights: np.ndarray,
    values: np.ndarray,
    keep: float,
    direction: int,
) -> np.ndarray:
    """Allocate known signals by average rank above each stock's floor.

    All members receive their base floor. The entire remaining allocation
    is spread over known signals by positive direction-adjusted average
    ranks; tied scores receive equal overlay weights. No known signals
    means the complete base portfolio. Weights are long only.

    Args:
        base_weights: Point-in-time base weights, zero outside the universe.
        values: Known factor scores; NaN marks absent observations.
        keep: Per-stock target floor relative to its base weight.
        direction: Positive one favors larger scores; negative one reverses.

    Returns:
        Base weights when no scores are known, otherwise floor plus overlay.
    """
    base = np.asarray(base_weights, dtype=float)
    weights = base.copy()
    known = (base > 0) & np.isfinite(values)
    if not known.any() or keep == 1:
        return weights
    scores = np.asarray(values)[known] * direction
    _, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True
    )
    ends = np.cumsum(counts)
    ranks = (ends - (counts - 1) / 2)[inverse]
    weights = keep * base
    weights[known] += (1 - keep) * base.sum() * ranks / ranks.sum()
    return weights
