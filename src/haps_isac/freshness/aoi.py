"""Legitimate age-of-information recursion."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def update_aoi(current: FloatArray, delivered: BoolArray, cap_slots: int) -> FloatArray:
    """Reset generate-at-will deliveries to one slot and increment all others."""

    ages = np.asarray(current, dtype=np.float64)
    events = np.asarray(delivered, dtype=np.bool_)
    if ages.shape != events.shape:
        raise ValueError("AoI and delivery arrays must have the same shape")
    if cap_slots <= 1 or np.any(ages < 1.0):
        raise ValueError("invalid AoI state or cap")
    return np.where(events, 1.0, np.minimum(ages + 1.0, float(cap_slots)))
