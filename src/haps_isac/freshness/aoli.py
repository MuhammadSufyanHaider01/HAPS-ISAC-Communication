"""Age-of-leaked-information recursion."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def update_aoli(
    current: FloatArray,
    intercepted: BoolArray,
    cap_slots: int,
) -> FloatArray:
    """Reset only complete fresh interceptions; larger age is more private."""

    ages = np.asarray(current, dtype=np.float64)
    events = np.asarray(intercepted, dtype=np.bool_)
    if ages.shape != events.shape:
        raise ValueError("AoLI and interception arrays must have the same shape")
    if cap_slots <= 1 or np.any(ages < 1.0):
        raise ValueError("invalid AoLI state or cap")
    return np.where(events, 1.0, np.minimum(ages + 1.0, float(cap_slots)))
