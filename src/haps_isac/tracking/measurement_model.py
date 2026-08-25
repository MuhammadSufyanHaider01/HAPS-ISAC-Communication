"""Quality-aware range, azimuth, and radial-velocity measurements."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def measurement_function(state: FloatArray, haps_position_m: FloatArray) -> FloatArray:
    target = np.asarray(state, dtype=np.float64)
    haps = np.asarray(haps_position_m, dtype=np.float64)
    if target.shape != (4,) or haps.shape != (3,):
        raise ValueError("invalid target or HAPS state dimensions")
    dx = target[0] - haps[0]
    dy = target[1] - haps[1]
    dz = -haps[2]
    slant_range = math.sqrt(dx**2 + dy**2 + dz**2)
    if slant_range <= 0.0:
        raise ValueError("target and HAPS positions cannot coincide")
    azimuth = math.atan2(dy, dx)
    radial_velocity = (dx * target[2] + dy * target[3]) / slant_range
    return np.asarray([slant_range, azimuth, radial_velocity], dtype=np.float64)


def measurement_jacobian(state: FloatArray, haps_position_m: FloatArray) -> FloatArray:
    target = np.asarray(state, dtype=np.float64)
    haps = np.asarray(haps_position_m, dtype=np.float64)
    dx = target[0] - haps[0]
    dy = target[1] - haps[1]
    dz = -haps[2]
    horizontal_squared = dx**2 + dy**2
    slant_squared = horizontal_squared + dz**2
    slant_range = math.sqrt(slant_squared)
    if horizontal_squared <= 1e-12 or slant_range <= 1e-12:
        raise ValueError("measurement Jacobian is singular at the HAPS nadir")
    velocity_projection = dx * target[2] + dy * target[3]
    return np.asarray(
        [
            [dx / slant_range, dy / slant_range, 0.0, 0.0],
            [-dy / horizontal_squared, dx / horizontal_squared, 0.0, 0.0],
            [
                target[2] / slant_range
                - velocity_projection * dx / slant_range**3,
                target[3] / slant_range
                - velocity_projection * dy / slant_range**3,
                dx / slant_range,
                dy / slant_range,
            ],
        ],
        dtype=np.float64,
    )


def wrap_angle(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
