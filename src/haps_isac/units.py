"""Physical unit conversion helpers."""

from __future__ import annotations

import math


def db_to_linear(value_db: float) -> float:
    return math.pow(10.0, value_db / 10.0)


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        raise ValueError("a linear power ratio must be positive")
    return 10.0 * math.log10(value)


def dbm_to_watts(value_dbm: float) -> float:
    return math.pow(10.0, (value_dbm - 30.0) / 10.0)


def watts_to_dbm(value_w: float) -> float:
    if value_w <= 0.0:
        raise ValueError("power in watts must be positive")
    return 10.0 * math.log10(value_w) + 30.0


def thermal_noise_watts(
    bandwidth_hz: float,
    density_dbm_hz: float = -174.0,
    noise_figure_db: float = 0.0,
) -> float:
    if bandwidth_hz <= 0.0:
        raise ValueError("bandwidth must be positive")
    return dbm_to_watts(
        density_dbm_hz + 10.0 * math.log10(bandwidth_hz) + noise_figure_db
    )
