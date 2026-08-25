"""Version 1 HAPS transmission and computation energy models."""

from __future__ import annotations

import math


def processing_delay_slots(
    required_cycles: float,
    cpu_frequency_hz: float,
    slot_duration_s: float,
) -> int:
    if required_cycles <= 0.0 or cpu_frequency_hz <= 0.0 or slot_duration_s <= 0.0:
        raise ValueError("processing-delay inputs must be positive")
    return max(1, math.ceil(required_cycles / (cpu_frequency_hz * slot_duration_s)))


def computation_energy_j(
    effective_capacitance: float,
    required_cycles: float,
    cpu_frequency_hz: float,
) -> float:
    if (
        effective_capacitance < 0.0
        or required_cycles < 0.0
        or cpu_frequency_hz < 0.0
    ):
        raise ValueError("computation-energy inputs must be non-negative")
    return effective_capacitance * required_cycles * cpu_frequency_hz**2


def haps_slot_energy_j(
    communication_power_w: float,
    sensing_power_w: float,
    slot_duration_s: float,
    compute_energy_j: float,
) -> float:
    if min(
        communication_power_w,
        sensing_power_w,
        slot_duration_s,
        compute_energy_j,
    ) < 0.0:
        raise ValueError("slot-energy inputs must be non-negative")
    return (
        slot_duration_s * (communication_power_w + sensing_power_w)
        + compute_energy_j
    )
