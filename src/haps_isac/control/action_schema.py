"""Raw, transformed, completed, and executable action schemas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class RawPolicyAction:
    """Unconstrained action emitted by a future numerical policy."""

    pair: int
    ris_code: int
    continuous: FloatArray

    def __post_init__(self) -> None:
        continuous = np.asarray(self.continuous, dtype=np.float64)
        if continuous.shape != (7,):
            raise ValueError("continuous raw action must have shape (7,)")
        if not np.all(np.isfinite(continuous)):
            raise ValueError("continuous raw action must be finite")
        object.__setattr__(self, "continuous", continuous.copy())


@dataclass(frozen=True, slots=True)
class HighLevelAction:
    """Canonical bounded action in the documented order."""

    pair: int
    ris_code: int
    eta_haps: float
    eta_communication: float
    eta_near: float
    eta_jamming: float
    aav_heading_rad: float
    aav_speed_fraction: float
    eta_cpu: float

    def continuous_vector(self) -> FloatArray:
        return np.asarray(
            [
                self.eta_haps,
                self.eta_communication,
                self.eta_near,
                self.eta_jamming,
                self.aav_heading_rad,
                self.aav_speed_fraction,
                self.eta_cpu,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class CompletedPhysicalAction:
    """Complete Version 1 physical controls before safety repair."""

    high_level: HighLevelAction
    communication_power_w: float
    sensing_power_w: float
    near_power_w: float
    far_power_w: float
    cpu_frequency_hz: float
    communication_beam: ComplexArray
    sensing_beam: ComplexArray
    sensing_combiner: ComplexArray


@dataclass(frozen=True, slots=True)
class RepairLog:
    """Auditable description of deterministic action repair."""

    distance: float
    reasons: tuple[str, ...]
    fallback_used: bool
    hard_feasible: bool


@dataclass(frozen=True, slots=True)
class RepairedExecutableAction:
    """Physical action executed by the environment."""

    physical: CompletedPhysicalAction
    log: RepairLog
