"""Normalized Version 1 stage-cost construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class StageCost:
    total: float
    normalized_aoi: float
    normalized_aosi: float
    normalized_uncertainty: float
    normalized_energy: float
    energy_j: float

    @property
    def reward(self) -> float:
        return -self.total


def build_stage_cost(
    config: ExperimentConfig,
    aoi: FloatArray,
    aosi: int,
    covariance_trace: float,
    energy_j: float,
) -> StageCost:
    if covariance_trace < 0.0 or energy_j < 0.0:
        raise ValueError("uncertainty and energy must be non-negative")
    normalized_aoi = float(np.mean(aoi) / config.freshness.aoi_cap_slots)
    normalized_aosi = float(aosi / config.freshness.aosi_cap_slots)
    normalized_uncertainty = covariance_trace / config.objective.covariance_reference
    normalized_energy = energy_j / config.objective.energy_reference_j
    total = (
        config.objective.weight_aoi * normalized_aoi
        + config.objective.weight_aosi * normalized_aosi
        + config.objective.weight_uncertainty * normalized_uncertainty
        + config.objective.weight_energy * normalized_energy
    )
    return StageCost(
        total=float(total),
        normalized_aoi=normalized_aoi,
        normalized_aosi=normalized_aosi,
        normalized_uncertainty=normalized_uncertainty,
        normalized_energy=normalized_energy,
        energy_j=energy_j,
    )
