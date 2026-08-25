"""Internal causal simulator state."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from haps_isac.control.action_schema import HighLevelAction

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ChannelSnapshot:
    """True slot channels and the estimates visible before action selection."""

    haps_ue_true: ComplexArray
    haps_ue_estimate: ComplexArray
    haps_target_true: ComplexArray
    haps_target_estimate: ComplexArray
    csi_uncertainty: FloatArray


@dataclass(frozen=True, slots=True)
class SensingJob:
    """Posterior produced from a timestamped echo but released after CPU delay."""

    timestamp: int
    ready_slot: int
    posterior_mean: FloatArray
    posterior_covariance: FloatArray


@dataclass(frozen=True, slots=True)
class SimulatorState:
    """Complete Markov state, including information hidden from the policy."""

    slot: int
    ue_positions_m: FloatArray
    target_true_state: FloatArray
    channels: ChannelSnapshot
    hidden_filter_mean: FloatArray
    hidden_filter_covariance: FloatArray
    available_source_mean: FloatArray
    available_source_covariance: FloatArray
    available_timestamp: int
    available_mean: FloatArray
    available_covariance: FloatArray
    pending_sensing_jobs: tuple[SensingJob, ...]
    aoi: FloatArray
    aoli: FloatArray
    aosi: int
    waiting_slots: FloatArray
    virtual_queues: FloatArray
    last_rates_bps: FloatArray
    last_delivery: BoolArray
    last_sic_margin: FloatArray
    previous_action: HighLevelAction | None
    last_repair_distance: float
    last_fallback_used: bool

    def clone(self) -> SimulatorState:
        return copy.deepcopy(self)
