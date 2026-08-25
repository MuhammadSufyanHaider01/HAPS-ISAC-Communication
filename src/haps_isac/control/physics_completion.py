"""Deterministic Version 1 high-level to physical-action completion."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig
from haps_isac.control.action_schema import CompletedPhysicalAction, HighLevelAction
from haps_isac.envs.state import SimulatorState
from haps_isac.physics.channels import azimuth_rad, ula_steering

ComplexArray = npt.NDArray[np.complex128]


def _normalize(vector: ComplexArray) -> ComplexArray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-15 or not math.isfinite(norm):
        raise ValueError("cannot normalize a zero or non-finite beam")
    return np.asarray(vector / norm, dtype=np.complex128)


def _target_beams(
    config: ExperimentConfig,
    state: SimulatorState,
) -> tuple[ComplexArray, ComplexArray]:
    haps_position = np.asarray(config.haps.position_m, dtype=np.float64)
    predicted_position = np.asarray(
        [state.available_mean[0], state.available_mean[1], 0.0],
        dtype=np.float64,
    )
    angle = azimuth_rad(haps_position, predicted_position)
    transmit = ula_steering(
        config.haps.num_tx_antennas,
        angle,
        unit_norm=True,
    )
    receive = ula_steering(
        config.haps.num_rx_antennas,
        angle,
        unit_norm=True,
    )
    return transmit, receive


def _communication_beam(state: SimulatorState, pair_index: int) -> ComplexArray:
    channels = state.channels.haps_ue_estimate[pair_index]
    urgency = np.maximum(state.aoi[pair_index], 1.0)
    weights = urgency / float(np.sum(urgency))
    candidate = weights[0] * channels[0] + weights[1] * channels[1]
    return _normalize(candidate)


def complete_action(
    config: ExperimentConfig,
    state: SimulatorState,
    action: HighLevelAction,
) -> CompletedPhysicalAction:
    """Construct powers, beams, combiner, and CPU allocation."""

    target_beam, sensing_combiner = _target_beams(config, state)
    total_power = action.eta_haps * config.haps.max_power_w
    if action.pair == 0:
        communication_power = 0.0
        sensing_power = total_power
        communication_beam = target_beam.copy()
    else:
        pair_index = action.pair - 1
        if not 0 <= pair_index < config.system.num_noma_pairs:
            pair_index = 0
        communication_power = action.eta_communication * total_power
        sensing_power = total_power - communication_power
        communication_beam = _communication_beam(state, pair_index)

    near_power = action.eta_near * communication_power
    far_power = communication_power - near_power
    cpu_frequency = config.haps.min_cpu_frequency_hz + action.eta_cpu * (
        config.haps.max_cpu_frequency_hz - config.haps.min_cpu_frequency_hz
    )
    return CompletedPhysicalAction(
        high_level=action,
        communication_power_w=float(communication_power),
        sensing_power_w=float(sensing_power),
        near_power_w=float(near_power),
        far_power_w=float(far_power),
        cpu_frequency_hz=float(cpu_frequency),
        communication_beam=communication_beam,
        sensing_beam=target_beam,
        sensing_combiner=sensing_combiner,
    )
