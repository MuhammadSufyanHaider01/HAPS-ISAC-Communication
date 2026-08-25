"""Ordered deterministic Version 1 feasibility repair and fallback."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from haps_isac.config import ExperimentConfig
from haps_isac.control.action_schema import (
    CompletedPhysicalAction,
    HighLevelAction,
    RepairedExecutableAction,
    RepairLog,
)
from haps_isac.envs.state import SimulatorState
from haps_isac.physics.channels import effective_beam_gain
from haps_isac.physics.noma import maximum_near_power_for_sic


def _normalized_vector(action: HighLevelAction, num_pairs: int) -> np.ndarray:
    return np.asarray(
        [
            action.pair / max(num_pairs, 1),
            action.eta_haps,
            action.eta_communication,
            2.0 * action.eta_near,
            action.eta_jamming,
            action.aav_heading_rad / math.pi,
            action.aav_speed_fraction,
            action.eta_cpu,
        ],
        dtype=np.float64,
    )


def _repair_distance(
    original: HighLevelAction,
    repaired: HighLevelAction,
    num_pairs: int,
) -> float:
    original_vector = _normalized_vector(original, num_pairs)
    repaired_vector = _normalized_vector(repaired, num_pairs)
    return float(np.mean(np.abs(original_vector - repaired_vector)))


def _fallback(
    config: ExperimentConfig,
    physical: CompletedPhysicalAction,
    reasons: list[str],
) -> RepairedExecutableAction:
    sensing_power = min(
        config.haps.max_power_w,
        max(config.constraints.minimum_sensing_power_w, 0.25 * config.haps.max_power_w),
    )
    high_level = HighLevelAction(
        pair=0,
        ris_code=0,
        eta_haps=sensing_power / config.haps.max_power_w,
        eta_communication=0.0,
        eta_near=0.0,
        eta_jamming=0.0,
        aav_heading_rad=0.0,
        aav_speed_fraction=0.0,
        eta_cpu=1.0,
    )
    fallback = replace(
        physical,
        high_level=high_level,
        communication_power_w=0.0,
        sensing_power_w=sensing_power,
        near_power_w=0.0,
        far_power_w=0.0,
        cpu_frequency_hz=config.haps.max_cpu_frequency_hz,
        communication_beam=physical.sensing_beam.copy(),
    )
    reasons.append("sensing_only_fallback")
    return RepairedExecutableAction(
        physical=fallback,
        log=RepairLog(
            distance=_repair_distance(
                physical.high_level,
                high_level,
                config.system.num_noma_pairs,
            ),
            reasons=tuple(reasons),
            fallback_used=True,
            hard_feasible=True,
        ),
    )


def repair_action(
    config: ExperimentConfig,
    state: SimulatorState,
    physical: CompletedPhysicalAction,
    receiver_noise_power_w: float,
) -> RepairedExecutableAction:
    """Repair sensing power and SIC, or return a guaranteed executable fallback."""

    reasons: list[str] = []
    action = physical.high_level
    if not 0 <= action.pair <= config.system.num_noma_pairs:
        reasons.append("invalid_pair")
        return _fallback(config, physical, reasons)

    communication_power = max(0.0, physical.communication_power_w)
    sensing_power = max(0.0, physical.sensing_power_w)
    total_power = communication_power + sensing_power
    if total_power > config.haps.max_power_w:
        scale = config.haps.max_power_w / total_power
        communication_power *= scale
        sensing_power *= scale
        reasons.append("total_power_projection")

    minimum_sensing = config.constraints.minimum_sensing_power_w
    if sensing_power < minimum_sensing:
        needed = minimum_sensing - sensing_power
        transferable = min(communication_power, needed)
        communication_power -= transferable
        sensing_power += transferable
        if sensing_power < minimum_sensing:
            sensing_power = minimum_sensing
        reasons.append("minimum_sensing_power")

    original_fraction = action.eta_near
    near_power = original_fraction * communication_power
    far_power = communication_power - near_power
    if action.pair > 0 and communication_power > 0.0:
        pair = action.pair - 1
        near_channel = state.channels.haps_ue_estimate[pair, 0]
        near_gain = effective_beam_gain(near_channel, physical.communication_beam)
        sensing_gain = effective_beam_gain(near_channel, physical.sensing_beam)
        disturbance = (
            config.channels.sensing_cancellation_fraction
            * sensing_power
            * sensing_gain
            + receiver_noise_power_w
        )
        maximum_near = maximum_near_power_for_sic(
            communication_power,
            near_gain,
            disturbance,
            config.constraints.sic_sinr_threshold,
        )
        if maximum_near < 0.0:
            reasons.append("sic_infeasible")
            return _fallback(config, physical, reasons)
        repaired_near = min(near_power, maximum_near, 0.5 * communication_power)
        if repaired_near + 1e-12 < near_power:
            reasons.append("near_power_sic_repair")
        near_power = max(0.0, repaired_near)
        far_power = communication_power - near_power

    total_power = communication_power + sensing_power
    repaired_fraction = near_power / communication_power if communication_power > 0 else 0.0
    repaired_high = replace(
        action,
        eta_haps=total_power / config.haps.max_power_w,
        eta_communication=communication_power / total_power if total_power > 0 else 0.0,
        eta_near=repaired_fraction,
        eta_jamming=0.0,
        aav_heading_rad=0.0,
        aav_speed_fraction=0.0,
    )
    repaired_physical = replace(
        physical,
        high_level=repaired_high,
        communication_power_w=communication_power,
        sensing_power_w=sensing_power,
        near_power_w=near_power,
        far_power_w=far_power,
    )
    hard_feasible = bool(
        total_power <= config.haps.max_power_w + 1e-9
        and sensing_power + 1e-9 >= minimum_sensing
        and 0.0 <= near_power <= far_power + 1e-9
        and config.haps.min_cpu_frequency_hz
        <= repaired_physical.cpu_frequency_hz
        <= config.haps.max_cpu_frequency_hz
        and np.all(np.isfinite(repaired_physical.communication_beam))
        and np.all(np.isfinite(repaired_physical.sensing_beam))
    )
    if not hard_feasible:
        reasons.append("final_invariant_failure")
        return _fallback(config, physical, reasons)
    return RepairedExecutableAction(
        physical=repaired_physical,
        log=RepairLog(
            distance=_repair_distance(
                physical.high_level,
                repaired_high,
                config.system.num_noma_pairs,
            ),
            reasons=tuple(reasons),
            fallback_used=False,
            hard_feasible=True,
        ),
    )
