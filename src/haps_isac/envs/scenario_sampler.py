"""Reproducible topology, initialization, and channel sampling."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig
from haps_isac.envs.state import ChannelSnapshot
from haps_isac.physics.channels import estimate_channel, rician_channel

FloatArray = npt.NDArray[np.float64]


def sample_ue_positions(
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample fixed near/far users in pair-specific angular sectors."""

    num_pairs = config.system.num_noma_pairs
    positions = np.zeros((num_pairs, 2, 3), dtype=np.float64)
    near_bounds = config.topology.near_radius_range_m
    far_bounds = config.topology.far_radius_range_m
    sector_width = 2.0 * math.pi / num_pairs
    for pair in range(num_pairs):
        center = -math.pi + (pair + 0.5) * sector_width
        near_angle = center + rng.uniform(-0.3, 0.3) * sector_width
        far_angle = center + rng.uniform(-0.3, 0.3) * sector_width
        near_radius = rng.uniform(*near_bounds)
        far_radius = rng.uniform(*far_bounds)
        positions[pair, 0] = [
            near_radius * math.cos(near_angle),
            near_radius * math.sin(near_angle),
            0.0,
        ]
        positions[pair, 1] = [
            far_radius * math.cos(far_angle),
            far_radius * math.sin(far_angle),
            0.0,
        ]
    return positions


def initial_target_state(config: ExperimentConfig) -> FloatArray:
    return np.asarray(config.target.initial_state, dtype=np.float64)


def initial_target_covariance(config: ExperimentConfig) -> FloatArray:
    standard_deviations = np.asarray(config.target.initial_std, dtype=np.float64)
    return np.diag(standard_deviations**2)


def sample_channels(
    config: ExperimentConfig,
    ue_positions_m: FloatArray,
    target_state: FloatArray,
    rng: np.random.Generator,
) -> ChannelSnapshot:
    """Generate the true slot channels first, then controller-visible estimates."""

    num_pairs = config.system.num_noma_pairs
    num_antennas = config.haps.num_tx_antennas
    haps_position = np.asarray(config.haps.position_m, dtype=np.float64)
    true_ue = np.empty((num_pairs, 2, num_antennas), dtype=np.complex128)
    estimated_ue = np.empty_like(true_ue)
    uncertainty = np.zeros((num_pairs, 2), dtype=np.float64)

    error_variance = (
        config.channels.csi_error_variance if config.features.imperfect_csi else 0.0
    )
    for pair in range(num_pairs):
        for user in range(2):
            channel = rician_channel(
                haps_position,
                ue_positions_m[pair, user],
                num_antennas,
                config.system.carrier_frequency_hz,
                config.channels.rician_k_factor,
                config.channels.additional_loss_db,
                config.channels.atmospheric_loss_db,
                rng,
            )
            estimate, total_variance = estimate_channel(channel, error_variance, rng)
            true_ue[pair, user] = channel
            estimated_ue[pair, user] = estimate
            uncertainty[pair, user] = total_variance

    target_position = np.asarray(
        [target_state[0], target_state[1], 0.0],
        dtype=np.float64,
    )
    true_target = rician_channel(
        haps_position,
        target_position,
        num_antennas,
        config.system.carrier_frequency_hz,
        config.channels.rician_k_factor,
        config.channels.additional_loss_db,
        config.channels.atmospheric_loss_db,
        rng,
    )
    estimated_target, _ = estimate_channel(true_target, error_variance, rng)
    return ChannelSnapshot(
        haps_ue_true=true_ue,
        haps_ue_estimate=estimated_ue,
        haps_target_true=true_target,
        haps_target_estimate=estimated_target,
        csi_uncertainty=uncertainty,
    )
