"""Compact causal policy observation construction."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig
from haps_isac.control.virtual_queues import normalized_virtual_queues
from haps_isac.envs.state import SimulatorState
from haps_isac.tracking.ekf import stabilize_covariance

FloatArray = npt.NDArray[np.float64]
PAIR_TOKEN_DIM = 14
GLOBAL_FEATURE_DIM = 25
PREVIOUS_ACTION_DIM = 9


def _gain_feature(channel: np.ndarray) -> float:
    gain_db = 10.0 * math.log10(max(float(np.vdot(channel, channel).real), 1e-30))
    return float(np.clip((gain_db + 180.0) / 80.0, 0.0, 1.0))


def _covariance_cholesky_features(
    covariance: np.ndarray,
    covariance_reference: float,
) -> FloatArray:
    stable = stabilize_covariance(covariance)
    cholesky = np.linalg.cholesky(stable)
    entries = cholesky[np.tril_indices(4)]
    scale = math.sqrt(covariance_reference)
    return np.asarray(np.clip(entries / scale, -1.0, 1.0), dtype=np.float64)


def _previous_action_features(
    config: ExperimentConfig,
    state: SimulatorState,
) -> np.ndarray:
    if state.previous_action is None:
        return np.zeros(PREVIOUS_ACTION_DIM, dtype=np.float32)
    action = state.previous_action
    continuous = action.continuous_vector().copy()
    continuous[2] *= 2.0
    continuous[4] /= math.pi
    return np.asarray(
        [
            action.pair / config.system.num_noma_pairs,
            0.0,
            *continuous.tolist(),
        ],
        dtype=np.float32,
    )


def build_observation(
    config: ExperimentConfig,
    state: SimulatorState,
) -> dict[str, np.ndarray]:
    num_pairs = config.system.num_noma_pairs
    tokens = np.zeros((num_pairs, PAIR_TOKEN_DIM), dtype=np.float32)
    packet_reference = max(
        config.traffic.packet_bits_near,
        config.traffic.packet_bits_far,
    )
    packet_features = np.asarray(
        [
            config.traffic.packet_bits_near / packet_reference,
            config.traffic.packet_bits_far / packet_reference,
        ]
    )
    for pair in range(num_pairs):
        estimated_channels = state.channels.haps_ue_estimate[pair]
        gains = np.asarray([_gain_feature(channel) for channel in estimated_channels])
        uncertainty = np.clip(state.channels.csi_uncertainty[pair], 0.0, 1.0)
        tokens[pair] = np.asarray(
            [
                *(state.aoi[pair] / config.freshness.aoi_cap_slots),
                *(state.aoli[pair] / config.freshness.aoli_cap_slots),
                *gains,
                *uncertainty,
                0.0,
                0.0,
                *packet_features,
                math.tanh(float(state.last_sic_margin[pair])),
                min(
                    state.waiting_slots[pair] / config.freshness.aoi_cap_slots,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )

    target_scale = config.target.scenario_radius_m
    target_features = np.asarray(
        [
            state.available_mean[0] / target_scale,
            state.available_mean[1] / target_scale,
            state.available_mean[2] / 50.0,
            state.available_mean[3] / 50.0,
        ],
        dtype=np.float64,
    )
    covariance_features = _covariance_cholesky_features(
        state.available_covariance,
        config.objective.covariance_reference,
    )
    global_features = np.asarray(
        [
            state.aosi / config.freshness.aosi_cap_slots,
            *target_features.tolist(),
            *covariance_features.tolist(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    if global_features.shape != (GLOBAL_FEATURE_DIM,):
        raise AssertionError("global feature contract changed unexpectedly")
    return {
        "pair_tokens": np.clip(tokens, -1.0, 1.0),
        "pair_mask": np.ones(num_pairs, dtype=np.int8),
        "global_features": np.clip(global_features, -1.0, 1.0),
        "virtual_queues": normalized_virtual_queues(
            state.virtual_queues,
            config.constraints.queue_reference,
            config.constraints.queue_clip,
        ).astype(np.float32),
        "previous_action": _previous_action_features(config, state),
    }
