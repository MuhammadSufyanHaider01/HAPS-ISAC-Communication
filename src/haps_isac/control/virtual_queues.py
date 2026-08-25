"""Long-term constraint virtual queues with explicit layout."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from haps_isac.config import ExperimentConfig

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class QueueSlices:
    aoli: slice
    reliability: slice
    secrecy: slice
    sensing: int
    uncertainty: int


def queue_slices(num_pairs: int) -> QueueSlices:
    users = 2 * num_pairs
    return QueueSlices(
        aoli=slice(0, users),
        reliability=slice(users, 2 * users),
        secrecy=slice(2 * users, 3 * users),
        sensing=3 * users,
        uncertainty=3 * users + 1,
    )


def constraint_increments(
    config: ExperimentConfig,
    next_aoli: FloatArray,
    delivered: BoolArray,
    secrecy_outage: BoolArray,
    scheduled_pair: int,
    sensing_detected: bool,
    covariance_trace: float,
) -> FloatArray:
    """Construct g_j values; secrecy outage is conditioned on scheduling."""

    num_pairs = config.system.num_noma_pairs
    users = 2 * num_pairs
    result = np.zeros(config.num_virtual_queues, dtype=np.float64)
    layout = queue_slices(num_pairs)
    result[layout.aoli] = (
        config.constraints.minimum_aoli_slots - np.asarray(next_aoli).reshape(users)
    ) / config.freshness.aoli_cap_slots
    result[layout.reliability] = (
        config.constraints.minimum_delivery_rate
        - np.asarray(delivered, dtype=np.float64).reshape(users)
    )

    secrecy = np.zeros((num_pairs, 2), dtype=np.float64)
    if scheduled_pair > 0:
        pair = scheduled_pair - 1
        secrecy[pair] = np.asarray(secrecy_outage[pair], dtype=np.float64) - (
            config.constraints.secrecy_outage_probability
        )
    result[layout.secrecy] = secrecy.reshape(users)
    result[layout.sensing] = (
        float(not sensing_detected) - config.constraints.sensing_outage_probability
    )
    result[layout.uncertainty] = (
        covariance_trace / config.constraints.maximum_covariance_trace - 1.0
    )
    return result


def update_virtual_queues(
    current: FloatArray,
    increments: FloatArray,
    clip_value: float,
) -> FloatArray:
    queues = np.asarray(current, dtype=np.float64)
    changes = np.asarray(increments, dtype=np.float64)
    if queues.shape != changes.shape:
        raise ValueError("queue and increment shapes must match")
    return np.clip(np.maximum(0.0, queues + changes), 0.0, clip_value)


def normalized_virtual_queues(
    queues: FloatArray,
    reference: float,
    clip_value: float,
) -> FloatArray:
    if reference <= 0.0 or clip_value <= 0.0:
        raise ValueError("queue normalization constants must be positive")
    clipped = np.clip(np.asarray(queues, dtype=np.float64), 0.0, clip_value)
    return np.asarray(
        np.clip(np.log1p(clipped) / np.log1p(reference), 0.0, 1.0), dtype=np.float64
    )
