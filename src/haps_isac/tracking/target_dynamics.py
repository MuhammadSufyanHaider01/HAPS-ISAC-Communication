"""Nearly constant-velocity target dynamics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def transition_matrix(slot_duration_s: float) -> FloatArray:
    if slot_duration_s <= 0.0:
        raise ValueError("slot duration must be positive")
    dt = slot_duration_s
    return np.asarray(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def process_covariance(slot_duration_s: float, acceleration_std_mps2: float) -> FloatArray:
    if slot_duration_s <= 0.0 or acceleration_std_mps2 < 0.0:
        raise ValueError("invalid process-noise parameters")
    dt = slot_duration_s
    variance = acceleration_std_mps2**2
    return variance * np.asarray(
        [
            [dt**4 / 4.0, 0.0, dt**3 / 2.0, 0.0],
            [0.0, dt**4 / 4.0, 0.0, dt**3 / 2.0],
            [dt**3 / 2.0, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2.0, 0.0, dt**2],
        ],
        dtype=np.float64,
    )


def propagate_true_state(
    state: FloatArray,
    slot_duration_s: float,
    acceleration_std_mps2: float,
    rng: np.random.Generator,
) -> FloatArray:
    state_array = np.asarray(state, dtype=np.float64)
    if state_array.shape != (4,):
        raise ValueError("target state must have shape (4,)")
    dt = slot_duration_s
    acceleration = rng.normal(0.0, acceleration_std_mps2, size=2)
    process_input = np.asarray(
        [
            0.5 * dt**2 * acceleration[0],
            0.5 * dt**2 * acceleration[1],
            dt * acceleration[0],
            dt * acceleration[1],
        ],
        dtype=np.float64,
    )
    return transition_matrix(dt) @ state_array + process_input


def predict_gaussian(
    mean: FloatArray,
    covariance: FloatArray,
    slot_duration_s: float,
    acceleration_std_mps2: float,
    steps: int = 1,
) -> tuple[FloatArray, FloatArray]:
    if steps < 0:
        raise ValueError("prediction steps must be non-negative")
    predicted_mean = np.asarray(mean, dtype=np.float64).copy()
    predicted_covariance = np.asarray(covariance, dtype=np.float64).copy()
    transition = transition_matrix(slot_duration_s)
    process_noise = process_covariance(slot_duration_s, acceleration_std_mps2)
    for _ in range(steps):
        predicted_mean = transition @ predicted_mean
        predicted_covariance = (
            transition @ predicted_covariance @ transition.T + process_noise
        )
    return predicted_mean, 0.5 * (predicted_covariance + predicted_covariance.T)
