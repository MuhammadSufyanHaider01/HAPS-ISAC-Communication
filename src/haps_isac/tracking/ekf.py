"""Numerically stable extended Kalman filtering."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from haps_isac.tracking.measurement_model import (
    measurement_function,
    measurement_jacobian,
    wrap_angle,
)
from haps_isac.tracking.target_dynamics import predict_gaussian

FloatArray = npt.NDArray[np.float64]


def stabilize_covariance(covariance: FloatArray, eigenvalue_floor: float = 1e-9) -> FloatArray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, eigenvalue_floor)
    stabilized = (eigenvectors * clipped) @ eigenvectors.T
    return np.asarray(0.5 * (stabilized + stabilized.T), dtype=np.float64)


def predict(
    mean: FloatArray,
    covariance: FloatArray,
    slot_duration_s: float,
    acceleration_std_mps2: float,
) -> tuple[FloatArray, FloatArray]:
    predicted_mean, predicted_covariance = predict_gaussian(
        mean,
        covariance,
        slot_duration_s,
        acceleration_std_mps2,
    )
    return predicted_mean, stabilize_covariance(predicted_covariance)


def normalized_innovation_squared(
    prior_mean: FloatArray,
    prior_covariance: FloatArray,
    measurement: FloatArray,
    measurement_covariance: FloatArray,
    haps_position_m: FloatArray,
) -> float:
    """Return the EKF innovation consistency statistic for one measurement."""

    mean = np.asarray(prior_mean, dtype=np.float64)
    covariance = stabilize_covariance(np.asarray(prior_covariance, dtype=np.float64))
    observation = np.asarray(measurement, dtype=np.float64)
    observation_covariance = np.asarray(measurement_covariance, dtype=np.float64)
    if mean.shape != (4,) or covariance.shape != (4, 4):
        raise ValueError("invalid EKF prior dimensions")
    if observation.shape != (3,) or observation_covariance.shape != (3, 3):
        raise ValueError("invalid EKF measurement dimensions")
    jacobian = measurement_jacobian(mean, haps_position_m)
    innovation = observation - measurement_function(mean, haps_position_m)
    innovation[1] = wrap_angle(float(innovation[1]))
    innovation_covariance = (
        jacobian @ covariance @ jacobian.T + observation_covariance
    )
    return float(innovation @ np.linalg.solve(innovation_covariance, innovation))


def update(
    prior_mean: FloatArray,
    prior_covariance: FloatArray,
    measurement: FloatArray,
    measurement_covariance: FloatArray,
    haps_position_m: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Apply an EKF correction with angle wrapping and Joseph covariance form."""

    mean = np.asarray(prior_mean, dtype=np.float64)
    covariance = stabilize_covariance(np.asarray(prior_covariance, dtype=np.float64))
    observation = np.asarray(measurement, dtype=np.float64)
    observation_covariance = np.asarray(measurement_covariance, dtype=np.float64)
    if mean.shape != (4,) or covariance.shape != (4, 4):
        raise ValueError("invalid EKF prior dimensions")
    if observation.shape != (3,) or observation_covariance.shape != (3, 3):
        raise ValueError("invalid EKF measurement dimensions")

    jacobian = measurement_jacobian(mean, haps_position_m)
    innovation = observation - measurement_function(mean, haps_position_m)
    innovation[1] = wrap_angle(float(innovation[1]))
    innovation_covariance = (
        jacobian @ covariance @ jacobian.T + observation_covariance
    )
    gain = np.linalg.solve(
        innovation_covariance.T,
        (covariance @ jacobian.T).T,
    ).T
    posterior_mean = mean + gain @ innovation
    identity = np.eye(4, dtype=np.float64)
    residual_map = identity - gain @ jacobian
    posterior_covariance = (
        residual_map @ covariance @ residual_map.T
        + gain @ observation_covariance @ gain.T
    )
    return posterior_mean, stabilize_covariance(posterior_covariance)
