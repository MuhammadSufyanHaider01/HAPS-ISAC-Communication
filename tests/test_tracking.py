"""Target dynamics, measurement, and EKF tests."""

from __future__ import annotations

import numpy as np

from haps_isac.tracking.ekf import predict, update
from haps_isac.tracking.measurement_model import (
    measurement_function,
    measurement_jacobian,
)
from haps_isac.tracking.target_dynamics import propagate_true_state


def test_measurement_jacobian_matches_finite_difference() -> None:
    state = np.asarray([15_000.0, 5_000.0, 18.0, 6.0])
    haps = np.asarray([0.0, 0.0, 20_000.0])
    analytical = measurement_jacobian(state, haps)
    numerical = np.zeros_like(analytical)
    epsilon = 1e-3
    for column in range(4):
        perturbation = np.zeros(4)
        perturbation[column] = epsilon
        numerical[:, column] = (
            measurement_function(state + perturbation, haps)
            - measurement_function(state - perturbation, haps)
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytical, numerical, rtol=2e-5, atol=2e-7)


def test_prediction_increases_uncertainty_and_update_is_psd() -> None:
    mean = np.asarray([15_000.0, 5_000.0, 18.0, 6.0])
    covariance = np.diag([100.0, 100.0, 4.0, 4.0])
    predicted_mean, predicted_covariance = predict(mean, covariance, 0.1, 1.0)
    assert np.trace(predicted_covariance) > np.trace(covariance)
    haps = np.asarray([0.0, 0.0, 20_000.0])
    measurement = measurement_function(predicted_mean, haps)
    posterior_mean, posterior_covariance = update(
        predicted_mean,
        predicted_covariance,
        measurement,
        np.diag([1.0, 1e-6, 0.1]),
        haps,
    )
    assert np.all(np.linalg.eigvalsh(posterior_covariance) >= 0.0)
    assert np.trace(posterior_covariance) < np.trace(predicted_covariance)
    np.testing.assert_allclose(posterior_mean, predicted_mean)


def test_zero_acceleration_propagation_matches_constant_velocity() -> None:
    state = np.asarray([10.0, 20.0, 3.0, -2.0])
    propagated = propagate_true_state(
        state,
        0.5,
        0.0,
        np.random.default_rng(4),
    )
    np.testing.assert_allclose(propagated, [11.5, 19.0, 3.0, -2.0])
