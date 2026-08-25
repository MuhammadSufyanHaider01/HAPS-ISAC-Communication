"""Direct Version 1 aerial channel models."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

C_LIGHT_MPS = 299_792_458.0
FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


def euclidean_distance_m(position_a: FloatArray, position_b: FloatArray) -> float:
    distance = float(np.linalg.norm(np.asarray(position_a) - np.asarray(position_b)))
    if distance <= 0.0:
        raise ValueError("channel endpoints must be distinct")
    return distance


def free_space_path_loss_db(distance_m: float, carrier_frequency_hz: float) -> float:
    if distance_m <= 0.0 or carrier_frequency_hz <= 0.0:
        raise ValueError("distance and carrier frequency must be positive")
    return 20.0 * math.log10(
        4.0 * math.pi * carrier_frequency_hz * distance_m / C_LIGHT_MPS
    )


def large_scale_power_gain(
    distance_m: float,
    carrier_frequency_hz: float,
    additional_loss_db: float = 0.0,
    atmospheric_loss_db: float = 0.0,
) -> float:
    path_loss_db = (
        free_space_path_loss_db(distance_m, carrier_frequency_hz)
        + additional_loss_db
        + atmospheric_loss_db
    )
    return math.pow(10.0, -path_loss_db / 10.0)


def azimuth_rad(origin_m: FloatArray, destination_m: FloatArray) -> float:
    delta = np.asarray(destination_m, dtype=np.float64) - np.asarray(
        origin_m, dtype=np.float64
    )
    return math.atan2(float(delta[1]), float(delta[0]))


def ula_steering(
    num_antennas: int,
    azimuth: float,
    *,
    unit_norm: bool,
) -> ComplexArray:
    if num_antennas <= 0:
        raise ValueError("num_antennas must be positive")
    indices = np.arange(num_antennas, dtype=np.float64)
    vector = np.exp(1j * math.pi * indices * math.sin(azimuth)).astype(np.complex128)
    if unit_norm:
        vector /= math.sqrt(num_antennas)
    return np.asarray(vector, dtype=np.complex128)


def rician_channel(
    transmitter_position_m: FloatArray,
    receiver_position_m: FloatArray,
    num_transmit_antennas: int,
    carrier_frequency_hz: float,
    k_factor: float,
    additional_loss_db: float,
    atmospheric_loss_db: float,
    rng: np.random.Generator,
) -> ComplexArray:
    """Generate a Rician MISO channel with array gain retained in its norm."""

    if k_factor < 0.0:
        raise ValueError("Rician K factor must be non-negative")
    distance = euclidean_distance_m(transmitter_position_m, receiver_position_m)
    beta = large_scale_power_gain(
        distance,
        carrier_frequency_hz,
        additional_loss_db,
        atmospheric_loss_db,
    )
    los = ula_steering(
        num_transmit_antennas,
        azimuth_rad(transmitter_position_m, receiver_position_m),
        unit_norm=False,
    )
    nlos = (
        rng.standard_normal(num_transmit_antennas)
        + 1j * rng.standard_normal(num_transmit_antennas)
    ) / math.sqrt(2.0)
    channel = math.sqrt(beta) * (
        math.sqrt(k_factor / (1.0 + k_factor)) * los
        + math.sqrt(1.0 / (1.0 + k_factor)) * nlos
    )
    return np.asarray(channel, dtype=np.complex128)


def estimate_channel(
    true_channel: ComplexArray,
    error_variance: float,
    rng: np.random.Generator,
) -> tuple[ComplexArray, float]:
    """Produce the controller-visible estimate and total error variance."""

    if error_variance < 0.0:
        raise ValueError("CSI error variance must be non-negative")
    if error_variance == 0.0:
        return np.asarray(true_channel, dtype=np.complex128).copy(), 0.0
    shape = true_channel.shape
    error = math.sqrt(error_variance / 2.0) * (
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    )
    estimate = np.asarray(true_channel, dtype=np.complex128) + error
    return estimate.astype(np.complex128), float(np.prod(shape) * error_variance)


def effective_beam_gain(channel: ComplexArray, beam: ComplexArray) -> float:
    if channel.shape != beam.shape:
        raise ValueError("channel and beam dimensions must match")
    return float(abs(np.vdot(channel, beam)) ** 2)
