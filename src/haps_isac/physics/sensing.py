"""Full-duplex sensing SINR, detection, and measurement generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from haps_isac.tracking.measurement_model import measurement_function, wrap_angle

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class SensingResult:
    sinr: float
    detected: bool
    measurement_covariance: FloatArray


def sensing_sinr(
    communication_power_w: float,
    sensing_power_w: float,
    communication_beam: ComplexArray,
    sensing_beam: ComplexArray,
    target_transmit_steering: ComplexArray,
    num_receive_antennas: int,
    echo_gain: float,
    noise_power_w: float,
    residual_self_interference_fraction: float,
) -> float:
    """Use the complete known HAPS waveform as sensing illumination."""

    if min(
        communication_power_w,
        sensing_power_w,
        echo_gain,
        noise_power_w,
        residual_self_interference_fraction,
    ) < 0.0:
        raise ValueError("sensing SINR inputs must be non-negative")
    if num_receive_antennas <= 0:
        raise ValueError("receive array size must be positive")
    if (
        communication_beam.shape != sensing_beam.shape
        or communication_beam.shape != target_transmit_steering.shape
    ):
        raise ValueError("sensing transmit vectors must have equal dimensions")

    communication_illumination = communication_power_w * abs(
        np.vdot(target_transmit_steering, communication_beam)
    ) ** 2
    sensing_illumination = sensing_power_w * abs(
        np.vdot(target_transmit_steering, sensing_beam)
    ) ** 2
    desired_power = (
        echo_gain
        * float(communication_illumination + sensing_illumination)
        * num_receive_antennas
    )
    residual_interference = residual_self_interference_fraction * (
        communication_power_w + sensing_power_w
    )
    return desired_power / (residual_interference + noise_power_w)


def measurement_covariance_from_sinr(
    sinr: float,
    sinr_floor: float,
    range_variance_scale: float,
    angle_variance_scale: float,
    doppler_variance_scale: float,
) -> FloatArray:
    if min(
        sinr,
        sinr_floor,
        range_variance_scale,
        angle_variance_scale,
        doppler_variance_scale,
    ) < 0.0 or sinr_floor == 0.0:
        raise ValueError("invalid sensing covariance parameters")
    quality = max(sinr, sinr_floor)
    return np.diag(
        [
            range_variance_scale / quality,
            angle_variance_scale / quality,
            doppler_variance_scale / quality,
        ]
    ).astype(np.float64)


def evaluate_sensing(
    sinr: float,
    threshold: float,
    sinr_floor: float,
    range_variance_scale: float,
    angle_variance_scale: float,
    doppler_variance_scale: float,
) -> SensingResult:
    covariance = measurement_covariance_from_sinr(
        sinr,
        sinr_floor,
        range_variance_scale,
        angle_variance_scale,
        doppler_variance_scale,
    )
    return SensingResult(
        sinr=float(sinr),
        detected=bool(sinr >= threshold),
        measurement_covariance=covariance,
    )


def sample_measurement(
    target_state: FloatArray,
    haps_position_m: FloatArray,
    covariance: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    ideal = measurement_function(target_state, haps_position_m)
    noise = rng.multivariate_normal(np.zeros(3, dtype=np.float64), covariance)
    measurement = ideal + noise
    measurement[1] = wrap_angle(float(measurement[1]))
    return measurement
