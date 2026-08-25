"""NOMA decoding, SIC, SINR, and rate calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NomaResult:
    far_user_sinr: float
    near_decodes_far_sinr: float
    near_user_sinr: float
    far_rate_bps: float
    near_rate_bps: float
    sic_margin: float


def _validate_nonnegative(**values: float) -> None:
    invalid = [name for name, value in values.items() if value < 0.0]
    if invalid:
        raise ValueError(f"non-negative values required: {', '.join(invalid)}")


def legitimate_noma_result(
    near_power_w: float,
    far_power_w: float,
    near_gain: float,
    far_gain: float,
    near_disturbance_w: float,
    far_disturbance_w: float,
    sic_residual_fraction: float,
    bandwidth_hz: float,
    sic_threshold: float,
) -> NomaResult:
    """Evaluate the fixed far-first NOMA decoding order."""

    _validate_nonnegative(
        near_power_w=near_power_w,
        far_power_w=far_power_w,
        near_gain=near_gain,
        far_gain=far_gain,
        near_disturbance_w=near_disturbance_w,
        far_disturbance_w=far_disturbance_w,
        sic_residual_fraction=sic_residual_fraction,
    )
    if bandwidth_hz <= 0.0 or not 0.0 <= sic_residual_fraction <= 1.0:
        raise ValueError("invalid bandwidth or SIC residual fraction")

    far_user_sinr = far_power_w * far_gain / (
        near_power_w * far_gain + far_disturbance_w
    )
    near_decodes_far_sinr = far_power_w * near_gain / (
        near_power_w * near_gain + near_disturbance_w
    )
    near_user_sinr = near_power_w * near_gain / (
        sic_residual_fraction * far_power_w * near_gain + near_disturbance_w
    )
    far_rate = bandwidth_hz * math.log2(
        1.0 + min(far_user_sinr, near_decodes_far_sinr)
    )
    near_rate = bandwidth_hz * math.log2(1.0 + near_user_sinr)
    return NomaResult(
        far_user_sinr=far_user_sinr,
        near_decodes_far_sinr=near_decodes_far_sinr,
        near_user_sinr=near_user_sinr,
        far_rate_bps=far_rate,
        near_rate_bps=near_rate,
        sic_margin=near_decodes_far_sinr - sic_threshold,
    )


def packet_completed(rate_bps: float, payload_duration_s: float, packet_bits: float) -> bool:
    _validate_nonnegative(
        rate_bps=rate_bps,
        payload_duration_s=payload_duration_s,
        packet_bits=packet_bits,
    )
    return payload_duration_s * rate_bps >= packet_bits


def maximum_near_power_for_sic(
    communication_power_w: float,
    near_gain: float,
    near_disturbance_w: float,
    sic_threshold: float,
) -> float:
    """Closed-form maximum near-stream power that preserves far-stream SIC."""

    _validate_nonnegative(
        communication_power_w=communication_power_w,
        near_gain=near_gain,
        near_disturbance_w=near_disturbance_w,
        sic_threshold=sic_threshold,
    )
    if near_gain == 0.0:
        return -math.inf
    numerator = communication_power_w * near_gain - sic_threshold * near_disturbance_w
    denominator = near_gain * (1.0 + sic_threshold)
    return numerator / denominator
