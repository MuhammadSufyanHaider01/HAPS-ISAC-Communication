"""Eavesdropper reception, interception, and secrecy metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from haps_isac.physics.noma import packet_completed


@dataclass(frozen=True, slots=True)
class SecrecyResult:
    target_far_sinr: float
    target_near_sinr: float
    target_far_rate_bps: float
    target_near_rate_bps: float
    far_secrecy_rate_bps: float
    near_secrecy_rate_bps: float
    far_intercepted: bool
    near_intercepted: bool
    far_outage: bool
    near_outage: bool


def evaluate_eavesdropper(
    near_power_w: float,
    far_power_w: float,
    target_gain: float,
    target_disturbance_w: float,
    target_sic_residual_fraction: float,
    bandwidth_hz: float,
    payload_duration_s: float,
    near_packet_bits: float,
    far_packet_bits: float,
    legitimate_near_rate_bps: float,
    legitimate_far_rate_bps: float,
    secrecy_rate_target_bps: float,
) -> SecrecyResult:
    """Conservatively allow the target to apply the legitimate far-first SIC order."""

    values = (
        near_power_w,
        far_power_w,
        target_gain,
        target_disturbance_w,
        bandwidth_hz,
        payload_duration_s,
        near_packet_bits,
        far_packet_bits,
        legitimate_near_rate_bps,
        legitimate_far_rate_bps,
        secrecy_rate_target_bps,
    )
    if any(value < 0.0 for value in values):
        raise ValueError("secrecy inputs must be non-negative")
    if not 0.0 <= target_sic_residual_fraction <= 1.0:
        raise ValueError("target SIC residual must lie in [0, 1]")

    far_sinr = far_power_w * target_gain / (
        near_power_w * target_gain + target_disturbance_w
    )
    near_sinr = near_power_w * target_gain / (
        target_sic_residual_fraction * far_power_w * target_gain
        + target_disturbance_w
    )
    far_rate = bandwidth_hz * math.log2(1.0 + far_sinr)
    near_rate = bandwidth_hz * math.log2(1.0 + near_sinr)
    far_secrecy = max(0.0, legitimate_far_rate_bps - far_rate)
    near_secrecy = max(0.0, legitimate_near_rate_bps - near_rate)
    return SecrecyResult(
        target_far_sinr=far_sinr,
        target_near_sinr=near_sinr,
        target_far_rate_bps=far_rate,
        target_near_rate_bps=near_rate,
        far_secrecy_rate_bps=far_secrecy,
        near_secrecy_rate_bps=near_secrecy,
        far_intercepted=packet_completed(far_rate, payload_duration_s, far_packet_bits),
        near_intercepted=packet_completed(
            near_rate, payload_duration_s, near_packet_bits
        ),
        far_outage=far_secrecy < secrecy_rate_target_bps,
        near_outage=near_secrecy < secrecy_rate_target_bps,
    )
