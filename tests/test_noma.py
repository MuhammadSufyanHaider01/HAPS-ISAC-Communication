"""NOMA, SIC, and rate tests."""

from __future__ import annotations

import math

import pytest

from haps_isac.physics.noma import (
    legitimate_noma_result,
    maximum_near_power_for_sic,
    packet_completed,
)


def test_noma_sinrs_match_hand_calculation() -> None:
    result = legitimate_noma_result(
        near_power_w=1.0,
        far_power_w=3.0,
        near_gain=2.0,
        far_gain=1.0,
        near_disturbance_w=1.0,
        far_disturbance_w=1.0,
        sic_residual_fraction=0.0,
        bandwidth_hz=1.0,
        sic_threshold=1.0,
    )
    assert result.far_user_sinr == pytest.approx(1.5)
    assert result.near_decodes_far_sinr == pytest.approx(2.0)
    assert result.near_user_sinr == pytest.approx(2.0)
    assert result.far_rate_bps == pytest.approx(math.log2(2.5))
    assert result.sic_margin == pytest.approx(1.0)


def test_more_disturbance_reduces_rates() -> None:
    low = legitimate_noma_result(1, 3, 2, 1, 0.1, 0.1, 0, 1, 1)
    high = legitimate_noma_result(1, 3, 2, 1, 10, 10, 0, 1, 1)
    assert low.near_rate_bps > high.near_rate_bps
    assert low.far_rate_bps > high.far_rate_bps


def test_closed_form_sic_boundary() -> None:
    maximum = maximum_near_power_for_sic(4.0, 2.0, 1.0, 1.0)
    result = legitimate_noma_result(
        maximum,
        4.0 - maximum,
        2.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
    )
    assert maximum == pytest.approx(1.75)
    assert result.sic_margin == pytest.approx(0.0, abs=1e-12)
    assert packet_completed(10.0, 0.1, 1.0)
