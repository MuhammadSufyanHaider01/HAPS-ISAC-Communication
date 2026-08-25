"""Channel and physical-unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from haps_isac.physics.channels import (
    effective_beam_gain,
    free_space_path_loss_db,
    large_scale_power_gain,
    rician_channel,
)
from haps_isac.units import db_to_linear, dbm_to_watts, linear_to_db, watts_to_dbm


def test_path_loss_and_gain_are_monotone_with_distance() -> None:
    frequency = 28.0e9
    assert free_space_path_loss_db(20_000.0, frequency) > free_space_path_loss_db(
        10_000.0, frequency
    )
    assert large_scale_power_gain(20_000.0, frequency) < large_scale_power_gain(
        10_000.0, frequency
    )


def test_power_conversions_round_trip() -> None:
    assert linear_to_db(db_to_linear(13.5)) == pytest.approx(13.5)
    assert watts_to_dbm(dbm_to_watts(37.0)) == pytest.approx(37.0)


def test_rician_channel_is_seed_reproducible_and_dimensionally_valid() -> None:
    transmitter = np.asarray([0.0, 0.0, 20_000.0])
    receiver = np.asarray([5_000.0, 2_000.0, 0.0])
    arguments = (
        transmitter,
        receiver,
        8,
        28.0e9,
        10.0,
        1.0,
        0.5,
    )
    first = rician_channel(*arguments, np.random.default_rng(11))
    second = rician_channel(*arguments, np.random.default_rng(11))
    assert first.shape == (8,)
    assert np.iscomplexobj(first)
    np.testing.assert_allclose(first, second)
    assert effective_beam_gain(first, np.zeros(8, dtype=np.complex128)) == 0.0
