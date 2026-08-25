"""Freshness and processing-delay tests."""

from __future__ import annotations

import numpy as np

from haps_isac.freshness.aoi import update_aoi
from haps_isac.freshness.aoli import update_aoli
from haps_isac.freshness.aosi import update_aosi
from haps_isac.physics.energy import processing_delay_slots


def test_aoi_and_aoli_reset_only_on_their_own_events() -> None:
    ages = np.asarray([[3.0, 5.0], [9.0, 10.0]])
    events = np.asarray([[True, False], [False, True]])
    expected = np.asarray([[1.0, 6.0], [10.0, 1.0]])
    np.testing.assert_array_equal(update_aoi(ages, events, 10), expected)
    np.testing.assert_array_equal(update_aoli(ages, events, 10), expected)


def test_age_caps_and_aosi_includes_processing_delay() -> None:
    ages = np.asarray([[9.0, 10.0]])
    no_events = np.asarray([[False, False]])
    np.testing.assert_array_equal(
        update_aoi(ages, no_events, 10),
        np.asarray([[10.0, 10.0]]),
    )
    assert update_aosi(3, next_slot=8, accepted_timestamp=5, cap_slots=20) == 3
    assert update_aosi(3, next_slot=8, accepted_timestamp=None, cap_slots=20) == 4


def test_cpu_frequency_controls_integer_processing_delay() -> None:
    assert processing_delay_slots(2.0e8, 1.0e9, 0.1) == 2
    assert processing_delay_slots(2.0e8, 3.0e9, 0.1) == 1
