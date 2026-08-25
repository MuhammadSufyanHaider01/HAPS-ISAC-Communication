"""Causal environment and Gymnasium contract tests."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from haps_isac.envs.haps_isac_env import HapsIsacEnv

SENSING_ONLY_SLOW_CPU = {
    "pair": 0,
    "ris_code": 0,
    "continuous": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
}


def test_reset_and_step_match_declared_spaces() -> None:
    env = HapsIsacEnv()
    observation, _ = env.reset(seed=9)
    assert env.observation_space.contains(observation)
    next_observation, reward, terminated, truncated, info = env.step(
        SENSING_ONLY_SLOW_CPU
    )
    assert env.observation_space.contains(next_observation)
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert info["hard_feasible"]


def test_delayed_echo_cannot_update_same_or_next_slot_when_delay_is_two() -> None:
    env = HapsIsacEnv()
    env.reset(seed=7)
    _, _, _, _, first_info = env.step(SENSING_ONLY_SLOW_CPU)
    assert first_info["sensing_detected"]
    assert first_info["accepted_sensing_timestamp"] is None
    assert env.state.available_timestamp == 0
    _, _, _, _, second_info = env.step(SENSING_ONLY_SLOW_CPU)
    assert second_info["accepted_sensing_timestamp"] == 0
    assert env.state.slot == 2
    assert env.state.aosi == 2


def test_gymnasium_checker_accepts_environment() -> None:
    check_env(HapsIsacEnv(), skip_render_check=True)
