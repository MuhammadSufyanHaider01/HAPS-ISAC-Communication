"""Seed-stream, replay, and clone-isolation tests."""

from __future__ import annotations

import numpy as np

from haps_isac.envs.haps_isac_env import HapsIsacEnv

ACTION = {
    "pair": 2,
    "ris_code": 0,
    "continuous": np.asarray([0.8, 0.65, 0.2, 0.0, 0.0, 0.0, 0.7]),
}


def test_fixed_seed_produces_identical_trajectories() -> None:
    first = HapsIsacEnv()
    second = HapsIsacEnv()
    first_observation, _ = first.reset(seed=17)
    second_observation, _ = second.reset(seed=17)
    for key in first_observation:
        np.testing.assert_array_equal(first_observation[key], second_observation[key])
    for _ in range(5):
        first_step = first.step(ACTION)
        second_step = second.step(ACTION)
        for key in first_step[0]:
            np.testing.assert_array_equal(first_step[0][key], second_step[0][key])
        assert first_step[1:4] == second_step[1:4]
        assert first_step[4]["stage_cost"] == second_step[4]["stage_cost"]
        np.testing.assert_array_equal(
            first_step[4]["delivery"],
            second_step[4]["delivery"],
        )


def test_candidate_evaluation_does_not_mutate_live_state() -> None:
    env = HapsIsacEnv()
    env.reset(seed=23)
    snapshot = env.state.clone()
    before_target = env.state.target_true_state.copy()
    first = env.evaluate_candidate(snapshot, ACTION, rollout_seed=99)
    second = env.evaluate_candidate(snapshot, ACTION, rollout_seed=99)
    assert env.state.slot == 0
    np.testing.assert_array_equal(env.state.target_true_state, before_target)
    assert first[1] == second[1]
    np.testing.assert_array_equal(first[0]["pair_tokens"], second[0]["pair_tokens"])
