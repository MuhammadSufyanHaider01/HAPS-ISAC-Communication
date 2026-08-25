"""Action completion, repair, and fallback tests."""

from __future__ import annotations

import numpy as np
import pytest

from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.physics_completion import complete_action
from haps_isac.control.safety_repair import repair_action
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.units import thermal_noise_watts


def _noise(env: HapsIsacEnv) -> float:
    config = env.config
    return thermal_noise_watts(
        config.system.bandwidth_hz,
        config.channels.thermal_noise_density_dbm_hz,
        config.channels.receiver_noise_figure_db,
    )


def test_repair_enforces_power_sensing_and_noma_invariants() -> None:
    env = HapsIsacEnv()
    env.reset(seed=5)
    action = HighLevelAction(1, 0, 1.0, 0.99, 0.5, 0.0, 0.0, 0.0, 0.5)
    completed = complete_action(env.config, env.state, action)
    repaired = repair_action(env.config, env.state, completed, _noise(env))
    physical = repaired.physical
    assert physical.communication_power_w + physical.sensing_power_w <= (
        env.config.haps.max_power_w + 1e-9
    )
    assert physical.sensing_power_w >= env.config.constraints.minimum_sensing_power_w
    assert 0.0 <= physical.near_power_w <= physical.far_power_w
    assert repaired.log.hard_feasible
    assert np.linalg.norm(physical.communication_beam) == pytest.approx(1.0)


def test_invalid_pair_uses_documented_fallback() -> None:
    env = HapsIsacEnv()
    env.reset(seed=5)
    action = HighLevelAction(-1, 0, 0.1, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0)
    completed = complete_action(env.config, env.state, action)
    repaired = repair_action(env.config, env.state, completed, _noise(env))
    assert repaired.log.fallback_used
    assert repaired.physical.high_level.pair == 0
    assert repaired.physical.communication_power_w == 0.0


def test_random_repair_fuzz_is_always_executable() -> None:
    env = HapsIsacEnv()
    env.reset(seed=13)
    rng = np.random.default_rng(21)
    for _ in range(1_000):
        action = HighLevelAction(
            pair=int(rng.integers(0, env.config.system.num_noma_pairs + 1)),
            ris_code=0,
            eta_haps=float(rng.uniform(0.0, 1.0)),
            eta_communication=float(rng.uniform(0.0, 1.0)),
            eta_near=float(rng.uniform(0.0, 0.5)),
            eta_jamming=0.0,
            aav_heading_rad=0.0,
            aav_speed_fraction=0.0,
            eta_cpu=float(rng.uniform(0.0, 1.0)),
        )
        repaired = repair_action(
            env.config,
            env.state,
            complete_action(env.config, env.state, action),
            _noise(env),
        )
        assert repaired.log.hard_feasible
        assert np.all(np.isfinite(repaired.physical.sensing_beam))
