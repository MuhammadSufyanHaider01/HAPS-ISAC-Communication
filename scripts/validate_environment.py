"""Validate Version 1 simulator invariants, baselines, replay, and oracle."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

import numpy as np

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.baselines.optimization_policy import one_step_grid_oracle
from haps_isac.baselines.random_policy import RandomPolicy
from haps_isac.envs.haps_isac_env import HapsIsacEnv

PolicyAction = dict[str, object]
PolicyFunction = Callable[[dict[str, np.ndarray]], PolicyAction]


def _finite_observation(observation: dict[str, np.ndarray]) -> bool:
    return all(np.all(np.isfinite(value)) for value in observation.values())


def run_policy(
    config_path: str,
    policy_factory: Callable[[int], PolicyFunction],
    steps: int,
    seed: int,
) -> dict[str, float | int]:
    env = HapsIsacEnv(config_path)
    policy = policy_factory(seed)
    observation, _ = env.reset(seed=seed)
    totals = {
        "stage_cost": 0.0,
        "mean_aoi": 0.0,
        "mean_aoli": 0.0,
        "aosi": 0.0,
        "tracking_cov_trace": 0.0,
        "energy_j": 0.0,
        "repair_distance": 0.0,
        "fallbacks": 0,
        "hard_failures": 0,
        "detections": 0,
        "deliveries": 0,
    }
    episodes = 1
    for index in range(steps):
        if not _finite_observation(observation):
            raise RuntimeError(f"non-finite observation at transition {index}")
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(reward):
            raise RuntimeError(f"non-finite reward at transition {index}")
        if not env.observation_space.contains(observation):
            raise RuntimeError(f"observation contract violation at transition {index}")
        totals["stage_cost"] += float(info["stage_cost"])
        totals["mean_aoi"] += float(info["mean_aoi"])
        totals["mean_aoli"] += float(info["mean_aoli"])
        totals["aosi"] += float(info["aosi"])
        totals["tracking_cov_trace"] += float(info["tracking_cov_trace"])
        totals["energy_j"] += float(info["communication_energy_j"])
        totals["repair_distance"] += float(info["repair_distance"])
        totals["fallbacks"] += int(info["fallback_used"])
        totals["hard_failures"] += int(not info["hard_feasible"])
        totals["detections"] += int(info["sensing_detected"])
        totals["deliveries"] += int(np.sum(info["delivery"]))
        if terminated or truncated:
            episodes += 1
            observation, _ = env.reset(seed=seed + episodes)

    denominator = float(steps)
    return {
        "transitions": steps,
        "episodes": episodes,
        "mean_stage_cost": totals["stage_cost"] / denominator,
        "mean_aoi": totals["mean_aoi"] / denominator,
        "mean_aoli": totals["mean_aoli"] / denominator,
        "mean_aosi": totals["aosi"] / denominator,
        "mean_tracking_cov_trace": totals["tracking_cov_trace"] / denominator,
        "mean_energy_j": totals["energy_j"] / denominator,
        "mean_repair_distance": totals["repair_distance"] / denominator,
        "fallback_rate": totals["fallbacks"] / denominator,
        "hard_failure_rate": totals["hard_failures"] / denominator,
        "sensing_detection_rate": totals["detections"] / denominator,
        "deliveries_per_transition": totals["deliveries"] / denominator,
    }


def replay_check(config_path: str, seed: int) -> bool:
    first = HapsIsacEnv(config_path)
    second = HapsIsacEnv(config_path)
    first_observation, _ = first.reset(seed=seed)
    second_observation, _ = second.reset(seed=seed)
    action = {
        "pair": 1,
        "ris_code": 0,
        "continuous": np.asarray([0.8, 0.6, 0.2, 0.0, 0.0, 0.0, 0.7]),
    }
    for _ in range(20):
        for key in first_observation:
            if not np.array_equal(first_observation[key], second_observation[key]):
                return False
        first_step = first.step(action)
        second_step = second.step(action)
        if first_step[1] != second_step[1]:
            return False
        first_observation, second_observation = first_step[0], second_step[0]
    return True


def clone_check(config_path: str, seed: int) -> bool:
    env = HapsIsacEnv(config_path)
    env.reset(seed=seed)
    before = env.state.clone()
    action = {
        "pair": 2,
        "ris_code": 0,
        "continuous": np.asarray([0.9, 0.7, 0.2, 0.0, 0.0, 0.0, 1.0]),
    }
    first = env.evaluate_candidate(before, action, rollout_seed=seed + 100)
    second = env.evaluate_candidate(before, action, rollout_seed=seed + 100)
    return bool(
        env.state.slot == before.slot
        and np.array_equal(env.state.target_true_state, before.target_true_state)
        and first[1] == second[1]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/system_v1.yaml")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--skip-oracle", action="store_true")
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        raise ValueError("--steps must be positive")

    reference_env = HapsIsacEnv(arguments.config)
    pairs = reference_env.config.system.num_noma_pairs
    report: dict[str, Any] = {
        "config": arguments.config,
        "replay_identical": replay_check(arguments.config, arguments.seed),
        "candidate_clone_isolated": clone_check(arguments.config, arguments.seed),
        "random": run_policy(
            arguments.config,
            lambda seed: RandomPolicy(pairs, seed).act,
            arguments.steps,
            arguments.seed,
        ),
        "greedy": run_policy(
            arguments.config,
            lambda _seed: GreedyPolicy(pairs).act,
            arguments.steps,
            arguments.seed,
        ),
    }
    if not arguments.skip_oracle:
        reference_env.reset(seed=arguments.seed)
        oracle = one_step_grid_oracle(reference_env, rollout_seed=arguments.seed + 1)
        report["one_step_oracle"] = {
            "stage_cost": oracle.stage_cost,
            "repair_distance": oracle.repair_distance,
            "hard_feasible": oracle.hard_feasible,
            "candidates_evaluated": oracle.candidates_evaluated,
            "pair": oracle.action.pair,
        }
    if not report["replay_identical"] or not report["candidate_clone_isolated"]:
        raise RuntimeError("deterministic replay or candidate isolation failed")
    if report["random"]["hard_failure_rate"] != 0.0:
        raise RuntimeError("random stress test found a hard feasibility failure")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
