"""Common-random-number stochastic rollout verification."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.envs.observation_builder import build_observation
from haps_isac.envs.state import SimulatorState
from haps_isac.teachers.base_teacher import VerificationConfig
from haps_isac.teachers.response_parser import ParsedCandidate


@dataclass(frozen=True, slots=True)
class RolloutRecord:
    candidate_index: int
    rollout_index: int
    rollout_seed: int
    steps_executed: int
    discounted_cost: float
    mean_stage_cost: float
    mean_aoi: float
    max_aoi: float
    mean_aoli: float
    mean_aosi: float
    delivery_rate: float
    secrecy_outage_rate: float
    sensing_detection_rate: float
    mean_tracking_mse: float
    mean_tracking_cov_trace: float
    total_energy_j: float
    mean_constraint_violation: float
    max_virtual_queue: float
    mean_repair_distance: float
    fallback_rate: float
    hard_feasible: bool
    trajectory: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CandidateRolloutSummary:
    candidate_index: int
    rollout_count: int
    mean_cost: float
    cost_std: float
    cost_standard_error: float
    median_cost: float
    cost_p05: float
    cost_p10: float
    cost_p90: float
    cost_p95: float
    worst_cost: float
    cvar_cost: float
    feasible_probability: float
    mean_constraint_violation: float
    mean_repair_distance: float
    fallback_rate: float
    risk_score: float
    rollouts: tuple[RolloutRecord, ...]


def common_rollout_seeds(state_id: str, master_seed: int, count: int) -> tuple[int, ...]:
    if master_seed < 0 or count <= 0:
        raise ValueError("master_seed must be non-negative and count must be positive")
    digest = hashlib.sha256(f"{master_seed}:{state_id}".encode()).digest()
    entropy = [master_seed, int.from_bytes(digest[:8], "big")]
    sequence = np.random.SeedSequence(entropy)
    return tuple(
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in sequence.spawn(count)
    )


def _constraint_violation(info: dict[str, Any]) -> float:
    keys = (
        "reliability_violation",
        "secrecy_violation",
        "sensing_violation",
        "uncertainty_violation",
        "power_violation",
        "mobility_violation",
    )
    return float(sum(max(0.0, float(info[key])) for key in keys))


def _step_log(step: int, info: dict[str, Any], env: HapsIsacEnv) -> dict[str, Any]:
    return {
        "step": step,
        "slot": int(info["slot"]),
        "stage_cost": float(info["stage_cost"]),
        "mean_aoi": float(info["mean_aoi"]),
        "mean_aoli": float(info["mean_aoli"]),
        "aosi": int(info["aosi"]),
        "energy_j": float(info["total_energy_j"]),
        "tracking_mse": float(info["tracking_mse"]),
        "tracking_cov_trace": float(info["tracking_cov_trace"]),
        "sensing_sinr": float(info["sensing_sinr"]),
        "sensing_detected": bool(info["sensing_detected"]),
        "secrecy_outage_count": int(np.sum(info["secrecy_outage"])),
        "delivery_count": int(np.sum(info["delivery"])),
        "constraint_violation": _constraint_violation(info),
        "repair_distance": float(info["repair_distance"]),
        "fallback_used": bool(info["fallback_used"]),
        "hard_feasible": bool(info["hard_feasible"]),
        "virtual_queues": env.state.virtual_queues.copy(),
    }


def run_candidate_rollout(
    env: HapsIsacEnv,
    state: SimulatorState,
    candidate: ParsedCandidate,
    rollout_index: int,
    rollout_seed: int,
    settings: VerificationConfig,
    retain_trajectory: bool = True,
) -> RolloutRecord:
    rollout_env = env.fork_from_state(state, rollout_seed)
    observation = build_observation(env.config, rollout_env.state)
    continuation = GreedyPolicy(env.config.system.num_noma_pairs)
    discount = 1.0
    discount_total = 0.0
    discounted_cost = 0.0
    logs: list[dict[str, Any]] = []
    for step in range(settings.rollout_horizon_slots):
        action: Any = candidate.action if step == 0 else continuation.act(observation)
        observation, _, terminated, truncated, info = rollout_env.step(action)
        stage_cost = float(info["stage_cost"])
        discounted_cost += discount * stage_cost
        discount_total += discount
        discount *= settings.discount_factor
        logs.append(_step_log(step, info, rollout_env))
        if terminated or truncated:
            break
    if not logs:
        raise RuntimeError("rollout produced no transitions")

    step_count = len(logs)
    deliveries = sum(int(item["delivery_count"]) for item in logs)
    secrecy_outages = sum(int(item["secrecy_outage_count"]) for item in logs)
    users = env.config.num_users
    trajectory = tuple(logs) if retain_trajectory else ()
    return RolloutRecord(
        candidate_index=candidate.candidate_index,
        rollout_index=rollout_index,
        rollout_seed=rollout_seed,
        steps_executed=step_count,
        discounted_cost=discounted_cost / discount_total,
        mean_stage_cost=float(np.mean([item["stage_cost"] for item in logs])),
        mean_aoi=float(np.mean([item["mean_aoi"] for item in logs])),
        max_aoi=float(max(item["mean_aoi"] for item in logs)),
        mean_aoli=float(np.mean([item["mean_aoli"] for item in logs])),
        mean_aosi=float(np.mean([item["aosi"] for item in logs])),
        delivery_rate=deliveries / (step_count * users),
        secrecy_outage_rate=secrecy_outages / (step_count * users),
        sensing_detection_rate=float(np.mean([item["sensing_detected"] for item in logs])),
        mean_tracking_mse=float(np.mean([item["tracking_mse"] for item in logs])),
        mean_tracking_cov_trace=float(np.mean([item["tracking_cov_trace"] for item in logs])),
        total_energy_j=float(sum(item["energy_j"] for item in logs)),
        mean_constraint_violation=float(np.mean([item["constraint_violation"] for item in logs])),
        max_virtual_queue=float(max(np.max(item["virtual_queues"]) for item in logs)),
        mean_repair_distance=float(np.mean([item["repair_distance"] for item in logs])),
        fallback_rate=float(np.mean([item["fallback_used"] for item in logs])),
        hard_feasible=all(bool(item["hard_feasible"]) for item in logs),
        trajectory=trajectory,
    )


def risk_score_from_rollouts(
    rollouts: tuple[RolloutRecord, ...],
    settings: VerificationConfig,
) -> float:
    """Recompute the complete risk score for an aligned rollout sample."""

    if not rollouts:
        raise ValueError("at least one rollout is required")
    costs = np.asarray([item.discounted_cost for item in rollouts], dtype=np.float64)
    threshold = float(np.quantile(costs, settings.cvar_alpha))
    cvar = float(np.mean(costs[costs >= threshold]))
    mean_cost = float(np.mean(costs))
    constraint_violation = float(np.mean([item.mean_constraint_violation for item in rollouts]))
    repair_distance = float(np.mean([item.mean_repair_distance for item in rollouts]))
    fallback_rate = float(np.mean([item.fallback_rate for item in rollouts]))
    feasible_probability = float(np.mean([item.hard_feasible for item in rollouts]))
    return float(
        mean_cost
        + settings.cvar_weight * (cvar - mean_cost)
        + settings.constraint_weight * constraint_violation
        + settings.repair_weight * repair_distance
        + settings.fallback_weight * fallback_rate
        + 1_000.0 * (1.0 - feasible_probability)
    )


def summarize_rollouts(
    candidate_index: int,
    rollouts: tuple[RolloutRecord, ...],
    settings: VerificationConfig,
) -> CandidateRolloutSummary:
    if not rollouts:
        raise ValueError("at least one rollout is required")
    costs = np.asarray([item.discounted_cost for item in rollouts], dtype=np.float64)
    threshold = float(np.quantile(costs, settings.cvar_alpha))
    tail = costs[costs >= threshold]
    cvar = float(np.mean(tail))
    mean_cost = float(np.mean(costs))
    constraint_violation = float(np.mean([item.mean_constraint_violation for item in rollouts]))
    repair_distance = float(np.mean([item.mean_repair_distance for item in rollouts]))
    fallback_rate = float(np.mean([item.fallback_rate for item in rollouts]))
    feasible_probability = float(np.mean([item.hard_feasible for item in rollouts]))
    risk_score = risk_score_from_rollouts(rollouts, settings)
    return CandidateRolloutSummary(
        candidate_index=candidate_index,
        rollout_count=len(rollouts),
        mean_cost=mean_cost,
        cost_std=float(np.std(costs, ddof=1)) if len(costs) > 1 else 0.0,
        cost_standard_error=(
            float(np.std(costs, ddof=1) / math.sqrt(len(costs))) if len(costs) > 1 else 0.0
        ),
        median_cost=float(np.median(costs)),
        cost_p05=float(np.quantile(costs, 0.05)),
        cost_p10=float(np.quantile(costs, 0.10)),
        cost_p90=float(np.quantile(costs, 0.90)),
        cost_p95=float(np.quantile(costs, 0.95)),
        worst_cost=float(np.max(costs)),
        cvar_cost=cvar,
        feasible_probability=feasible_probability,
        mean_constraint_violation=constraint_violation,
        mean_repair_distance=repair_distance,
        fallback_rate=fallback_rate,
        risk_score=float(risk_score),
        rollouts=rollouts,
    )


def verify_candidate(
    env: HapsIsacEnv,
    state: SimulatorState,
    state_id: str,
    candidate: ParsedCandidate,
    master_seed: int,
    settings: VerificationConfig,
    retain_trajectories: bool = True,
    rollout_count: int | None = None,
) -> CandidateRolloutSummary:
    effective_count = settings.monte_carlo_rollouts if rollout_count is None else rollout_count
    if effective_count <= 0 or effective_count > settings.max_monte_carlo_rollouts:
        raise ValueError("rollout_count must be within the configured adaptive range")
    seeds = common_rollout_seeds(
        state_id,
        master_seed,
        effective_count,
    )
    rollouts = tuple(
        run_candidate_rollout(
            env,
            state,
            candidate,
            rollout_index,
            rollout_seed,
            settings,
            retain_trajectory=retain_trajectories,
        )
        for rollout_index, rollout_seed in enumerate(seeds)
    )
    return summarize_rollouts(candidate.candidate_index, rollouts, settings)
