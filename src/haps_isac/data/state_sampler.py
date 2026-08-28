"""Deterministic stratified state plans for sharded teacher generation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.baselines.random_policy import RandomPolicy
from haps_isac.config import ExperimentConfig
from haps_isac.control.virtual_queues import queue_slices
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.envs.observation_builder import build_observation
from haps_isac.envs.state import SimulatorState
from haps_isac.teachers.base_teacher import DatasetSamplingConfig

STATE_CATEGORIES = (
    "ordinary",
    "freshness_stress",
    "sensing_stress",
    "secrecy_stress",
    "boundary_rare",
)
SPLITS = ("train", "validation", "test")


def _stable_seed(master_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{master_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def exact_assignments(
    total: int,
    fractions: Mapping[str, float],
    master_seed: int,
    label: str,
) -> tuple[str, ...]:
    """Return shuffled labels with deterministic largest-remainder quotas."""

    if total <= 0:
        raise ValueError("total must be positive")
    if not fractions or any(value < 0.0 for value in fractions.values()):
        raise ValueError("fractions must be non-empty and non-negative")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to one")
    names = tuple(fractions)
    raw = {name: total * fractions[name] for name in names}
    quotas = {name: int(np.floor(raw[name])) for name in names}
    remaining = total - sum(quotas.values())
    tie_breaks = {name: _stable_seed(master_seed, f"{label}:remainder:{name}") for name in names}
    remainder_order = sorted(
        names,
        key=lambda name: (-(raw[name] - quotas[name]), tie_breaks[name], name),
    )
    for name in remainder_order[:remaining]:
        quotas[name] += 1
    assignments = [name for name in names for _ in range(quotas[name])]
    generator = np.random.default_rng(_stable_seed(master_seed, f"{label}:shuffle"))
    generator.shuffle(assignments)
    return tuple(assignments)


def shard_bounds(total: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    if total <= 0:
        raise ValueError("total must be positive")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be within shard count")
    base, remainder = divmod(total, shard_count)
    start = shard_index * base + min(shard_index, remainder)
    stop = start + base + int(shard_index < remainder)
    return start, stop


@dataclass(frozen=True, slots=True)
class SampledState:
    state: SimulatorState
    observation: dict[str, np.ndarray]
    global_state_index: int
    scenario_id: str
    episode_id: int
    split: str
    category: str
    sampling_seed: int


def _scaled_covariance(state: SimulatorState, target_trace: float) -> npt.NDArray[np.float64]:
    covariance = np.asarray(state.available_covariance, dtype=np.float64)
    trace = max(float(np.trace(covariance)), float(np.finfo(np.float64).tiny))
    return np.asarray(covariance * (float(target_trace) / trace), dtype=np.float64)


def _apply_stress(
    config: ExperimentConfig,
    state: SimulatorState,
    category: str,
) -> SimulatorState:
    if category == "ordinary":
        return state
    queues = state.virtual_queues.copy()
    layout = queue_slices(config.system.num_noma_pairs)
    aoi = state.aoi
    waiting_slots = state.waiting_slots
    hidden_filter_covariance = state.hidden_filter_covariance
    available_source_covariance = state.available_source_covariance
    available_covariance = state.available_covariance
    pending_sensing_jobs = state.pending_sensing_jobs
    aosi = state.aosi

    if category in {"freshness_stress", "boundary_rare"}:
        fraction = 0.75 if category == "freshness_stress" else 0.95
        target_aoi = max(1, int(round(fraction * config.freshness.aoi_cap_slots)))
        offsets = np.arange(state.aoi.size, dtype=np.float64).reshape(state.aoi.shape) % 5
        aoi = np.minimum(
            config.freshness.aoi_cap_slots,
            np.maximum(state.aoi, target_aoi - offsets),
        )
        waiting_slots = np.maximum(state.waiting_slots, target_aoi / 2.0)
        queues[layout.reliability] = np.maximum(
            queues[layout.reliability],
            config.constraints.queue_reference * fraction,
        )

    if category in {"sensing_stress", "boundary_rare"}:
        fraction = 0.90 if category == "sensing_stress" else 1.10
        covariance = _scaled_covariance(
            state,
            config.constraints.maximum_covariance_trace * fraction,
        )
        target_aosi = max(1, int(round(0.75 * config.freshness.aosi_cap_slots)))
        if category == "boundary_rare":
            target_aosi = max(1, config.freshness.aosi_cap_slots - 1)
        hidden_filter_covariance = covariance.copy()
        available_source_covariance = covariance.copy()
        available_covariance = covariance.copy()
        pending_sensing_jobs = ()
        aosi = max(state.aosi, target_aosi)
        queues[layout.sensing] = max(
            queues[layout.sensing], config.constraints.queue_reference * fraction
        )
        queues[layout.uncertainty] = max(
            queues[layout.uncertainty], config.constraints.queue_reference * fraction
        )

    if category in {"secrecy_stress", "boundary_rare"}:
        fraction = 0.75 if category == "secrecy_stress" else 0.95
        queues[layout.secrecy] = np.maximum(
            queues[layout.secrecy], config.constraints.queue_reference * fraction
        )

    if category == "boundary_rare":
        queues[:] = np.maximum(
            queues,
            min(config.constraints.queue_clip, 0.95 * config.constraints.queue_reference),
        )
    return replace(
        state,
        hidden_filter_covariance=hidden_filter_covariance,
        available_source_covariance=available_source_covariance,
        available_covariance=available_covariance,
        pending_sensing_jobs=pending_sensing_jobs,
        aoi=aoi,
        aosi=aosi,
        waiting_slots=waiting_slots,
        virtual_queues=queues,
    )


class StratifiedStateSampler:
    """Build independently reproducible states from one global index plan."""

    def __init__(
        self,
        system_config: ExperimentConfig,
        sampling_config: DatasetSamplingConfig,
        total_states: int,
        master_seed: int,
    ) -> None:
        self.system_config = system_config
        self.sampling_config = sampling_config
        self.total_states = total_states
        self.master_seed = master_seed
        self.categories = exact_assignments(
            total_states,
            sampling_config.state_fractions.model_dump(),
            master_seed,
            "state-category",
        )
        self.splits = exact_assignments(
            total_states,
            sampling_config.split_fractions.model_dump(),
            master_seed,
            "scenario-split",
        )

    def sample(self, env: HapsIsacEnv, global_state_index: int) -> SampledState:
        if not 0 <= global_state_index < self.total_states:
            raise IndexError("global state index is outside the sampling plan")
        sampling_seed = _stable_seed(self.master_seed, f"state:{global_state_index:08d}")
        observation, _ = env.reset(seed=sampling_seed)
        maximum_rollin = self.sampling_config.maximum_rollin_slots
        rollin_slots = _stable_seed(self.master_seed, f"rollin:{global_state_index:08d}") % (
            maximum_rollin + 1
        )
        if global_state_index % 2 == 0:
            policy: GreedyPolicy | RandomPolicy = GreedyPolicy(
                self.system_config.system.num_noma_pairs
            )
        else:
            policy = RandomPolicy(
                self.system_config.system.num_noma_pairs,
                _stable_seed(self.master_seed, f"rollin-policy:{global_state_index:08d}"),
            )
        for _ in range(rollin_slots):
            observation, _, _, truncated, _ = env.step(policy.act(observation))
            if truncated:
                raise RuntimeError("state roll-in reached the episode boundary")
        category = self.categories[global_state_index]
        state = _apply_stress(self.system_config, env.state.clone(), category)
        return SampledState(
            state=state,
            observation=build_observation(self.system_config, state),
            global_state_index=global_state_index,
            scenario_id=f"scenario-{self.master_seed}-{global_state_index:08d}",
            episode_id=global_state_index,
            split=self.splits[global_state_index],
            category=category,
            sampling_seed=sampling_seed,
        )
