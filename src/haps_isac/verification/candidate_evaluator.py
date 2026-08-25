"""One-step candidate completion, repair, and diagnostic evaluation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from haps_isac.control.action_schema import HighLevelAction
from haps_isac.control.action_transform import clip_high_level_action
from haps_isac.control.physics_completion import complete_action
from haps_isac.control.safety_repair import repair_action
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.envs.state import SimulatorState
from haps_isac.teachers.response_parser import ParsedCandidate
from haps_isac.units import thermal_noise_watts


@dataclass(frozen=True, slots=True)
class OneStepEvaluation:
    candidate_index: int
    proposed_action: HighLevelAction
    executed_action: HighLevelAction
    reason_codes: tuple[str, ...]
    teacher_confidence: float
    pre_repair_feasible: bool
    hard_feasible: bool
    fallback_used: bool
    repair_distance: float
    repair_reasons: tuple[str, ...]
    communication_power_w: float
    sensing_power_w: float
    near_power_w: float
    far_power_w: float
    cpu_frequency_hz: float
    physical_action_hash: str
    reward: float
    stage_cost: float
    constraint_violation: float
    queue_before: tuple[float, ...]
    queue_after: tuple[float, ...]
    metrics: dict[str, Any]


def action_as_dict(action: HighLevelAction) -> dict[str, int | float]:
    return {
        "pair": action.pair,
        "ris_code": action.ris_code,
        "eta_haps": action.eta_haps,
        "eta_communication": action.eta_communication,
        "eta_near": action.eta_near,
        "eta_jamming": action.eta_jamming,
        "aav_heading_rad": action.aav_heading_rad,
        "aav_speed_fraction": action.aav_speed_fraction,
        "eta_cpu": action.eta_cpu,
    }


def _physical_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        canonical = np.ascontiguousarray(value)
        digest.update(str(canonical.dtype).encode("ascii"))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


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


def evaluate_one_step(
    env: HapsIsacEnv,
    state: SimulatorState,
    candidate: ParsedCandidate,
    rollout_seed: int,
) -> OneStepEvaluation:
    """Evaluate without mutating env and expose completion/repair diagnostics."""

    config = env.config
    proposed = clip_high_level_action(
        candidate.action,
        config.system.num_noma_pairs,
        config.features.aerial_ris,
    )
    completed = complete_action(config, state, proposed)
    receiver_noise = thermal_noise_watts(
        config.system.bandwidth_hz,
        config.channels.thermal_noise_density_dbm_hz,
        config.channels.receiver_noise_figure_db,
    )
    executable = repair_action(config, state, completed, receiver_noise)
    candidate_env = env.fork_from_state(state, rollout_seed)
    _, reward, _, _, info = candidate_env.step(proposed)
    after = candidate_env.state
    physical = executable.physical
    pre_repair_feasible = not executable.log.reasons and executable.log.distance <= 1e-12
    return OneStepEvaluation(
        candidate_index=candidate.candidate_index,
        proposed_action=proposed,
        executed_action=physical.high_level,
        reason_codes=candidate.reason_codes,
        teacher_confidence=candidate.confidence,
        pre_repair_feasible=pre_repair_feasible,
        hard_feasible=bool(info["hard_feasible"]),
        fallback_used=bool(info["fallback_used"]),
        repair_distance=float(info["repair_distance"]),
        repair_reasons=tuple(str(value) for value in info["repair_reasons"]),
        communication_power_w=physical.communication_power_w,
        sensing_power_w=physical.sensing_power_w,
        near_power_w=physical.near_power_w,
        far_power_w=physical.far_power_w,
        cpu_frequency_hz=physical.cpu_frequency_hz,
        physical_action_hash=_physical_hash(
            physical.communication_beam,
            physical.sensing_beam,
            physical.sensing_combiner,
        ),
        reward=float(reward),
        stage_cost=float(info["stage_cost"]),
        constraint_violation=_constraint_violation(info),
        queue_before=tuple(float(value) for value in state.virtual_queues),
        queue_after=tuple(float(value) for value in after.virtual_queues),
        metrics=dict(info),
    )


def preliminary_score(evaluation: OneStepEvaluation) -> float:
    """Cheap deterministic score used only to select rollout candidates."""

    return (
        evaluation.stage_cost
        + 2.0 * evaluation.constraint_violation
        + 0.25 * evaluation.repair_distance
        + 2.0 * float(evaluation.fallback_used)
        + 1_000.0 * float(not evaluation.hard_feasible)
    )
