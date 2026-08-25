"""Reduced-grid one-step oracle using immutable candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from haps_isac.control.action_schema import HighLevelAction
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.envs.state import SimulatorState


@dataclass(frozen=True, slots=True)
class OracleResult:
    action: HighLevelAction
    stage_cost: float
    repair_distance: float
    hard_feasible: bool
    candidates_evaluated: int


def one_step_grid_oracle(
    env: HapsIsacEnv,
    *,
    state: SimulatorState | None = None,
    rollout_seed: int = 0,
    haps_fractions: tuple[float, ...] = (0.5, 1.0),
    communication_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    near_fractions: tuple[float, ...] = (0.1, 0.25, 0.4),
    cpu_fractions: tuple[float, ...] = (0.0, 1.0),
) -> OracleResult:
    snapshot = env.state if state is None else state
    candidates: list[
        tuple[tuple[bool, float, float], HighLevelAction, dict[str, Any]]
    ] = []
    for pair, eta_haps, eta_communication, eta_near, eta_cpu in product(
        range(env.config.system.num_noma_pairs + 1),
        haps_fractions,
        communication_fractions,
        near_fractions,
        cpu_fractions,
    ):
        action = HighLevelAction(
            pair=pair,
            ris_code=0,
            eta_haps=eta_haps,
            eta_communication=eta_communication,
            eta_near=eta_near,
            eta_jamming=0.0,
            aav_heading_rad=0.0,
            aav_speed_fraction=0.0,
            eta_cpu=eta_cpu,
        )
        _, _, _, _, info = env.evaluate_candidate(snapshot, action, rollout_seed)
        rank = (
            not bool(info["hard_feasible"]),
            float(info["stage_cost"]),
            float(info["repair_distance"]),
        )
        candidates.append((rank, action, info))
    best_rank, best_action, best_info = min(candidates, key=lambda item: item[0])
    return OracleResult(
        action=best_action,
        stage_cost=best_rank[1],
        repair_distance=best_rank[2],
        hard_feasible=bool(best_info["hard_feasible"]),
        candidates_evaluated=len(candidates),
    )
