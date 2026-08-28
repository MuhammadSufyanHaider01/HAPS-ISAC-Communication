"""Versioned plotting and training records for verified demonstrations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

SCHEMA_VERSION = 5


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def configuration_hash(*configs: BaseModel) -> str:
    payload = [config.model_dump(mode="json") for config in configs]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    """Convert dataclasses and numerical objects to strict finite JSON values."""

    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            if math.isinf(value):
                return "Infinity" if value > 0.0 else "-Infinity"
            raise ValueError("NaN values are forbidden in dataset records")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class RunManifest:
    schema_version: int
    run_id: str
    created_at: str
    completed_at: str | None
    git_commit: str
    git_dirty: bool
    configuration_hash: str
    system_config_path: str
    teacher_config_path: str
    teacher_provider: str
    teacher_model_id: str
    teacher_model_revision: str
    prompt_version: str
    master_seed: int
    num_candidates: int
    rollout_horizon_slots: int
    monte_carlo_rollouts: int
    hardware: dict[str, Any]
    software: dict[str, Any]
    slurm_job_id: str | None
    table_counts: dict[str, int]
    export_errors: tuple[str, ...] = ()
    max_monte_carlo_rollouts: int | None = None
    total_requested_states: int | None = None
    global_state_start: int = 0
    global_state_stop: int | None = None
    shard_index: int = 0
    shard_count: int = 1
    sampling_strategy: str = "sequential"
    state_distribution: dict[str, float] = field(default_factory=dict)
    split_fractions: dict[str, float] = field(default_factory=dict)
    candidate_pool_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StateLogRecord:
    schema_version: int
    run_id: str
    state_id: str
    scenario_id: str
    episode_id: int
    slot: int
    split: str
    causal_state_hash: str
    observation: dict[str, Any]
    teacher_guidance: dict[str, Any]
    state_metrics: dict[str, Any]
    verifier_only: dict[str, Any]
    global_state_index: int
    state_category: str
    sampling_seed: int
    logged_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class TeacherRequestLog:
    schema_version: int
    run_id: str
    request_id: str
    state_id: str
    model_id: str
    model_revision: str
    prompt_version: str
    prompt_hash: str
    cache_key: str
    sampling_seed: int
    status: str
    cached: bool
    retries: int
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    parse_valid: bool
    schema_valid: bool
    candidates_returned: int
    unique_candidates: int
    truncated: bool
    error: str | None
    prompt: str
    raw_response: str
    reasoning_response: str | None
    logged_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class CandidateLogRecord:
    schema_version: int
    run_id: str
    state_id: str
    request_id: str
    candidate_index: int
    candidate_source: str
    candidate_source_label: str
    reason_codes: tuple[str, ...]
    template_id: str | None
    template_compliant: bool | None
    teacher_confidence: float
    proposed_action: dict[str, Any]
    executed_action: dict[str, Any]
    pre_repair_feasible: bool
    hard_feasible: bool
    fallback_used: bool
    repair_distance: float
    repair_reasons: tuple[str, ...]
    physical_action_hash: str
    communication_power_w: float
    sensing_power_w: float
    near_power_w: float
    far_power_w: float
    cpu_frequency_hz: float
    one_step_reward: float
    one_step_stage_cost: float
    one_step_constraint_violation: float
    queue_before: tuple[float, ...]
    queue_after: tuple[float, ...]
    one_step_metrics: dict[str, Any]
    rollout_verified: bool
    rollout_summary: dict[str, Any] | None
    rank: int | None
    quality_weight: float
    selected: bool
    rejection_reason: str | None
    logged_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class RolloutLogRecord:
    schema_version: int
    run_id: str
    state_id: str
    policy_label: str
    candidate_index: int
    rollout_index: int
    rollout_seed: int
    retained_trajectory: bool
    metrics: dict[str, Any]
    trajectory: tuple[dict[str, Any], ...]
    logged_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class SelectionLogRecord:
    schema_version: int
    run_id: str
    state_id: str
    selected_candidate_index: int
    selected_candidate_source: str
    candidate_pool_count: int
    teacher_candidate_count: int
    safe_template_expected_count: int
    safe_template_coverage_rate: float
    candidate_source_counts: dict[str, int]
    greedy_candidate_index: int | None
    verified_candidate_count: int
    external_baseline_rollout_count: int
    score_margin: float
    standardized_margin: float
    selection_uncertain: bool
    margin_confidence_lower: float
    margin_confidence_upper: float
    selection_probability: float
    baseline_scores: dict[str, float]
    oracle_regret: float | None
    oracle_diagnostic: dict[str, Any] | None
    acceptance_status: str
    acceptance_reason: str
    decision_status: str
    practical_equivalence_margin: float
    equivalent_candidate_indices: tuple[int, ...]
    verification_rollouts: int
    adaptive_rounds: int
    logged_at: str = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class DistillationTargetRecord:
    candidate_index: int
    action: dict[str, Any]
    weight: float
    verifier_score: float


@dataclass(frozen=True, slots=True)
class DemonstrationRecord:
    schema_version: int
    run_id: str
    state_id: str
    scenario_id: str
    split: str
    observation: dict[str, Any]
    selected_candidate_index: int
    selected_action: dict[str, Any]
    executed_action: dict[str, Any]
    quality_weight: float
    verifier_score: float
    repair_distance: float
    fallback_used: bool
    selection_uncertain: bool
    decision_status: str
    target_candidates: tuple[DistillationTargetRecord, ...]
    target_entropy: float
    logged_at: str = field(default_factory=utc_now)
