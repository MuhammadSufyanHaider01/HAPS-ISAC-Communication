"""End-to-end teacher querying, simulator verification, and dataset logging."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import uuid
from dataclasses import fields
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.baselines.random_policy import RandomPolicy
from haps_isac.config import ExperimentConfig
from haps_isac.control.action_transform import action_from_mapping
from haps_isac.data.dataset_writer import DatasetWriter
from haps_isac.data.demonstration_schema import (
    SCHEMA_VERSION,
    CandidateLogRecord,
    DemonstrationRecord,
    RolloutLogRecord,
    RunManifest,
    SelectionLogRecord,
    StateLogRecord,
    TeacherRequestLog,
    configuration_hash,
    utc_now,
)
from haps_isac.data.split_manager import assign_split
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.envs.state import SimulatorState
from haps_isac.teachers.base_teacher import (
    BaseTeacher,
    MockTeacher,
    TeacherConfig,
    TeacherRequest,
    VerificationConfig,
)
from haps_isac.teachers.gemma_teacher import GemmaTeacher
from haps_isac.teachers.prompt_builder import build_teacher_prompt
from haps_isac.teachers.query_cache import QueryCache
from haps_isac.teachers.qwen_teacher import QwenTeacher
from haps_isac.teachers.response_parser import (
    ParsedCandidate,
    ParsedTeacherResponse,
    TeacherResponseError,
    parse_teacher_response,
)
from haps_isac.verification.candidate_evaluator import (
    action_as_dict,
    evaluate_one_step,
    preliminary_score,
)
from haps_isac.verification.candidate_selector import (
    CandidateRanking,
    SelectionResult,
    select_candidate,
)
from haps_isac.verification.rollout_verifier import (
    CandidateRolloutSummary,
    RolloutRecord,
    common_rollout_seeds,
    verify_candidate,
)


def _stable_seed(master_seed: int, label: str) -> int:
    return common_rollout_seeds(label, master_seed, 1)[0]


def _git_metadata() -> tuple[str, bool]:
    environment_commit = os.environ.get("HAPS_GIT_COMMIT", "").strip() or None
    raw_environment_dirty = os.environ.get("HAPS_GIT_DIRTY", "").strip().lower()
    environment_dirty: bool | None = None
    if raw_environment_dirty in {"1", "true", "yes"}:
        environment_dirty = True
    elif raw_environment_dirty in {"0", "false", "no"}:
        environment_dirty = False
    if environment_commit is not None and environment_dirty is not None:
        return environment_commit, environment_dirty

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return environment_commit or "unknown", (
            environment_dirty if environment_dirty is not None else True
        )


def _nvidia_metadata() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as error:
        return {"available": False, "error": f"{type(error).__name__}: {error}"}
    devices: list[dict[str, str]] = []
    fields = ("index", "name", "uuid", "driver_version", "memory_total_mib")
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            devices.append(dict(zip(fields, values, strict=True)))
    return {"available": bool(devices), "devices": devices}


def _hardware_metadata() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "teacher_backend": os.environ.get("HAPS_TEACHER_BACKEND"),
        "python_environment": os.environ.get("HAPS_PYTHON_ENVIRONMENT"),
        "loaded_modules": os.environ.get("LOADEDMODULES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia": _nvidia_metadata(),
    }


def _software_metadata() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in (
        "torch",
        "torchvision",
        "Pillow",
        "vllm",
        "transformers",
        "pandas",
        "pyarrow",
        "scipy",
        "gymnasium",
        "pydantic",
        "PyYAML",
    ):
        try:
            packages[package] = package_version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "packages": packages,
    }


def build_manifest(
    run_id: str,
    system_config_path: str,
    teacher_config_path: str,
    system_config: ExperimentConfig,
    teacher_config: TeacherConfig,
    master_seed: int,
) -> RunManifest:
    commit, dirty = _git_metadata()
    return RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        created_at=utc_now(),
        completed_at=None,
        git_commit=commit,
        git_dirty=dirty,
        configuration_hash=configuration_hash(system_config, teacher_config),
        system_config_path=system_config_path,
        teacher_config_path=teacher_config_path,
        teacher_provider=teacher_config.provider,
        teacher_model_id=teacher_config.model_id,
        teacher_model_revision=teacher_config.model_revision,
        prompt_version=teacher_config.prompt_version,
        master_seed=master_seed,
        num_candidates=teacher_config.num_candidates,
        rollout_horizon_slots=teacher_config.verification.rollout_horizon_slots,
        monte_carlo_rollouts=teacher_config.verification.monte_carlo_rollouts,
        hardware=_hardware_metadata(),
        software=_software_metadata(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        table_counts={},
    )


def build_teacher(
    config: TeacherConfig,
    num_pairs: int,
) -> BaseTeacher:
    cache = QueryCache(config.cache_directory) if config.cache_enabled else None
    if config.provider == "qwen":
        return QwenTeacher(config, cache)
    if config.provider == "gemma":
        return GemmaTeacher(config, cache)
    return MockTeacher(config, num_pairs, cache)


def _state_metrics(state: SimulatorState) -> dict[str, Any]:
    covariance_trace = float(np.trace(state.available_covariance))
    estimated_channel_power = np.sum(
        np.abs(state.channels.haps_ue_estimate) ** 2,
        axis=-1,
    )
    return {
        "mean_aoi": float(np.mean(state.aoi)),
        "max_aoi": float(np.max(state.aoi)),
        "mean_aoli": float(np.mean(state.aoli)),
        "min_aoli": float(np.min(state.aoli)),
        "aosi": state.aosi,
        "queue_l1": float(np.sum(state.virtual_queues)),
        "queue_l2": float(np.linalg.norm(state.virtual_queues)),
        "queue_max": float(np.max(state.virtual_queues)),
        "tracking_covariance_trace": covariance_trace,
        "estimated_channel_power": estimated_channel_power,
        "pending_sensing_jobs": len(state.pending_sensing_jobs),
        "last_repair_distance": state.last_repair_distance,
        "last_fallback_used": state.last_fallback_used,
    }


def _verifier_only(state: SimulatorState) -> dict[str, Any]:
    true_channel_power = np.sum(np.abs(state.channels.haps_ue_true) ** 2, axis=-1)
    estimation_error = state.available_mean - state.target_true_state
    covariance_inverse = np.linalg.pinv(state.available_covariance)
    nees = float(estimation_error @ covariance_inverse @ estimation_error)
    return {
        "target_true_state": state.target_true_state,
        "target_available_mean": state.available_mean,
        "target_available_covariance": state.available_covariance,
        "target_position_squared_error": float(np.dot(estimation_error[:2], estimation_error[:2])),
        "target_state_nees": nees,
        "true_channel_power": true_channel_power,
        "channel_power_error": (
            np.sum(np.abs(state.channels.haps_ue_estimate) ** 2, axis=-1) - true_channel_power
        ),
    }


def _without(value: Any, excluded: set[str]) -> dict[str, Any]:
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name not in excluded
    }


def _baseline_scores(
    env: HapsIsacEnv,
    state: SimulatorState,
    observation: dict[str, np.ndarray],
    state_id: str,
    master_seed: int,
    settings: VerificationConfig,
) -> tuple[
    dict[str, float],
    tuple[tuple[str, CandidateRolloutSummary], ...],
]:
    seed = _stable_seed(master_seed, f"{state_id}:baseline")
    greedy_mapping = GreedyPolicy(env.config.system.num_noma_pairs).act(observation)
    random_mapping = RandomPolicy(
        env.config.system.num_noma_pairs,
        _stable_seed(master_seed, f"{state_id}:random-policy"),
    ).act(observation)
    baseline_candidates = (
        ParsedCandidate(
            candidate_index=-1,
            action=action_from_mapping(greedy_mapping),
            reason_codes=("greedy_baseline",),
            confidence=1.0,
            canonical_key=(-1,),
        ),
        ParsedCandidate(
            candidate_index=-2,
            action=action_from_mapping(random_mapping),
            reason_codes=("random_baseline",),
            confidence=1.0,
            canonical_key=(-2,),
        ),
    )
    one_steps = tuple(
        evaluate_one_step(env, state, candidate, seed) for candidate in baseline_candidates
    )
    summaries = tuple(
        verify_candidate(
            env,
            state,
            state_id,
            candidate,
            master_seed,
            settings,
            retain_trajectories=False,
        )
        for candidate in baseline_candidates
    )
    scores: dict[str, float] = {}
    labeled_summaries: list[tuple[str, CandidateRolloutSummary]] = []
    for label, one_step, summary in zip(
        ("greedy", "random"),
        one_steps,
        summaries,
        strict=True,
    ):
        scores[f"{label}_one_step_cost"] = one_step.stage_cost
        scores[f"{label}_verified_risk_score"] = summary.risk_score
        scores[f"{label}_verified_mean_cost"] = summary.mean_cost
        scores[f"{label}_verified_cvar_cost"] = summary.cvar_cost
        scores[f"{label}_verified_constraint_violation"] = summary.mean_constraint_violation
        scores[f"{label}_verified_repair_distance"] = summary.mean_repair_distance
        scores[f"{label}_verified_fallback_rate"] = summary.fallback_rate
        labeled_summaries.append((f"{label}_baseline", summary))
    return scores, tuple(labeled_summaries)


def _rollout_log(
    run_id: str,
    state_id: str,
    rollout: RolloutRecord,
    retain: bool,
    policy_label: str = "teacher_candidate",
) -> RolloutLogRecord:
    metrics = _without(rollout, {"trajectory"})
    return RolloutLogRecord(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        state_id=state_id,
        policy_label=policy_label,
        candidate_index=rollout.candidate_index,
        rollout_index=rollout.rollout_index,
        rollout_seed=rollout.rollout_seed,
        retained_trajectory=retain,
        metrics=metrics,
        trajectory=rollout.trajectory if retain else (),
    )


def _summary_metrics(summary: CandidateRolloutSummary) -> dict[str, Any]:
    return _without(summary, {"rollouts"})


def _ranking_map(selection: SelectionResult) -> dict[int, CandidateRanking]:
    return {item.candidate_index: item for item in selection.rankings}


def _trajectory_audit_selected(
    run_id: str,
    state_id: str,
    candidate_index: int,
    rollout_index: int,
    fraction: float,
) -> bool:
    if fraction <= 0.0:
        return False
    seed = _stable_seed(
        0,
        f"{run_id}:{state_id}:{candidate_index}:{rollout_index}:audit",
    )
    return seed / float(2**32) < fraction


def _log_invalid_request(
    writer: DatasetWriter,
    run_id: str,
    state_id: str,
    request: TeacherRequest,
    call: Any,
    config: TeacherConfig,
    error: str,
) -> None:
    writer.append(
        "teacher_requests",
        TeacherRequestLog(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            request_id=request.request_id,
            state_id=state_id,
            model_id=config.model_id,
            model_revision=config.model_revision,
            prompt_version=config.prompt_version,
            prompt_hash=request.prompt_hash,
            cache_key=call.cache_key,
            sampling_seed=request.seed,
            status="parse_error" if call.status == "ok" else call.status,
            cached=call.cached,
            retries=call.retries,
            latency_s=call.latency_s,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            parse_valid=False,
            schema_valid=False,
            candidates_returned=0,
            unique_candidates=0,
            truncated=bool(
                call.completion_tokens is not None
                and call.completion_tokens >= config.sampling.max_tokens
            ),
            error=error,
            prompt=request.prompt,
            raw_response=call.raw_text,
            reasoning_response=call.reasoning_text,
        ),
    )


def _log_valid_request(
    writer: DatasetWriter,
    run_id: str,
    request: TeacherRequest,
    call: Any,
    parsed: ParsedTeacherResponse,
    config: TeacherConfig,
) -> None:
    writer.append(
        "teacher_requests",
        TeacherRequestLog(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            request_id=request.request_id,
            state_id=request.state_id,
            model_id=config.model_id,
            model_revision=config.model_revision,
            prompt_version=config.prompt_version,
            prompt_hash=request.prompt_hash,
            cache_key=call.cache_key,
            sampling_seed=request.seed,
            status="ok",
            cached=call.cached,
            retries=call.retries,
            latency_s=call.latency_s,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            parse_valid=True,
            schema_valid=True,
            candidates_returned=len(parsed.candidates),
            unique_candidates=parsed.unique_candidate_count,
            truncated=False,
            error=None,
            prompt=request.prompt,
            raw_response=call.raw_text,
            reasoning_response=call.reasoning_text,
        ),
    )


def _verify_and_log(
    writer: DatasetWriter,
    run_id: str,
    state_id: str,
    request_id: str,
    scenario_id: str,
    split: str,
    observation_payload: dict[str, Any],
    env: HapsIsacEnv,
    state: SimulatorState,
    parsed: ParsedTeacherResponse,
    teacher_config: TeacherConfig,
    master_seed: int,
) -> ParsedCandidate:
    common_seed = common_rollout_seeds(state_id, master_seed, 1)[0]
    one_steps = {
        candidate.candidate_index: evaluate_one_step(
            env,
            state,
            candidate,
            common_seed,
        )
        for candidate in parsed.candidates
    }
    shortlisted = tuple(
        sorted(
            parsed.candidates,
            key=lambda item: preliminary_score(one_steps[item.candidate_index]),
        )[: teacher_config.verification.shortlist_size]
    )
    summaries = tuple(
        verify_candidate(
            env,
            state,
            state_id,
            candidate,
            master_seed,
            teacher_config.verification,
            retain_trajectories=True,
        )
        for candidate in shortlisted
    )
    selection = select_candidate(
        one_steps,
        summaries,
        teacher_config.verification,
    )
    summary_map = {item.candidate_index: item for item in summaries}
    rankings = _ranking_map(selection)
    selected_index = selection.selected_candidate_index

    for summary in summaries:
        for rollout in summary.rollouts:
            retain = (
                (
                    teacher_config.logging.retain_selected_trajectories
                    and summary.candidate_index == selected_index
                )
                or (
                    teacher_config.logging.retain_failure_trajectories and not rollout.hard_feasible
                )
                or _trajectory_audit_selected(
                    run_id,
                    state_id,
                    summary.candidate_index,
                    rollout.rollout_index,
                    teacher_config.logging.trajectory_audit_fraction,
                )
            )
            writer.append("rollouts", _rollout_log(run_id, state_id, rollout, retain))

    for candidate in parsed.candidates:
        evaluation = one_steps[candidate.candidate_index]
        candidate_summary = summary_map.get(candidate.candidate_index)
        ranking = rankings.get(candidate.candidate_index)
        is_selected = candidate.candidate_index == selected_index
        writer.append(
            "candidates",
            CandidateLogRecord(
                schema_version=SCHEMA_VERSION,
                run_id=run_id,
                state_id=state_id,
                request_id=request_id,
                candidate_index=candidate.candidate_index,
                reason_codes=candidate.reason_codes,
                teacher_confidence=candidate.confidence,
                proposed_action=action_as_dict(evaluation.proposed_action),
                executed_action=action_as_dict(evaluation.executed_action),
                pre_repair_feasible=evaluation.pre_repair_feasible,
                hard_feasible=evaluation.hard_feasible,
                fallback_used=evaluation.fallback_used,
                repair_distance=evaluation.repair_distance,
                repair_reasons=evaluation.repair_reasons,
                physical_action_hash=evaluation.physical_action_hash,
                communication_power_w=evaluation.communication_power_w,
                sensing_power_w=evaluation.sensing_power_w,
                near_power_w=evaluation.near_power_w,
                far_power_w=evaluation.far_power_w,
                cpu_frequency_hz=evaluation.cpu_frequency_hz,
                one_step_reward=evaluation.reward,
                one_step_stage_cost=evaluation.stage_cost,
                one_step_constraint_violation=evaluation.constraint_violation,
                queue_before=evaluation.queue_before,
                queue_after=evaluation.queue_after,
                one_step_metrics=evaluation.metrics,
                rollout_verified=candidate_summary is not None,
                rollout_summary=(
                    _summary_metrics(candidate_summary) if candidate_summary is not None else None
                ),
                rank=ranking.rank if ranking is not None else None,
                quality_weight=(ranking.quality_weight if ranking is not None else 0.0),
                selected=is_selected,
                rejection_reason=(
                    None
                    if is_selected
                    else (
                        "not_shortlisted"
                        if candidate_summary is None
                        else "higher_verified_risk_score"
                    )
                ),
            ),
        )

    baseline_scores, baseline_summaries = _baseline_scores(
        env,
        state,
        {key: np.asarray(value) for key, value in observation_payload.items()},
        state_id,
        master_seed,
        teacher_config.verification,
    )
    for policy_label, baseline_summary in baseline_summaries:
        for rollout in baseline_summary.rollouts:
            writer.append(
                "rollouts",
                _rollout_log(
                    run_id,
                    state_id,
                    rollout,
                    retain=False,
                    policy_label=policy_label,
                ),
            )
    writer.append(
        "selections",
        SelectionLogRecord(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            state_id=state_id,
            selected_candidate_index=selected_index,
            score_margin=selection.score_margin,
            standardized_margin=selection.standardized_margin,
            selection_uncertain=selection.selection_uncertain,
            margin_confidence_lower=selection.margin_confidence_lower,
            margin_confidence_upper=selection.margin_confidence_upper,
            selection_probability=selection.selection_probability,
            baseline_scores=baseline_scores,
            oracle_regret=None,
            acceptance_status="accepted",
            acceptance_reason="lowest_risk_score_after_common_random_rollouts",
        ),
    )
    selected_candidate = parsed.candidates[selected_index]
    selected_evaluation = one_steps[selected_index]
    selected_summary = summary_map[selected_index]
    selected_ranking = rankings[selected_index]
    writer.append(
        "demonstrations",
        DemonstrationRecord(
            schema_version=SCHEMA_VERSION,
            run_id=run_id,
            state_id=state_id,
            scenario_id=scenario_id,
            split=split,
            observation=observation_payload,
            selected_candidate_index=selected_index,
            selected_action=action_as_dict(selected_evaluation.executed_action),
            executed_action=action_as_dict(selected_evaluation.executed_action),
            quality_weight=selected_ranking.quality_weight,
            verifier_score=selected_summary.risk_score,
            repair_distance=selected_evaluation.repair_distance,
            fallback_used=selected_evaluation.fallback_used,
            selection_uncertain=selection.selection_uncertain,
        ),
    )
    return ParsedCandidate(
        candidate_index=selected_candidate.candidate_index,
        action=selected_evaluation.executed_action,
        reason_codes=selected_candidate.reason_codes,
        confidence=selected_candidate.confidence,
        canonical_key=selected_candidate.canonical_key,
    )


def generate_demonstrations(
    system_config: ExperimentConfig,
    teacher_config: TeacherConfig,
    system_config_path: str,
    teacher_config_path: str,
    output_directory: str | Path,
    requested_states: int,
    master_seed: int,
    run_id: str | None = None,
    export_parquet: bool | None = None,
) -> RunManifest:
    """Generate requested states, logging failures without poisoning demonstrations."""

    if requested_states <= 0:
        raise ValueError("requested_states must be positive")
    effective_run_id = run_id or f"teacher-{uuid.uuid4().hex[:12]}"
    manifest = build_manifest(
        effective_run_id,
        system_config_path,
        teacher_config_path,
        system_config,
        teacher_config,
        master_seed,
    )
    writer = DatasetWriter(
        output_directory,
        manifest,
        flush_every=teacher_config.logging.flush_every,
    )
    teacher = build_teacher(teacher_config, system_config.system.num_noma_pairs)
    env = HapsIsacEnv(system_config)
    greedy = GreedyPolicy(system_config.system.num_noma_pairs)
    episode_id = 0
    observation, _ = env.reset(seed=master_seed)
    generated = 0
    try:
        while generated < requested_states:
            if env.state.slot >= system_config.system.episode_slots:
                episode_id += 1
                observation, _ = env.reset(seed=master_seed + episode_id)
            state = env.state.clone()
            scenario_id = f"scenario-{master_seed + episode_id}"
            state_id = f"{effective_run_id}:episode-{episode_id:06d}:slot-{state.slot:05d}"
            split = assign_split(scenario_id, master_seed)
            prompt = build_teacher_prompt(
                system_config,
                observation,
                state,
                state_id,
                teacher_config.prompt_version,
                teacher_config.num_candidates,
            )
            writer.append(
                "states",
                StateLogRecord(
                    schema_version=SCHEMA_VERSION,
                    run_id=effective_run_id,
                    state_id=state_id,
                    scenario_id=scenario_id,
                    episode_id=episode_id,
                    slot=state.slot,
                    split=split,
                    causal_state_hash=prompt.causal_state_hash,
                    observation=prompt.causal_payload,
                    teacher_guidance={"sic_safe_templates": prompt.sic_safe_templates},
                    state_metrics=_state_metrics(state),
                    verifier_only=_verifier_only(state),
                ),
            )
            request = TeacherRequest(
                request_id=f"{state_id}:request-000",
                state_id=state_id,
                prompt=prompt.prompt,
                prompt_hash=prompt.prompt_hash,
                seed=_stable_seed(master_seed, f"{state_id}:teacher"),
            )
            call = teacher.generate(request)
            parsed: ParsedTeacherResponse | None = None
            parse_error: str | None = call.error
            if call.status == "ok":
                try:
                    parsed = parse_teacher_response(
                        call.raw_text,
                        state_id,
                        teacher_config.num_candidates,
                        system_config.system.num_noma_pairs,
                    )
                except TeacherResponseError as error:
                    parse_error = str(error)
            if parsed is None:
                _log_invalid_request(
                    writer,
                    effective_run_id,
                    state_id,
                    request,
                    call,
                    teacher_config,
                    parse_error or "unknown teacher failure",
                )
                writer.append(
                    "selections",
                    SelectionLogRecord(
                        schema_version=SCHEMA_VERSION,
                        run_id=effective_run_id,
                        state_id=state_id,
                        selected_candidate_index=-1,
                        score_margin=0.0,
                        standardized_margin=0.0,
                        selection_uncertain=True,
                        margin_confidence_lower=0.0,
                        margin_confidence_upper=0.0,
                        selection_probability=0.0,
                        baseline_scores={},
                        oracle_regret=None,
                        acceptance_status="rejected",
                        acceptance_reason=parse_error or "teacher request failed",
                    ),
                )
                observation, _, _, truncated, _ = env.step(greedy.act(observation))
            else:
                _log_valid_request(
                    writer,
                    effective_run_id,
                    request,
                    call,
                    parsed,
                    teacher_config,
                )
                selected = _verify_and_log(
                    writer,
                    effective_run_id,
                    state_id,
                    request.request_id,
                    scenario_id,
                    split,
                    prompt.causal_payload,
                    env,
                    state,
                    parsed,
                    teacher_config,
                    master_seed,
                )
                observation, _, _, truncated, _ = env.step(selected.action)
            generated += 1
            if truncated:
                episode_id += 1
                observation, _ = env.reset(seed=master_seed + episode_id)
        return writer.finalize(
            teacher_config.logging.export_parquet if export_parquet is None else export_parquet
        )
    except BaseException:
        writer.close()
        raise
