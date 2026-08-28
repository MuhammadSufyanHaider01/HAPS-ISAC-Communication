"""Fair common-state benchmarking for reasoning-teacher adapters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from haps_isac.baselines.greedy_policy import GreedyPolicy
from haps_isac.config import ExperimentConfig
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.envs.state import SimulatorState
from haps_isac.teachers.base_teacher import (
    BaseTeacher,
    MockTeacher,
    TeacherConfig,
    TeacherRequest,
)
from haps_isac.teachers.gemma_teacher import GemmaTeacher
from haps_isac.teachers.prompt_builder import build_teacher_prompt
from haps_isac.teachers.query_cache import QueryCache
from haps_isac.teachers.qwen_teacher import QwenTeacher
from haps_isac.teachers.response_parser import (
    TeacherResponseError,
    parse_teacher_response,
)
from haps_isac.verification.candidate_evaluator import (
    evaluate_one_step,
    preliminary_score,
)
from haps_isac.verification.candidate_selector import select_candidate
from haps_isac.verification.rollout_verifier import (
    common_rollout_seeds,
    verify_candidate,
)


@dataclass(frozen=True, slots=True)
class BenchmarkSnapshot:
    state_id: str
    observation: dict[str, np.ndarray]
    state: SimulatorState


@dataclass(frozen=True, slots=True)
class TeacherStateBenchmark:
    teacher_label: str
    state_id: str
    status: str
    error: str | None
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_hit: bool
    candidates_returned: int
    unique_candidates: int
    pre_repair_feasible_rate: float
    post_repair_hard_feasible_rate: float
    mean_repair_distance: float
    fallback_rate: float
    selected_candidate_index: int | None
    selected_risk_score: float | None
    selection_uncertain: bool | None


@dataclass(frozen=True, slots=True)
class TeacherBenchmarkSummary:
    teacher_label: str
    states: int
    schema_valid_rate: float
    mean_unique_candidate_ratio: float
    pre_repair_feasible_rate: float
    post_repair_hard_feasible_rate: float
    mean_repair_distance: float
    fallback_rate: float
    mean_selected_risk_score: float
    selection_uncertain_rate: float
    mean_latency_s: float
    mean_completion_tokens: float


@dataclass(frozen=True, slots=True)
class TeacherTournamentResult:
    winner: str
    summaries: tuple[TeacherBenchmarkSummary, ...]
    records: tuple[TeacherStateBenchmark, ...]


def build_state_bank(
    config: ExperimentConfig,
    count: int,
    master_seed: int,
) -> tuple[BenchmarkSnapshot, ...]:
    """Create teacher-independent causal states using the fixed greedy policy."""

    if count <= 0:
        raise ValueError("state-bank count must be positive")
    env = HapsIsacEnv(config)
    policy = GreedyPolicy(config.system.num_noma_pairs)
    episode = 0
    observation, _ = env.reset(seed=master_seed)
    snapshots: list[BenchmarkSnapshot] = []
    while len(snapshots) < count:
        snapshots.append(
            BenchmarkSnapshot(
                state_id=(f"benchmark:episode-{episode:06d}:slot-{env.state.slot:05d}"),
                observation={key: value.copy() for key, value in observation.items()},
                state=env.state.clone(),
            )
        )
        observation, _, _, truncated, _ = env.step(policy.act(observation))
        if truncated:
            episode += 1
            observation, _ = env.reset(seed=master_seed + episode)
    return tuple(snapshots)


def _teacher(config: TeacherConfig, num_pairs: int) -> BaseTeacher:
    cache = QueryCache(config.cache_directory) if config.cache_enabled else None
    if config.provider == "qwen":
        return QwenTeacher(config, cache)
    if config.provider == "gemma":
        return GemmaTeacher(config, cache)
    return MockTeacher(config, num_pairs, cache)


def benchmark_teacher(
    label: str,
    system_config: ExperimentConfig,
    teacher_config: TeacherConfig,
    snapshots: tuple[BenchmarkSnapshot, ...],
    master_seed: int,
) -> tuple[TeacherStateBenchmark, ...]:
    teacher = _teacher(teacher_config, system_config.system.num_noma_pairs)
    env = HapsIsacEnv(system_config)
    records: list[TeacherStateBenchmark] = []
    for snapshot in snapshots:
        prompt = build_teacher_prompt(
            system_config,
            snapshot.observation,
            snapshot.state,
            snapshot.state_id,
            teacher_config.prompt_version,
            teacher_config.num_candidates,
            teacher_config.verification,
        )
        sampling_seed = common_rollout_seeds(
            f"{snapshot.state_id}:teacher",
            master_seed,
            1,
        )[0]
        request = TeacherRequest(
            request_id=f"{label}:{snapshot.state_id}",
            state_id=snapshot.state_id,
            prompt=prompt.prompt,
            prompt_hash=prompt.prompt_hash,
            seed=sampling_seed,
        )
        call = teacher.generate(request)
        if call.status != "ok":
            records.append(
                TeacherStateBenchmark(
                    teacher_label=label,
                    state_id=snapshot.state_id,
                    status=call.status,
                    error=call.error,
                    latency_s=call.latency_s,
                    prompt_tokens=call.prompt_tokens,
                    completion_tokens=call.completion_tokens,
                    cache_hit=call.cached,
                    candidates_returned=0,
                    unique_candidates=0,
                    pre_repair_feasible_rate=0.0,
                    post_repair_hard_feasible_rate=0.0,
                    mean_repair_distance=0.0,
                    fallback_rate=0.0,
                    selected_candidate_index=None,
                    selected_risk_score=None,
                    selection_uncertain=None,
                )
            )
            continue
        try:
            parsed = parse_teacher_response(
                call.raw_text,
                snapshot.state_id,
                teacher_config.num_candidates,
                system_config.system.num_noma_pairs,
            )
        except TeacherResponseError as error:
            records.append(
                TeacherStateBenchmark(
                    teacher_label=label,
                    state_id=snapshot.state_id,
                    status="parse_error",
                    error=str(error),
                    latency_s=call.latency_s,
                    prompt_tokens=call.prompt_tokens,
                    completion_tokens=call.completion_tokens,
                    cache_hit=call.cached,
                    candidates_returned=0,
                    unique_candidates=0,
                    pre_repair_feasible_rate=0.0,
                    post_repair_hard_feasible_rate=0.0,
                    mean_repair_distance=0.0,
                    fallback_rate=0.0,
                    selected_candidate_index=None,
                    selected_risk_score=None,
                    selection_uncertain=None,
                )
            )
            continue

        one_step_seed = common_rollout_seeds(snapshot.state_id, master_seed, 1)[0]
        one_steps = {
            candidate.candidate_index: evaluate_one_step(
                env,
                snapshot.state,
                candidate,
                one_step_seed,
            )
            for candidate in parsed.candidates
        }
        shortlist = tuple(
            sorted(
                parsed.candidates,
                key=lambda candidate: preliminary_score(one_steps[candidate.candidate_index]),
            )[: teacher_config.verification.shortlist_size]
        )
        summaries = tuple(
            verify_candidate(
                env,
                snapshot.state,
                snapshot.state_id,
                candidate,
                master_seed,
                teacher_config.verification,
                retain_trajectories=False,
            )
            for candidate in shortlist
        )
        selection = select_candidate(
            one_steps,
            summaries,
            teacher_config.verification,
        )
        selected_summary = next(
            item for item in summaries if item.candidate_index == selection.selected_candidate_index
        )
        evaluations = tuple(one_steps.values())
        records.append(
            TeacherStateBenchmark(
                teacher_label=label,
                state_id=snapshot.state_id,
                status="ok",
                error=None,
                latency_s=call.latency_s,
                prompt_tokens=call.prompt_tokens,
                completion_tokens=call.completion_tokens,
                cache_hit=call.cached,
                candidates_returned=len(parsed.candidates),
                unique_candidates=parsed.unique_candidate_count,
                pre_repair_feasible_rate=float(
                    np.mean([item.pre_repair_feasible for item in evaluations])
                ),
                post_repair_hard_feasible_rate=float(
                    np.mean([item.hard_feasible for item in evaluations])
                ),
                mean_repair_distance=float(np.mean([item.repair_distance for item in evaluations])),
                fallback_rate=float(np.mean([item.fallback_used for item in evaluations])),
                selected_candidate_index=selection.selected_candidate_index,
                selected_risk_score=selected_summary.risk_score,
                selection_uncertain=selection.selection_uncertain,
            )
        )
    return tuple(records)


def summarize_teacher(
    label: str,
    records: tuple[TeacherStateBenchmark, ...],
    expected_candidates: int,
) -> TeacherBenchmarkSummary:
    valid = [record for record in records if record.status == "ok"]
    selected_scores = [
        float(record.selected_risk_score)
        for record in valid
        if record.selected_risk_score is not None
    ]
    completion_tokens = [
        float(record.completion_tokens)
        for record in records
        if record.completion_tokens is not None
    ]

    def valid_mean(attribute: str) -> float:
        return (
            float(np.mean([float(getattr(record, attribute)) for record in valid]))
            if valid
            else 0.0
        )

    return TeacherBenchmarkSummary(
        teacher_label=label,
        states=len(records),
        schema_valid_rate=len(valid) / len(records) if records else 0.0,
        mean_unique_candidate_ratio=(
            float(np.mean([record.unique_candidates / expected_candidates for record in valid]))
            if valid
            else 0.0
        ),
        pre_repair_feasible_rate=valid_mean("pre_repair_feasible_rate"),
        post_repair_hard_feasible_rate=valid_mean("post_repair_hard_feasible_rate"),
        mean_repair_distance=valid_mean("mean_repair_distance"),
        fallback_rate=valid_mean("fallback_rate"),
        mean_selected_risk_score=(
            float(np.mean(selected_scores)) if selected_scores else float("inf")
        ),
        selection_uncertain_rate=(
            float(
                np.mean(
                    [
                        bool(record.selection_uncertain)
                        for record in valid
                        if record.selection_uncertain is not None
                    ]
                )
            )
            if valid
            else 0.0
        ),
        mean_latency_s=(
            float(np.mean([record.latency_s for record in records])) if records else 0.0
        ),
        mean_completion_tokens=(float(np.mean(completion_tokens)) if completion_tokens else 0.0),
    )


def run_tournament(
    system_config: ExperimentConfig,
    teachers: dict[str, TeacherConfig],
    state_count: int,
    master_seed: int,
) -> TeacherTournamentResult:
    if len(teachers) < 2:
        raise ValueError("a tournament requires at least two teachers")
    snapshots = build_state_bank(system_config, state_count, master_seed)
    all_records: list[TeacherStateBenchmark] = []
    summaries: list[TeacherBenchmarkSummary] = []
    for label, teacher_config in teachers.items():
        records = benchmark_teacher(
            label,
            system_config,
            teacher_config,
            snapshots,
            master_seed,
        )
        all_records.extend(records)
        summaries.append(summarize_teacher(label, records, teacher_config.num_candidates))
    ordered = sorted(
        summaries,
        key=lambda item: (
            -item.schema_valid_rate,
            -item.post_repair_hard_feasible_rate,
            item.mean_selected_risk_score,
            item.mean_repair_distance,
            item.mean_latency_s,
            item.teacher_label,
        ),
    )
    return TeacherTournamentResult(
        winner=ordered[0].teacher_label,
        summaries=tuple(summaries),
        records=tuple(all_records),
    )
