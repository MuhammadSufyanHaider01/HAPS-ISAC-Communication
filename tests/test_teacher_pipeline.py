"""Teacher parsing, verification, persistence, and end-to-end smoke tests."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from haps_isac.config import load_config
from haps_isac.control.virtual_queues import queue_slices
from haps_isac.data import generation as generation_module
from haps_isac.data.audit import audit_dataset
from haps_isac.data.dataset_loader import DatasetLoader
from haps_isac.data.generation import generate_demonstrations
from haps_isac.data.merge import merge_shards
from haps_isac.data.quality_report import build_teacher_quality_report
from haps_isac.data.split_manager import DEFAULT_SPLIT_FRACTIONS
from haps_isac.data.state_sampler import StratifiedStateSampler, exact_assignments, shard_bounds
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.teachers.base_teacher import (
    MockTeacher,
    TeacherConfig,
    TeacherRequest,
    load_teacher_config,
)
from haps_isac.teachers.benchmark import run_tournament
from haps_isac.teachers.candidate_pool import canonicalize_teacher_response
from haps_isac.teachers.prompt_builder import build_teacher_prompt
from haps_isac.teachers.query_cache import QueryCache, cache_key_for
from haps_isac.teachers.qwen_teacher import (
    QwenTeacher,
    qwen_chat_template_kwargs,
    qwen_device_map,
)
from haps_isac.teachers.response_parser import (
    TeacherResponseError,
    parse_teacher_response,
)
from haps_isac.verification.candidate_evaluator import evaluate_one_step
from haps_isac.verification.candidate_selector import select_candidate
from haps_isac.verification.rollout_verifier import (
    common_rollout_seeds,
    summarize_rollouts,
    verify_candidate,
)


def _mock_response(state_id: str, count: int, pairs: int) -> str:
    candidates = [
        {
            "template_id": "p1_sense2w",
            "eta_near": 0.1 + 0.02 * index,
            "eta_cpu": 0.6,
            "reason_codes": ["unit_test"],
            "confidence": 0.5,
        }
        for index in range(count)
    ]
    return json.dumps({"schema_version": 1, "state_id": state_id, "candidates": candidates})


def _fast_mock_teacher() -> TeacherConfig:
    base = load_teacher_config("configs/teacher.yaml")
    return base.model_copy(
        update={
            "provider": "mock",
            "model_id": "mock/test",
            "model_revision": "deterministic-v1",
            "num_candidates": 4,
            "cache_enabled": False,
            "verification": base.verification.model_copy(
                update={
                    "monte_carlo_rollouts": 1,
                    "max_monte_carlo_rollouts": 1,
                    "rollout_batch_size": 1,
                    "rollout_horizon_slots": 1,
                    "shortlist_size": 2,
                    "uncertainty_bootstrap_samples": 50,
                }
            ),
            "dataset": base.dataset.model_copy(update={"maximum_rollin_slots": 0}),
            "logging": base.logging.model_copy(update={"export_parquet": False, "flush_every": 1}),
        }
    )


def test_default_dataset_split_matches_plan() -> None:
    fractions = DEFAULT_SPLIT_FRACTIONS
    assert (fractions.train, fractions.validation, fractions.test) == (0.7, 0.15, 0.15)


def test_git_metadata_prefers_submission_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HAPS_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("HAPS_GIT_DIRTY", "0")
    assert generation_module._git_metadata() == ("a" * 40, False)


def test_prompt_is_deterministic_and_causal() -> None:
    config = load_config("configs/system_v1.yaml")
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=9)
    first = build_teacher_prompt(config, observation, env.state, "state-1", "1.0", 4)
    second = build_teacher_prompt(config, observation, env.state, "state-1", "1.0", 4)
    assert first == second
    assert "target_true_state" not in first.prompt
    assert "haps_ue_true" not in first.prompt
    assert "template_id" in first.prompt
    assert "valid_template_ids" in first.prompt
    assert "free refinements eta_near and eta_cpu" in first.prompt
    assert "optimization_contract" in first.prompt
    assert "near_aoi_fraction" in first.prompt
    assert "virtual_queue_transform" in first.prompt
    assert "target_true_state" not in first.semantic_state_packet
    assert set(first.causal_payload) == {
        "pair_tokens",
        "pair_mask",
        "global_features",
        "virtual_queues",
        "previous_action",
    }
    assert first.sic_safe_templates
    sensing_only = first.sensing_only_template
    assert sensing_only["pair"] == 0
    assert sensing_only["sensing_power_w"] >= config.constraints.minimum_sensing_power_w
    assert {int(template["pair"]) for template in first.sic_safe_templates} == {
        1,
        2,
        3,
        4,
    }
    for index, template in enumerate(first.sic_safe_templates):
        raw = json.dumps(
            {
                "schema_version": 1,
                "state_id": f"template-{index}",
                "candidates": [
                    {
                        "template_id": template["template_id"],
                        "eta_near": template["recommended_eta_near"],
                        "eta_cpu": 0.5,
                        "reason_codes": ["unit_test"],
                        "confidence": 1.0,
                    }
                ],
            }
        )
        parsed = parse_teacher_response(raw, f"template-{index}", 1, 4)
        parsed = canonicalize_teacher_response(parsed, first)
        evaluation = evaluate_one_step(
            env,
            env.state,
            parsed.candidates[0],
            rollout_seed=19,
        )
        assert evaluation.pre_repair_feasible
        assert not evaluation.fallback_used

    sensing_raw = json.dumps(
        {
            "schema_version": 1,
            "state_id": "sensing-template",
            "candidates": [
                {
                    "template_id": sensing_only["template_id"],
                    "eta_near": sensing_only["eta_near"],
                    "eta_cpu": 0.5,
                    "reason_codes": ["unit_test"],
                    "confidence": 1.0,
                }
            ],
        }
    )
    sensing_parsed = parse_teacher_response(sensing_raw, "sensing-template", 1, 4)
    sensing_parsed = canonicalize_teacher_response(sensing_parsed, first)
    sensing_evaluation = evaluate_one_step(
        env,
        env.state,
        sensing_parsed.candidates[0],
        rollout_seed=23,
    )
    assert sensing_evaluation.pre_repair_feasible
    assert not sensing_evaluation.fallback_used


def test_template_id_projection_reconstructs_fixed_controls() -> None:
    config = load_config("configs/system_v1.yaml")
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=13)
    prompt = build_teacher_prompt(config, observation, env.state, "state-template", "2.1", 1)
    raw = json.dumps(
        {
            "schema_version": 1,
            "state_id": "state-template",
            "candidates": [
                {
                    "template_id": "p1_sense2w_max_near",
                    "eta_near": 0.5,
                    "eta_cpu": 0.7,
                    "reason_codes": ["secrecy_priority"],
                    "confidence": 0.8,
                }
            ],
        }
    )
    parsed = canonicalize_teacher_response(
        parse_teacher_response(raw, "state-template", 1, 4),
        prompt,
    )
    candidate = parsed.candidates[0]
    assert candidate.template_id == "p1_sense2w"
    assert candidate.action.pair == 1
    assert candidate.action.eta_haps == prompt.sic_safe_templates[0]["eta_haps"]
    assert candidate.action.eta_communication == prompt.sic_safe_templates[0]["eta_communication"]
    assert candidate.action.eta_near <= prompt.sic_safe_templates[0]["maximum_eta_near"]
    assert candidate.action.eta_cpu == 0.7


def test_response_parser_enforces_identity_count_and_bounds() -> None:
    parsed = parse_teacher_response(_mock_response("s", 4, 4), "s", 4, 4)
    assert len(parsed.candidates) == 4
    assert parsed.unique_candidate_count == 4
    with pytest.raises(TeacherResponseError):
        parse_teacher_response(_mock_response("wrong", 4, 4), "s", 4, 4)
    with pytest.raises(TeacherResponseError):
        parse_teacher_response(_mock_response("s", 3, 4), "s", 4, 4)

    invalid = json.loads(_mock_response("s", 4, 4))
    invalid["candidates"][0]["eta_near"] = 0.9
    with pytest.raises(TeacherResponseError):
        parse_teacher_response(json.dumps(invalid), "s", 4, 4)


def test_response_parser_normalizes_only_observed_provider_aliases() -> None:
    payload = json.loads(_mock_response("s", 1, 4))
    candidate = payload["candidates"][0]
    candidate.pop("template_id")
    candidate["template_id_actual"] = "p2_sense10w"
    candidate["reason_codes"] = "provider emitted a scalar explanation"
    candidate["reason_codes_override"] = "provider correction metadata"
    parsed = parse_teacher_response(json.dumps(payload), "s", 1, 4)
    assert parsed.candidates[0].template_id == "p2_sense10w"
    assert parsed.candidates[0].template_id_raw is None
    assert parsed.candidates[0].reason_codes == ("provider emitted a scalar explanation",)
    assert parsed.normalization_notes == (
        "candidate[0].template_id_actual->template_id",
        "candidate[0].reason_codes_string_to_array",
        "candidate[0].reason_codes_override_removed",
    )

    too_many_reasons = json.loads(_mock_response("s", 1, 4))
    too_many_reasons["candidates"][0]["reason_codes"] = ["a", "b", "c", "d"]
    with pytest.raises(TeacherResponseError):
        parse_teacher_response(json.dumps(too_many_reasons), "s", 1, 4)


def test_template_id_actual_correction_wins_conflicting_provider_fields() -> None:
    config = load_config("configs/system_v1.yaml")
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=13)
    prompt = build_teacher_prompt(config, observation, env.state, "state-correction", "2.2", 1)
    raw = json.dumps(
        {
            "schema_version": 1,
            "state_id": "state-correction",
            "candidates": [
                {
                    "template_id": "p2_sense10w",
                    "template_id_actual": "p1_sense18w",
                    "eta_near": 0.2,
                    "eta_cpu": 0.5,
                    "reason_codes": "corrected by provider",
                    "confidence": 0.75,
                }
            ],
        }
    )
    parsed = canonicalize_teacher_response(
        parse_teacher_response(raw, "state-correction", 1, 4),
        prompt,
    )
    candidate = parsed.candidates[0]
    assert candidate.template_id == "p1_sense18w"
    assert candidate.template_id_raw == "p2_sense10w"
    assert candidate.template_resolution == "normalized_template_id"
    assert "candidate[0].template_id_conflict_prefer_actual" in parsed.normalization_notes


def test_template_id_nearest_repair_is_same_pair_and_logged() -> None:
    config = load_config("configs/system_v1.yaml")
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=13)
    prompt = build_teacher_prompt(config, observation, env.state, "state-nearest", "2.2", 1)
    prompt = replace(
        prompt,
        sic_safe_templates=tuple(
            template
            for template in prompt.sic_safe_templates
            if template["template_id"] != "p1_sense18w"
        ),
    )
    raw = json.dumps(
        {
            "schema_version": 1,
            "state_id": "state-nearest",
            "candidates": [
                {
                    "template_id": "p1_sense18w",
                    "eta_near": 0.2,
                    "eta_cpu": 0.5,
                    "reason_codes": ["unit_test"],
                    "confidence": 0.75,
                }
            ],
        }
    )
    parsed = canonicalize_teacher_response(
        parse_teacher_response(raw, "state-nearest", 1, 4),
        prompt,
    )
    candidate = parsed.candidates[0]
    assert candidate.template_id == "p1_sense10w"
    assert candidate.template_id_raw == "p1_sense18w"
    assert candidate.template_resolution == "nearest_available_template"
    assert candidate.action.pair == 1

    unresolved = json.loads(raw)
    unresolved["candidates"][0]["template_id"] = "p9_sense18w"
    with pytest.raises(TeacherResponseError, match="no resolvable template_id"):
        canonicalize_teacher_response(
            parse_teacher_response(json.dumps(unresolved), "state-nearest", 1, 4),
            prompt,
        )
    distant = json.loads(raw)
    distant["candidates"][0]["template_id"] = "p1_sense100w"
    with pytest.raises(TeacherResponseError, match="no resolvable template_id"):
        canonicalize_teacher_response(
            parse_teacher_response(json.dumps(distant), "state-nearest", 1, 4),
            prompt,
        )


def test_teacher_request_id_is_forwarded_for_server_telemetry() -> None:
    config = load_teacher_config("configs/teacher.yaml")
    request = TeacherRequest(
        request_id="state-1:request-000",
        state_id="state-1",
        prompt="test prompt",
        prompt_hash="a" * 64,
        seed=17,
    )
    body = QwenTeacher(config)._request_body(request)
    assert body["user"] == request.request_id
    assert body["seed"] == request.seed
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_qwen_device_map_only_shards_across_multiple_gpus() -> None:
    assert qwen_device_map(1) is None
    assert qwen_device_map(2) == "balanced"
    with pytest.raises(ValueError, match="cuda_device_count must be positive"):
        qwen_device_map(0)


def test_query_cache_is_content_addressed(tmp_path: pytest.TempPathFactory) -> None:
    directory = tmp_path / "cache"  # type: ignore[operator]
    cache = QueryCache(directory)
    key = cache_key_for("model", "revision", "a" * 64, {"temperature": 1.0}, 7)
    assert cache.get(key) is None
    cache.put(key, {"raw_text": "{}", "prompt_tokens": 3})
    assert cache.get(key) == {"raw_text": "{}", "prompt_tokens": 3}


def test_common_random_rollouts_are_reproducible() -> None:
    config = load_config("configs/system_v1.yaml")
    teacher = load_teacher_config("configs/teacher.yaml")
    settings = teacher.verification.model_copy(
        update={
            "monte_carlo_rollouts": 2,
            "rollout_horizon_slots": 3,
            "uncertainty_bootstrap_samples": 100,
        }
    )
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=17)
    state = env.state.clone()
    prompt = build_teacher_prompt(config, observation, state, "state-a", "1.0", 4)
    mock_config = teacher.model_copy(
        update={
            "provider": "mock",
            "model_id": "mock/test",
            "num_candidates": 4,
            "verification": settings,
        }
    )
    raw = MockTeacher(mock_config, config.system.num_noma_pairs).generate(
        TeacherRequest("request-a", "state-a", prompt.prompt, prompt.prompt_hash, 3)
    )
    parsed = parse_teacher_response(raw.raw_text, "state-a", 4, 4)
    parsed = canonicalize_teacher_response(parsed, prompt)
    seeds = common_rollout_seeds("state-a", 123, 2)
    assert seeds == common_rollout_seeds("state-a", 123, 2)
    assert seeds != common_rollout_seeds("state-b", 123, 2)

    one_steps = {
        item.candidate_index: evaluate_one_step(env, state, item, seeds[0])
        for item in parsed.candidates[:2]
    }
    summaries = tuple(
        verify_candidate(env, state, "state-a", item, 123, settings, False)
        for item in parsed.candidates[:2]
    )
    repeated = verify_candidate(
        env,
        state,
        "state-a",
        parsed.candidates[0],
        123,
        settings,
        False,
    )
    assert summaries[0].mean_cost == repeated.mean_cost
    assert env.state.slot == 0
    selection = select_candidate(one_steps, summaries, settings)
    assert selection.selected_candidate_index in (0, 1)
    assert selection.decision_status in {
        "decisive",
        "practically_equivalent",
        "unresolved",
    }
    assert np.isclose(
        sum(item.quality_weight for item in selection.rankings),
        1.0,
    )
    assert 0.0 <= selection.selection_probability <= 1.0
    assert selection.margin_confidence_lower <= selection.margin_confidence_upper
    paired_best_rollouts = tuple(
        replace(
            rollout,
            candidate_index=0,
            discounted_cost=float(index),
            mean_constraint_violation=0.0,
            mean_repair_distance=0.0,
            fallback_rate=0.0,
            hard_feasible=True,
        )
        for index, rollout in enumerate(summaries[0].rollouts, start=1)
    )
    paired_second_rollouts = tuple(
        replace(
            rollout,
            candidate_index=1,
            discounted_cost=float(index) + 0.01,
            mean_constraint_violation=0.0,
            mean_repair_distance=0.0,
            fallback_rate=0.0,
            hard_feasible=True,
        )
        for index, rollout in enumerate(summaries[1].rollouts, start=1)
    )
    paired_selection = select_candidate(
        one_steps,
        (
            summarize_rollouts(0, paired_best_rollouts, settings),
            summarize_rollouts(1, paired_second_rollouts, settings),
        ),
        settings,
    )
    assert paired_selection.margin_confidence_lower > 0.0
    assert not paired_selection.selection_uncertain
    assert paired_selection.decision_status == "practically_equivalent"
    assert paired_selection.equivalent_candidate_indices == (0, 1)


def test_mock_generation_writes_loadable_linked_tables(tmp_path: pytest.TempPathFactory) -> None:
    output = tmp_path / "dataset"  # type: ignore[operator]
    system = load_config("configs/system_v1.yaml")
    teacher = load_teacher_config("configs/teacher.yaml")
    teacher = teacher.model_copy(
        update={
            "provider": "mock",
            "model_id": "mock/test",
            "num_candidates": 4,
            "cache_enabled": False,
            "verification": teacher.verification.model_copy(
                update={
                    "monte_carlo_rollouts": 2,
                    "rollout_horizon_slots": 2,
                    "max_monte_carlo_rollouts": 2,
                    "shortlist_size": 2,
                    "uncertainty_bootstrap_samples": 100,
                }
            ),
            "logging": teacher.logging.model_copy(update={"export_parquet": False}),
        }
    )
    manifest = generate_demonstrations(
        system,
        teacher,
        "configs/system_v1.yaml",
        "configs/teacher.yaml",
        output,
        requested_states=2,
        master_seed=91,
        run_id="test-run",
        export_parquet=False,
    )
    assert manifest.table_counts["states"] == 2
    assert manifest.table_counts["teacher_requests"] == 2
    assert manifest.table_counts["selections"] == 2
    assert manifest.table_counts["demonstrations"] == 2
    candidates = [
        json.loads(line) for line in (output / "candidates.jsonl").read_text().splitlines()
    ]
    selections = [
        json.loads(line) for line in (output / "selections.jsonl").read_text().splitlines()
    ]
    assert sum(record["candidate_source"] == "teacher" for record in candidates) == 8
    assert sum(record["candidate_source"] == "greedy_baseline" for record in candidates) == 2
    assert all(record["safe_template_coverage_rate"] == 1.0 for record in selections)
    assert all(record["greedy_candidate_index"] is not None for record in selections)
    assert manifest.table_counts["candidates"] == sum(
        record["candidate_pool_count"] for record in selections
    )
    assert manifest.table_counts["rollouts"] == sum(
        (record["verified_candidate_count"] + record["external_baseline_rollout_count"])
        * record["verification_rollouts"]
        for record in selections
    )
    selected_by_state = {record["state_id"]: record for record in candidates if record["selected"]}
    for selection in selections:
        selected = selected_by_state[selection["state_id"]]
        assert selected["rollout_summary"]["risk_score"] <= (
            selection["baseline_scores"]["greedy_verified_risk_score"] + 1e-12
        )
        assert selection["oracle_diagnostic"]["reference"] == (
            "reduced_grid_one_step_stage_cost_not_global_optimum"
        )
    for table_name in manifest.table_counts:
        first_line = (output / f"{table_name}.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
        assert record["schema_version"] == 5
        assert record["logged_at"].endswith("+00:00")
    assert "torch" in manifest.software["packages"]
    loader = DatasetLoader(output)
    demonstrations = loader.demonstrations()
    assert len(demonstrations) == 2
    selected_split = demonstrations[0]["split"]
    split_count = sum(record["split"] == selected_split for record in demonstrations)
    batch = next(loader.batches(2, split=selected_split))
    assert batch.pair_tokens.shape == (split_count, 4, 14)
    assert batch.continuous_action.shape == (split_count, 7)
    assert batch.target_pair.shape[0] == split_count
    assert np.allclose(np.sum(batch.target_weight, axis=1), 1.0)
    assert np.all(batch.target_mask[:, 0] == 1)
    audit = audit_dataset(str(output))
    assert audit.passed
    assert audit.metrics["candidate_post_repair_hard_feasible_rate"] == 1.0
    assert audit.metrics["request_normalization_rate"] == 0.0
    assert audit.metrics["teacher_template_resolution_repair_rate"] == 0.0
    quality = build_teacher_quality_report(output)
    assert quality["counts"]["states"] == 2
    assert quality["baseline_comparison"]["compared_states"] == 2
    assert "eta_communication" in quality["action_field_changes"]
    assert quality["baseline_comparison"]["verified_compared_states"] == 2
    assert quality["scale_up_gates"]["request_schema_valid_rate"]["passed"]
    assert quality["scale_up_gates"]["request_normalization_rate"]["passed"]
    assert quality["scale_up_gates"]["teacher_template_resolution_repair_rate"]["passed"]
    assert quality["teacher_format"]["template_resolution_repair_rate"] == 0.0
    assert quality["selection_quality"]["safe_template_coverage_rate"] == 1.0
    assert quality["baseline_comparison"]["selected_no_worse_than_greedy_verified_rate"] == 1.0

    assert quality["distillation_targets"]["valid_rate"] == 1.0


def test_canonicalization_failure_rejects_request_without_placeholder_actions(
    tmp_path: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "canonicalization-failure"  # type: ignore[operator]
    system = load_config("configs/system_v1.yaml")
    teacher = _fast_mock_teacher()

    def fail_canonicalization(*_args: object, **_kwargs: object) -> object:
        raise TeacherResponseError("candidate has no resolvable template_id")

    monkeypatch.setattr(
        generation_module,
        "canonicalize_teacher_response",
        fail_canonicalization,
    )
    manifest = generate_demonstrations(
        system,
        teacher,
        "configs/system_v1.yaml",
        "configs/teacher.yaml",
        output,
        requested_states=1,
        master_seed=7,
        run_id="canonicalization-failure",
        export_parquet=False,
    )
    assert manifest.table_counts["teacher_requests"] == 1
    assert manifest.table_counts["candidates"] == 0
    assert manifest.table_counts["demonstrations"] == 0
    request = json.loads((output / "teacher_requests.jsonl").read_text().splitlines()[0])
    selection = json.loads((output / "selections.jsonl").read_text().splitlines()[0])
    assert request["schema_valid"] is False
    assert "no resolvable template_id" in request["error"]
    assert selection["acceptance_status"] == "rejected"


def test_tournament_uses_a_common_frozen_state_bank() -> None:
    system = load_config("configs/system_v1.yaml")
    base = load_teacher_config("configs/teacher.yaml")
    verification = base.verification.model_copy(
        update={
            "monte_carlo_rollouts": 2,
            "rollout_horizon_slots": 2,
            "shortlist_size": 2,
            "uncertainty_bootstrap_samples": 100,
        }
    )
    common = {
        "provider": "mock",
        "num_candidates": 3,
        "cache_enabled": False,
        "verification": verification,
    }
    teachers = {
        "alpha": base.model_copy(update={**common, "model_id": "mock/alpha"}),
        "beta": base.model_copy(update={**common, "model_id": "mock/beta"}),
    }
    result = run_tournament(system, teachers, state_count=2, master_seed=44)
    assert result.winner == "alpha"
    alpha_ids = [record.state_id for record in result.records if record.teacher_label == "alpha"]
    beta_ids = [record.state_id for record in result.records if record.teacher_label == "beta"]
    assert alpha_ids == beta_ids
    assert all(summary.schema_valid_rate == 1.0 for summary in result.summaries)


def test_transformers_server_validates_thinking_override() -> None:
    assert qwen_chat_template_kwargs({}) == {"enable_thinking": True}
    assert qwen_chat_template_kwargs({"chat_template_kwargs": {"enable_thinking": False}}) == {
        "enable_thinking": False
    }
    with pytest.raises(ValueError, match="enable_thinking must be boolean"):
        qwen_chat_template_kwargs({"chat_template_kwargs": {"enable_thinking": "false"}})


def test_audit_reports_missing_tables_without_crashing(
    tmp_path: pytest.TempPathFactory,
) -> None:
    output = tmp_path / "incomplete-dataset"  # type: ignore[operator]
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"num_candidates": 8}),
        encoding="utf-8",
    )
    audit = audit_dataset(str(output))
    assert not audit.passed
    assert "missing canonical table: candidates.jsonl" in audit.errors
    assert audit.counts["candidates"] == 0


def test_stratified_plan_has_exact_5000_state_quotas_and_deterministic_stress() -> None:
    system = load_config("configs/system_v1.yaml")
    teacher = _fast_mock_teacher()
    state_fractions = teacher.dataset.state_fractions.model_dump()
    split_fractions = teacher.dataset.split_fractions.model_dump()

    assert Counter(exact_assignments(5000, state_fractions, 123, "state-category")) == {
        "ordinary": 1750,
        "freshness_stress": 1250,
        "sensing_stress": 750,
        "secrecy_stress": 750,
        "boundary_rare": 500,
    }
    assert Counter(exact_assignments(5000, split_fractions, 123, "scenario-split")) == {
        "train": 3500,
        "validation": 750,
        "test": 750,
    }
    bounds = [shard_bounds(5000, index, 10) for index in range(10)]
    assert bounds[0] == (0, 500)
    assert bounds[-1] == (4500, 5000)
    assert all(stop - start == 500 for start, stop in bounds)

    sampler = StratifiedStateSampler(system, teacher.dataset, 20, 123)
    env = HapsIsacEnv(system)
    first_by_category = {}
    for index in range(20):
        sampled = sampler.sample(env, index)
        assert np.all(np.isfinite(sampled.observation["global_features"]))
        first_by_category.setdefault(sampled.category, sampled)
    assert set(first_by_category) == {
        "ordinary",
        "freshness_stress",
        "sensing_stress",
        "secrecy_stress",
        "boundary_rare",
    }
    freshness = first_by_category["freshness_stress"].state
    sensing = first_by_category["sensing_stress"].state
    secrecy = first_by_category["secrecy_stress"].state
    boundary = first_by_category["boundary_rare"].state
    layout = queue_slices(system.system.num_noma_pairs)
    assert float(np.min(freshness.aoi)) >= 0.70 * system.freshness.aoi_cap_slots
    assert np.isclose(
        np.trace(sensing.available_covariance),
        0.90 * system.constraints.maximum_covariance_trace,
    )
    assert float(np.min(secrecy.virtual_queues[layout.secrecy])) >= (
        0.75 * system.constraints.queue_reference
    )
    assert float(np.min(boundary.virtual_queues)) >= 0.95 * system.constraints.queue_reference

    repeated = StratifiedStateSampler(system, teacher.dataset, 20, 123).sample(
        HapsIsacEnv(system), first_by_category["boundary_rare"].global_state_index
    )
    np.testing.assert_allclose(repeated.state.aoi, boundary.aoi)
    np.testing.assert_allclose(repeated.state.virtual_queues, boundary.virtual_queues)


def test_generation_resume_trims_and_regenerates_incomplete_state(tmp_path: Path) -> None:
    output = tmp_path / "resume-dataset"
    system = load_config("configs/system_v1.yaml")
    teacher = _fast_mock_teacher()
    first = generate_demonstrations(
        system,
        teacher,
        "configs/system_v1.yaml",
        "configs/teacher.yaml",
        output,
        requested_states=3,
        master_seed=77,
        run_id="resume-test",
        export_parquet=False,
    )
    demonstration_path = output / "demonstrations.jsonl"
    lines = demonstration_path.read_text(encoding="utf-8").splitlines()
    demonstration_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    resumed = generate_demonstrations(
        system,
        teacher,
        "configs/system_v1.yaml",
        "configs/teacher.yaml",
        output,
        requested_states=3,
        master_seed=77,
        run_id="resume-test",
        export_parquet=False,
        resume=True,
    )
    assert resumed.created_at == first.created_at
    assert resumed.table_counts["states"] == 3
    assert resumed.table_counts["teacher_requests"] == 3
    assert resumed.table_counts["selections"] == 3
    assert resumed.table_counts["demonstrations"] == 3
    selections = [
        json.loads(line) for line in (output / "selections.jsonl").read_text().splitlines()
    ]
    assert resumed.table_counts["candidates"] == sum(
        record["candidate_pool_count"] for record in selections
    )
    assert resumed.table_counts["rollouts"] == sum(
        (record["verified_candidate_count"] + record["external_baseline_rollout_count"])
        * record["verification_rollouts"]
        for record in selections
    )
    assert audit_dataset(str(output)).passed


def test_complete_shards_merge_in_global_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HAPS_GIT_COMMIT", "b" * 40)
    monkeypatch.setenv("HAPS_GIT_DIRTY", "0")
    root = tmp_path / "sharded"
    system = load_config("configs/system_v1.yaml")
    teacher = _fast_mock_teacher()
    shard_directories = []
    for shard_index in range(2):
        start, stop = shard_bounds(4, shard_index, 2)
        directory = root / "shards" / f"shard-{shard_index:03d}"
        generate_demonstrations(
            system,
            teacher,
            "configs/system_v1.yaml",
            "configs/teacher.yaml",
            directory,
            requested_states=stop - start,
            master_seed=55,
            run_id=f"merge-test-shard-{shard_index:03d}",
            export_parquet=False,
            total_states=4,
            global_state_start=start,
            shard_index=shard_index,
            shard_count=2,
        )
        shard_directories.append(directory)

    output = root / "merged"
    manifest = merge_shards(
        tuple(reversed(shard_directories)),
        output,
        "merge-test",
        expected_shards=2,
        export_parquet=False,
    )
    assert manifest.table_counts["states"] == 4
    states = list(DatasetLoader(output).iter_table("states"))
    assert [record["global_state_index"] for record in states] == [0, 1, 2, 3]
    assert {record["run_id"] for record in states} == {"merge-test"}
    assert audit_dataset(str(output)).passed

def test_merge_accepts_legacy_endpoint_dependent_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy shards remain mergeable when only their runtime endpoint differs."""

    monkeypatch.setenv("HAPS_GIT_COMMIT", "c" * 40)
    monkeypatch.setenv("HAPS_GIT_DIRTY", "0")
    root = tmp_path / "legacy-sharded"
    system = load_config("configs/system_v1.yaml")
    teacher = _fast_mock_teacher()
    shard_directories = []
    for shard_index in range(2):
        start, stop = shard_bounds(4, shard_index, 2)
        directory = root / "shards" / f"shard-{shard_index:03d}"
        generate_demonstrations(
            system,
            teacher.model_copy(update={"base_url": f"http://127.0.0.1:{18000 + shard_index}"}),
            "configs/system_v1.yaml",
            "configs/teacher.yaml",
            directory,
            requested_states=stop - start,
            master_seed=55,
            run_id=f"legacy-merge-shard-{shard_index:03d}",
            export_parquet=False,
            total_states=4,
            global_state_start=start,
            shard_index=shard_index,
            shard_count=2,
        )
        shard_directories.append(directory)

    for directory in shard_directories:
        path = directory / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        # Simulate the pre-fix manifests generated by the production run.
        manifest["configuration_hash"] = f"legacy-{manifest['shard_index']}"
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    output = root / "merged"
    merged = merge_shards(tuple(shard_directories), output, "legacy-merge", expected_shards=2)
    assert merged.table_counts["states"] == 4
    assert audit_dataset(str(output)).passed
