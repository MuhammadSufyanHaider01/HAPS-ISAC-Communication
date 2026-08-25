"""Teacher parsing, verification, persistence, and end-to-end smoke tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from haps_isac.config import load_config
from haps_isac.data.audit import audit_dataset
from haps_isac.data.dataset_loader import DatasetLoader
from haps_isac.data.generation import generate_demonstrations
from haps_isac.data.split_manager import DEFAULT_SPLIT_FRACTIONS
from haps_isac.envs.haps_isac_env import HapsIsacEnv
from haps_isac.teachers.base_teacher import MockTeacher, TeacherRequest, load_teacher_config
from haps_isac.teachers.benchmark import run_tournament
from haps_isac.teachers.prompt_builder import build_teacher_prompt
from haps_isac.teachers.query_cache import QueryCache, cache_key_for
from haps_isac.teachers.qwen_teacher import QwenTeacher, qwen_chat_template_kwargs
from haps_isac.teachers.response_parser import (
    TeacherResponseError,
    parse_teacher_response,
)
from haps_isac.verification.candidate_evaluator import evaluate_one_step
from haps_isac.verification.candidate_selector import select_candidate
from haps_isac.verification.rollout_verifier import (
    common_rollout_seeds,
    verify_candidate,
)


def _mock_response(state_id: str, count: int, pairs: int) -> str:
    candidates = [
        {
            "pair": 1 + index % pairs,
            "ris_code": 0,
            "eta_haps": 0.7 + 0.02 * index,
            "eta_communication": 0.7,
            "eta_near": 0.1 + 0.02 * index,
            "eta_jamming": 0.0,
            "aav_heading_rad": 0.0,
            "aav_speed_fraction": 0.0,
            "eta_cpu": 0.6,
            "reason_codes": ["unit_test"],
            "confidence": 0.5,
        }
        for index in range(count)
    ]
    return json.dumps({"schema_version": 1, "state_id": state_id, "candidates": candidates})


def test_default_dataset_split_matches_plan() -> None:
    fractions = DEFAULT_SPLIT_FRACTIONS
    assert (fractions.train, fractions.validation, fractions.test) == (0.7, 0.15, 0.15)


def test_prompt_is_deterministic_and_causal() -> None:
    config = load_config("configs/system_v1.yaml")
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=9)
    first = build_teacher_prompt(config, observation, "state-1", "1.0", 4)
    second = build_teacher_prompt(config, observation, "state-1", "1.0", 4)
    assert first == second
    assert "target_true_state" not in first.prompt
    assert "haps_ue_true" not in first.prompt
    assert set(first.causal_payload) == {
        "pair_tokens",
        "pair_mask",
        "global_features",
        "virtual_queues",
        "previous_action",
    }


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
        update={"monte_carlo_rollouts": 2, "rollout_horizon_slots": 3}
    )
    env = HapsIsacEnv(config)
    observation, _ = env.reset(seed=17)
    state = env.state.clone()
    prompt = build_teacher_prompt(config, observation, "state-a", "1.0", 4)
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
    selection = select_candidate(one_steps, summaries, 0.1)
    assert selection.selected_candidate_index in (0, 1)
    assert np.isclose(
        sum(item.quality_weight for item in selection.rankings),
        1.0,
    )


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
                    "shortlist_size": 2,
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
    assert manifest.table_counts == {
        "states": 2,
        "teacher_requests": 2,
        "candidates": 8,
        "rollouts": 8,
        "selections": 2,
        "demonstrations": 2,
    }
    for table_name in manifest.table_counts:
        first_line = (output / f"{table_name}.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(first_line)
        assert record["schema_version"] == 2
        assert record["logged_at"].endswith("+00:00")
    assert "torch" in manifest.software["packages"]
    loader = DatasetLoader(output)
    demonstrations = loader.demonstrations()
    assert len(demonstrations) == 2
    batch = next(loader.batches(2, split=demonstrations[0]["split"]))
    assert batch.pair_tokens.shape == (2, 4, 14)
    assert batch.continuous_action.shape == (2, 7)
    audit = audit_dataset(str(output))
    assert audit.passed
    assert audit.metrics["candidate_post_repair_hard_feasible_rate"] == 1.0


def test_tournament_uses_a_common_frozen_state_bank() -> None:
    system = load_config("configs/system_v1.yaml")
    base = load_teacher_config("configs/teacher.yaml")
    verification = base.verification.model_copy(
        update={
            "monte_carlo_rollouts": 2,
            "rollout_horizon_slots": 2,
            "shortlist_size": 2,
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
    assert qwen_chat_template_kwargs(
        {"chat_template_kwargs": {"enable_thinking": False}}
    ) == {"enable_thinking": False}
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
