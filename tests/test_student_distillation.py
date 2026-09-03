from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from haps_isac.models.student_actor import GemmaStructuredStudent
from haps_isac.training.distill_trainer import (
    DistillationConfig,
    DistillationTrainer,
    build_distillation_targets,
    serialize_causal_observation,
)


def _observation() -> dict[str, Any]:
    return {
        "pair_tokens": [[0.1] * 14 for _ in range(4)],
        "pair_mask": [1, 1, 1, 1],
        "global_features": [0.0] * 25,
        "virtual_queues": [0.0] * 26,
        "previous_action": [0.0] * 9,
    }


def _action(pair: int, cpu: float) -> dict[str, Any]:
    return {
        "pair": pair,
        "ris_code": 0,
        "eta_haps": 0.5,
        "eta_communication": 0.5,
        "eta_near": 0.2,
        "eta_jamming": 0.0,
        "aav_heading_rad": 0.0,
        "aav_speed_fraction": 0.0,
        "eta_cpu": cpu,
    }


def _record(index: int) -> dict[str, Any]:
    first = _action(1, 0.5)
    second = _action(4, 1.0)
    return {
        "state_id": f"state-{index}",
        "scenario_id": "scenario-0",
        "split": "train",
        "observation": _observation(),
        "selected_action": first,
        "selected_candidate_index": 0,
        "target_candidates": [
            {"candidate_index": 0, "action": first, "weight": 0.75},
            {"candidate_index": 1, "action": second, "weight": 0.25},
        ],
        "quality_weight": 0.8,
        "verifier_score": 0.4,
    }


class _FakeTokenizer:
    def __call__(self, texts: list[str], **_: Any) -> dict[str, torch.Tensor]:
        rows = [list(range(1, min(8, len(text) + 1))) for text in texts]
        width = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for row, values in enumerate(rows):
            input_ids[row, : len(values)] = torch.tensor(values)
            attention[row, : len(values)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(32, 12)

    def forward(self, input_ids: torch.Tensor, **_: Any) -> SimpleNamespace:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_state_serialization_is_causal_and_deterministic() -> None:
    text = serialize_causal_observation(_observation())
    assert text == serialize_causal_observation(_observation())
    assert "HAPS_ISAC_CAUSAL_STATE_V1" in text
    assert "<ACTION>" in text
    assert "reasoning" not in text.lower()


def test_soft_targets_are_padded_and_normalized() -> None:
    arrays = build_distillation_targets([_record(0)], 5, 1)
    assert arrays["pair_target"].shape == (1, 5)
    assert arrays["continuous_targets"].shape == (1, 2, 7)
    assert torch.allclose(
        torch.from_numpy(arrays["pair_target"]).sum(dim=-1),
        torch.ones(1),
    )
    assert arrays["selected_continuous"][0, 2] == 0.4


def test_structured_student_forward_and_cpu_training(tmp_path: Path) -> None:
    model = GemmaStructuredStudent(
        _FakeBackbone(),
        num_scheduling_actions=5,
        num_ris_actions=1,
        bottleneck_size=16,
    )
    config = DistillationConfig(
        batch_size=2,
        max_epochs=1,
        early_stopping_patience=2,
        device="cpu",
        precision="fp32",
        bottleneck_size=16,
    )
    trainer = DistillationTrainer(model, _FakeTokenizer(), config, tmp_path)
    summary = trainer.fit([_record(0), _record(1), _record(2)], [_record(3)])
    assert summary["epochs_completed"] == 1
    assert summary["best_epoch"] == 1
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "checkpoints" / "best" / "trainable_state.pt").exists()
