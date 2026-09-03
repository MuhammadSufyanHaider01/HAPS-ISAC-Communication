"""Gemma PEFT backbone wrapper with a structured hybrid-action head.

The language model is used as a causal-state encoder only. The action head
reads the hidden state at the terminal ACTION sentinel and emits the
numerical policy distribution shared by offline distillation and PPO.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

CONTINUOUS_ACTION_DIM = 7
CONSTRAINT_HEAD_DIM = 5


@dataclass(frozen=True, slots=True)
class HybridActionOutput:
    """Outputs of the structured policy and critic heads."""

    scheduling_logits: Tensor
    ris_logits: Tensor
    continuous_mean: Tensor
    continuous_log_std: Tensor
    value: Tensor
    constraint_logits: Tensor
    pooled_hidden: Tensor


class StructuredActionHead(nn.Module):
    """Small trainable head attached to a Gemma hidden representation."""

    def __init__(
        self,
        hidden_size: int,
        num_scheduling_actions: int,
        num_ris_actions: int,
        constraint_dim: int = CONSTRAINT_HEAD_DIM,
        bottleneck_size: int = 512,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_scheduling_actions <= 0 or num_ris_actions <= 0:
            raise ValueError("categorical action dimensions must be positive")
        if constraint_dim <= 0:
            raise ValueError("constraint_dim must be positive")
        self.num_scheduling_actions = num_scheduling_actions
        self.num_ris_actions = num_ris_actions
        self.constraint_dim = constraint_dim
        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck_size),
            nn.GELU(),
            nn.LayerNorm(bottleneck_size),
        )
        self.scheduling = nn.Linear(bottleneck_size, num_scheduling_actions)
        self.ris = nn.Linear(bottleneck_size, num_ris_actions)
        self.continuous_mean = nn.Linear(bottleneck_size, CONTINUOUS_ACTION_DIM)
        self.continuous_log_std = nn.Linear(bottleneck_size, CONTINUOUS_ACTION_DIM)
        self.value = nn.Linear(bottleneck_size, 1)
        self.constraint = nn.Linear(bottleneck_size, constraint_dim)

    def forward(self, pooled_hidden: Tensor) -> HybridActionOutput:
        features = self.trunk(pooled_hidden)
        return HybridActionOutput(
            scheduling_logits=self.scheduling(features),
            ris_logits=self.ris(features),
            continuous_mean=self.continuous_mean(features),
            continuous_log_std=self.continuous_log_std(features).clamp(-5.0, 2.0),
            value=self.value(features).squeeze(-1),
            constraint_logits=self.constraint(features),
            pooled_hidden=pooled_hidden,
        )


def _hidden_size_from_config(config: Any) -> int:
    """Read hidden size across Gemma text/config variants."""

    for name in ("hidden_size", "d_model"):
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        return _hidden_size_from_config(text_config)
    raise ValueError("could not infer backbone hidden size from model config")


def _last_hidden_state(outputs: Any) -> Tensor:
    hidden = getattr(outputs, "last_hidden_state", None)
    if isinstance(hidden, Tensor):
        return hidden
    hidden_states = getattr(outputs, "hidden_states", None)
    if isinstance(hidden_states, (tuple, list)) and hidden_states:
        last_hidden = hidden_states[-1]
        if isinstance(last_hidden, Tensor):
            return last_hidden
    if isinstance(outputs, (tuple, list)) and outputs:
        candidate = outputs[0]
        if isinstance(candidate, Tensor) and candidate.ndim == 3:
            return candidate
    raise ValueError("backbone output does not expose a last hidden state")


class GemmaStructuredStudent(nn.Module):
    """Gemma encoder plus structured action/value/constraint heads."""

    def __init__(
        self,
        backbone: nn.Module,
        num_scheduling_actions: int,
        num_ris_actions: int,
        constraint_dim: int = CONSTRAINT_HEAD_DIM,
        action_token_id: int | None = None,
        bottleneck_size: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.action_token_id = action_token_id
        hidden_size = _hidden_size_from_config(backbone.config)
        self.hidden_size = hidden_size
        self.action_head = StructuredActionHead(
            hidden_size=hidden_size,
            num_scheduling_actions=num_scheduling_actions,
            num_ris_actions=num_ris_actions,
            constraint_dim=constraint_dim,
            bottleneck_size=bottleneck_size,
        )

    def _pool_action_state(
        self,
        hidden: Tensor,
        attention_mask: Tensor | None,
        action_positions: Tensor | None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden.shape
        if action_positions is None:
            if attention_mask is None:
                action_positions = torch.full(
                    (batch_size,), sequence_length - 1, dtype=torch.long, device=hidden.device
                )
            else:
                action_positions = attention_mask.to(dtype=torch.long).sum(dim=1).sub(1)
        action_positions = action_positions.to(device=hidden.device, dtype=torch.long).clamp(
            0, sequence_length - 1
        )
        rows = torch.arange(batch_size, device=hidden.device)
        return hidden[rows, action_positions]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        action_positions: Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> HybridActionOutput:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **backbone_kwargs,
        )
        hidden = _last_hidden_state(outputs)
        pooled = self._pool_action_state(hidden, attention_mask, action_positions)
        return self.action_head.forward(pooled)

    def trainable_parameter_count(self) -> int:
        """Return the number of parameters participating in optimization."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
