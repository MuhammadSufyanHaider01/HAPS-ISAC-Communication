"""Quality-weighted offline action distillation for the Gemma student.

This module deliberately keeps Hugging Face imports out of the training core.
The CLI constructs the Gemma/PEFT backbone; the trainer only requires a
tokenizer-compatible callable and the structured student module. That makes
the data contract and loss testable on CPU with a tiny fake encoder.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import Tensor

from haps_isac.models.student_actor import (
    CONSTRAINT_HEAD_DIM,
    CONTINUOUS_ACTION_DIM,
    GemmaStructuredStudent,
    HybridActionOutput,
)

CONTINUOUS_ACTION_FIELDS = (
    "eta_haps",
    "eta_communication",
    "eta_near",
    "eta_jamming",
    "aav_heading_rad",
    "aav_speed_fraction",
    "eta_cpu",
)
SIGMOID_CONTINUOUS_INDICES = (0, 1, 2, 3, 5, 6)
HEADING_INDEX = 4


@dataclass(frozen=True, slots=True)
class DistillationLossWeights:
    """Relative weights for the policy and auxiliary losses."""

    categorical: float = 1.0
    continuous: float = 1.0
    value: float = 0.10
    constraint: float = 0.10
    auxiliary: float = 0.0

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in asdict(self).values()):
            raise ValueError("loss weights must be non-negative")
        if self.categorical + self.continuous <= 0.0:
            raise ValueError("at least one policy loss weight must be positive")


@dataclass(frozen=True, slots=True)
class DistillationConfig:
    """Training and model defaults mirrored by configs/distillation.yaml."""

    optimizer: str = "adamw"
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    max_epochs: int = 20
    max_steps: int | None = None
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    categorical_label_smoothing: float = 0.02
    max_sequence_length: int = 4096
    seed: int = 20260824
    device: str = "auto"
    precision: str = "bf16"
    num_scheduling_actions: int = 5
    num_ris_actions: int = 1
    constraint_dim: int = CONSTRAINT_HEAD_DIM
    bottleneck_size: int = 512
    model_id: str = "google/gemma-4-E4B-it"
    model_revision: str = "main"
    peft_method: str = "qlora"
    load_in_4bit: bool = True
    compute_dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    gradient_checkpointing: bool = True
    loss_weights: DistillationLossWeights = DistillationLossWeights()

    @classmethod
    def from_yaml(cls, path: str | Path) -> DistillationConfig:
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream) or {}
        if not isinstance(raw, Mapping):
            raise ValueError("distillation config must be a YAML mapping")
        distill = raw.get("distillation", raw)
        student = raw.get("student", {})
        peft = raw.get("peft", {})
        losses = distill.get("loss_weights", raw.get("loss_weights", {}))
        if not isinstance(distill, Mapping) or not isinstance(student, Mapping):
            raise ValueError("distillation and student config sections must be mappings")
        if not isinstance(peft, Mapping) or not isinstance(losses, Mapping):
            raise ValueError("peft and loss_weights config sections must be mappings")
        defaults = cls()
        target_modules = peft.get("target_modules", defaults.target_modules)
        if not isinstance(target_modules, Sequence) or isinstance(target_modules, str):
            raise ValueError("peft.target_modules must be a sequence")
        default_losses = DistillationLossWeights()
        return cls(
            optimizer=str(distill.get("optimizer", defaults.optimizer)),
            learning_rate=float(distill.get("learning_rate", defaults.learning_rate)),
            weight_decay=float(distill.get("weight_decay", defaults.weight_decay)),
            batch_size=int(distill.get("batch_size", defaults.batch_size)),
            gradient_accumulation_steps=int(
                distill.get("gradient_accumulation_steps", defaults.gradient_accumulation_steps)
            ),
            max_epochs=int(distill.get("max_epochs", defaults.max_epochs)),
            max_steps=(int(distill["max_steps"]) if distill.get("max_steps") is not None else None),
            gradient_clip_norm=float(
                distill.get("gradient_clip_norm", defaults.gradient_clip_norm)
            ),
            early_stopping_patience=int(
                distill.get("early_stopping_patience", defaults.early_stopping_patience)
            ),
            categorical_label_smoothing=float(
                distill.get("categorical_label_smoothing", defaults.categorical_label_smoothing)
            ),
            max_sequence_length=int(
                student.get("max_sequence_length", defaults.max_sequence_length)
            ),
            seed=int(distill.get("seed", defaults.seed)),
            device=str(distill.get("device", defaults.device)),
            precision=str(distill.get("precision", defaults.precision)),
            num_scheduling_actions=int(
                distill.get("num_scheduling_actions", defaults.num_scheduling_actions)
            ),
            num_ris_actions=int(distill.get("num_ris_actions", defaults.num_ris_actions)),
            constraint_dim=int(distill.get("constraint_dim", defaults.constraint_dim)),
            bottleneck_size=int(distill.get("bottleneck_size", defaults.bottleneck_size)),
            model_id=str(student.get("model_id", defaults.model_id)),
            model_revision=str(student.get("model_revision", defaults.model_revision)),
            peft_method=str(peft.get("method", defaults.peft_method)),
            load_in_4bit=bool(peft.get("load_in_4bit", defaults.load_in_4bit)),
            compute_dtype=str(peft.get("compute_dtype", defaults.compute_dtype)),
            lora_rank=int(peft.get("lora_rank", defaults.lora_rank)),
            lora_alpha=int(peft.get("lora_alpha", defaults.lora_alpha)),
            lora_dropout=float(peft.get("lora_dropout", defaults.lora_dropout)),
            target_modules=tuple(str(value) for value in target_modules),
            gradient_checkpointing=bool(
                peft.get("gradient_checkpointing", defaults.gradient_checkpointing)
            ),
            loss_weights=DistillationLossWeights(
                categorical=float(losses.get("categorical", default_losses.categorical)),
                continuous=float(losses.get("continuous", default_losses.continuous)),
                value=float(losses.get("value", default_losses.value)),
                constraint=float(losses.get("constraint", default_losses.constraint)),
                auxiliary=float(losses.get("auxiliary", default_losses.auxiliary)),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DistillationBatch:
    """Tokenized observations and padded soft targets for one mini-batch."""

    input_ids: Tensor
    attention_mask: Tensor
    action_positions: Tensor
    pair_target: Tensor
    ris_target: Tensor
    continuous_targets: Tensor
    target_weight: Tensor
    target_mask: Tensor
    quality_weight: Tensor
    selected_pair: Tensor
    selected_ris: Tensor
    selected_continuous: Tensor
    value_target: Tensor
    constraint_target: Tensor
    constraint_mask: Tensor

    def to(self, device: torch.device) -> DistillationBatch:
        values = {name: getattr(self, name).to(device) for name in self.__dataclass_fields__}
        return DistillationBatch(**values)


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    """Named loss terms retained in the JSONL training log."""

    total: Tensor
    categorical: Tensor
    continuous: Tensor
    value: Tensor
    constraint: Tensor
    auxiliary: Tensor

    def as_float_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name).detach().cpu().item())
            for name in self.__dataclass_fields__
        }


def serialize_causal_observation(observation: Mapping[str, Any]) -> str:
    """Serialize only the causal numerical packet; teacher rationale is excluded."""

    required = {"pair_tokens", "pair_mask", "global_features", "virtual_queues", "previous_action"}
    missing = required.difference(observation)
    if missing:
        raise ValueError(f"observation is missing keys: {sorted(missing)}")

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, (float, np.floating)):
            return round(float(value), 6)
        if isinstance(value, (int, bool, str)) or value is None:
            return value
        if isinstance(value, np.ndarray):
            return normalize(value.tolist())
        raise TypeError(f"unsupported observation value: {type(value).__name__}")

    payload = {key: normalize(observation[key]) for key in sorted(required)}
    return (
        "HAPS_ISAC_CAUSAL_STATE_V1\n"
        + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n<ACTION>"
    )


def _continuous_action(action: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(
        [float(action[field]) for field in CONTINUOUS_ACTION_FIELDS], dtype=np.float32
    )
    values[2] *= 2.0
    values[HEADING_INDEX] /= math.pi
    if not np.all(np.isfinite(values)):
        raise ValueError("continuous action contains non-finite values")
    return np.clip(values, -1.0, 1.0)


def _record_targets(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    targets = record.get("target_candidates")
    if targets is None:
        return [
            {
                "candidate_index": record["selected_candidate_index"],
                "action": record["selected_action"],
                "weight": 1.0,
            }
        ]
    if not isinstance(targets, list) or not targets:
        raise ValueError(f"record {record.get('state_id')} has no distillation targets")
    if any(not isinstance(target, Mapping) for target in targets):
        raise ValueError(f"record {record.get('state_id')} has malformed targets")
    return list(targets)


def build_distillation_targets(
    records: Sequence[Mapping[str, Any]],
    num_scheduling_actions: int,
    num_ris_actions: int,
    constraint_dim: int = CONSTRAINT_HEAD_DIM,
) -> dict[str, np.ndarray]:
    """Convert variable-length soft targets into padded arrays for Torch."""

    if not records:
        raise ValueError("cannot build targets for an empty record set")
    target_sets = [_record_targets(record) for record in records]
    maximum_targets = max(len(targets) for targets in target_sets)
    pair_target = np.zeros((len(records), num_scheduling_actions), dtype=np.float32)
    ris_target = np.zeros((len(records), num_ris_actions), dtype=np.float32)
    continuous_targets = np.zeros(
        (len(records), maximum_targets, CONTINUOUS_ACTION_DIM), dtype=np.float32
    )
    target_weight = np.zeros((len(records), maximum_targets), dtype=np.float32)
    target_mask = np.zeros((len(records), maximum_targets), dtype=np.float32)
    selected_pair = np.zeros(len(records), dtype=np.int64)
    selected_ris = np.zeros(len(records), dtype=np.int64)
    selected_continuous = np.zeros((len(records), CONTINUOUS_ACTION_DIM), dtype=np.float32)
    value_target = np.zeros(len(records), dtype=np.float32)
    constraint_target = np.zeros((len(records), constraint_dim), dtype=np.float32)
    constraint_mask = np.zeros((len(records), constraint_dim), dtype=np.float32)

    for row, (record, targets) in enumerate(zip(records, target_sets, strict=True)):
        weights = np.asarray([float(target.get("weight", -1.0)) for target in targets])
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError(f"record {record.get('state_id')} has invalid target weights")
        weights = weights / weights.sum()
        for column, (target, weight) in enumerate(zip(targets, weights, strict=True)):
            action = target.get("action")
            if not isinstance(action, Mapping):
                raise ValueError(f"record {record.get('state_id')} has a malformed target action")
            pair = int(action["pair"])
            ris = int(action.get("ris_code", 0))
            if not 0 <= pair < num_scheduling_actions:
                raise ValueError(f"pair label {pair} is outside the configured action head")
            if not 0 <= ris < num_ris_actions:
                raise ValueError(f"RIS label {ris} is outside the configured action head")
            pair_target[row, pair] += float(weight)
            ris_target[row, ris] += float(weight)
            continuous_targets[row, column] = _continuous_action(action)
            target_weight[row, column] = float(weight)
            target_mask[row, column] = 1.0
        selected = record.get("selected_action") or targets[0]["action"]
        selected_pair[row] = int(selected["pair"])
        selected_ris[row] = int(selected.get("ris_code", 0))
        selected_continuous[row] = _continuous_action(selected)
        value = record.get("value_target", record.get("verifier_score", 0.0))
        value_target[row] = float(value) if math.isfinite(float(value)) else 0.0
        supplied_constraints = record.get("constraint_targets")
        if isinstance(supplied_constraints, Sequence) and not isinstance(
            supplied_constraints, (str, bytes)
        ):
            if len(supplied_constraints) != constraint_dim:
                raise ValueError("constraint_targets length does not match constraint head")
            values = np.asarray([float(value) for value in supplied_constraints], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError("constraint_targets contain non-finite values")
            constraint_target[row] = np.clip(values, 0.0, 1.0)
            constraint_mask[row] = 1.0

    return {
        "pair_target": pair_target,
        "ris_target": ris_target,
        "continuous_targets": continuous_targets,
        "target_weight": target_weight,
        "target_mask": target_mask,
        "selected_pair": selected_pair,
        "selected_ris": selected_ris,
        "selected_continuous": selected_continuous,
        "quality_weight": np.asarray(
            [max(1.0e-3, min(1.0, float(record.get("quality_weight", 1.0)))) for record in records],
            dtype=np.float32,
        ),
        "value_target": value_target,
        "constraint_target": constraint_target,
        "constraint_mask": constraint_mask,
    }


class DistillationCollator:
    """Tokenize causal observations and pad their verified target sets."""

    def __init__(
        self,
        tokenizer: Any,
        max_sequence_length: int,
        num_scheduling_actions: int,
        num_ris_actions: int,
        constraint_dim: int = CONSTRAINT_HEAD_DIM,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.num_scheduling_actions = num_scheduling_actions
        self.num_ris_actions = num_ris_actions
        self.constraint_dim = constraint_dim

    def __call__(self, records: Sequence[Mapping[str, Any]]) -> DistillationBatch:
        if not records:
            raise ValueError("cannot collate an empty record set")
        texts = [serialize_causal_observation(record["observation"]) for record in records]
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        arrays = build_distillation_targets(
            records,
            self.num_scheduling_actions,
            self.num_ris_actions,
            self.constraint_dim,
        )
        tensors = {name: torch.as_tensor(value) for name, value in arrays.items()}
        return DistillationBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            action_positions=attention_mask.to(dtype=torch.long).sum(dim=1).sub(1),
            **tensors,
        )


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.sum().clamp_min(1.0e-8)
    return (values * weights).sum() / denominator


def _inverse_sigmoid(values: Tensor) -> tuple[Tensor, Tensor]:
    clipped = values.clamp(1.0e-4, 1.0 - 1.0e-4)
    latent = torch.logit(clipped)
    log_abs_jacobian = torch.log(clipped) + torch.log1p(-clipped)
    return latent, log_abs_jacobian


def _inverse_tanh(values: Tensor) -> tuple[Tensor, Tensor]:
    clipped = values.clamp(-1.0 + 1.0e-4, 1.0 - 1.0e-4)
    latent = torch.atanh(clipped)
    log_abs_jacobian = torch.log1p(-(clipped * clipped))
    return latent, log_abs_jacobian


def continuous_action_nll(
    mean: Tensor,
    log_std: Tensor,
    targets: Tensor,
    target_weight: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Negative log likelihood in bounded action coordinates with Jacobians."""

    latent = torch.empty_like(targets)
    log_jacobian = torch.empty_like(targets)
    for index in SIGMOID_CONTINUOUS_INDICES:
        latent[..., index], log_jacobian[..., index] = _inverse_sigmoid(targets[..., index])
    latent[..., HEADING_INDEX], log_jacobian[..., HEADING_INDEX] = _inverse_tanh(
        targets[..., HEADING_INDEX]
    )
    expanded_mean = mean.unsqueeze(1)
    expanded_log_std = log_std.unsqueeze(1)
    normal_nll = 0.5 * (
        ((latent - expanded_mean) / expanded_log_std.exp()).square()
        + 2.0 * expanded_log_std
        + math.log(2.0 * math.pi)
    )
    per_target = (normal_nll - log_jacobian).sum(dim=-1)
    return _weighted_mean(
        (per_target * target_mask * target_weight).sum(dim=1),
        (target_mask * target_weight).sum(dim=1).clamp_min(1.0e-8),
    )


def compute_distillation_loss(
    outputs: HybridActionOutput,
    batch: DistillationBatch,
    weights: DistillationLossWeights,
    label_smoothing: float = 0.0,
) -> LossBreakdown:
    """Compute quality-weighted soft-target action distillation losses."""

    quality = batch.quality_weight
    pair_log_probs = torch.log_softmax(outputs.scheduling_logits, dim=-1)
    ris_log_probs = torch.log_softmax(outputs.ris_logits, dim=-1)
    pair_target = batch.pair_target
    ris_target = batch.ris_target
    if label_smoothing > 0.0:
        pair_target = (
            pair_target * (1.0 - label_smoothing) + label_smoothing / pair_target.shape[-1]
        )
        ris_target = ris_target * (1.0 - label_smoothing) + label_smoothing / ris_target.shape[-1]
    categorical_per_record = -(pair_target * pair_log_probs).sum(dim=-1)
    categorical_per_record -= (ris_target * ris_log_probs).sum(dim=-1)
    categorical = _weighted_mean(categorical_per_record, quality)
    continuous = continuous_action_nll(
        outputs.continuous_mean,
        outputs.continuous_log_std,
        batch.continuous_targets,
        batch.target_weight,
        batch.target_mask,
    )
    value = _weighted_mean(
        torch.nn.functional.smooth_l1_loss(outputs.value, batch.value_target, reduction="none"),
        quality,
    )
    constraint_mask_weight = batch.constraint_mask * quality.unsqueeze(-1)
    constraint_denominator = constraint_mask_weight.sum().clamp_min(1.0e-8)
    constraint = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.constraint_logits,
            batch.constraint_target,
            reduction="none",
        )
        * constraint_mask_weight
    ).sum() / constraint_denominator
    if float(batch.constraint_mask.sum().detach().cpu()) == 0.0:
        constraint = outputs.constraint_logits.sum() * 0.0
    auxiliary = outputs.value.sum() * 0.0
    total = (
        weights.categorical * categorical
        + weights.continuous * continuous
        + weights.value * value
        + weights.constraint * constraint
        + weights.auxiliary * auxiliary
    )
    return LossBreakdown(total, categorical, continuous, value, constraint, auxiliary)


def bounded_continuous_mean(mean: Tensor) -> Tensor:
    """Decode head means into the normalized seven-dimensional action vector."""

    output = mean.clone()
    for index in SIGMOID_CONTINUOUS_INDICES:
        output[..., index] = torch.sigmoid(output[..., index])
    output[..., HEADING_INDEX] = torch.tanh(output[..., HEADING_INDEX])
    return output


def prediction_metrics(outputs: HybridActionOutput, batch: DistillationBatch) -> dict[str, float]:
    """Compute action-agreement and bounded continuous error metrics."""

    pair_prediction = outputs.scheduling_logits.argmax(dim=-1)
    ris_prediction = outputs.ris_logits.argmax(dim=-1)
    quality = batch.quality_weight
    return {
        "pair_top1_accuracy": float(
            _weighted_mean((pair_prediction == batch.selected_pair).float(), quality).detach().cpu()
        ),
        "ris_top1_accuracy": float(
            _weighted_mean((ris_prediction == batch.selected_ris).float(), quality).detach().cpu()
        ),
        "continuous_mae": float(
            _weighted_mean(
                (bounded_continuous_mean(outputs.continuous_mean) - batch.selected_continuous)
                .abs()
                .mean(dim=-1),
                quality,
            )
            .detach()
            .cpu()
        ),
    }


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DistillationTrainer:
    """Mini-batch trainer with reproducible metrics and PEFT-friendly saves."""

    def __init__(
        self,
        model: GemmaStructuredStudent,
        tokenizer: Any,
        config: DistillationConfig,
        output_dir: str | Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(config.device)
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise ValueError("student model must expose a backbone")
        if hasattr(backbone, "hf_device_map"):
            try:
                backbone_device = next(backbone.parameters()).device
            except StopIteration as error:
                raise ValueError("quantized backbone has no parameters") from error
            self.model.action_head.to(backbone_device)
        else:
            self.model.to(self.device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("student has no trainable parameters")
        if config.optimizer.lower() != "adamw":
            raise ValueError(f"unsupported optimizer: {config.optimizer}")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.collator = DistillationCollator(
            tokenizer,
            config.max_sequence_length,
            config.num_scheduling_actions,
            config.num_ris_actions,
            config.constraint_dim,
        )
        self._metrics_path = self.output_dir / "metrics.jsonl"

    def _batches(
        self,
        records: Sequence[Mapping[str, Any]],
        shuffle: bool,
        seed: int,
    ) -> Iterable[Sequence[Mapping[str, Any]]]:
        indices = np.arange(len(records))
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(indices), self.config.batch_size):
            yield [records[int(index)] for index in indices[start : start + self.config.batch_size]]

    def _autocast(self) -> Any:
        if self.device.type != "cuda" or self.config.precision == "fp32":
            return torch.autocast(device_type=self.device.type, enabled=False)
        dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _run_epoch(
        self,
        records: Sequence[Mapping[str, Any]],
        epoch: int,
        training: bool,
    ) -> dict[str, float]:
        if not records:
            raise ValueError("cannot run an epoch with no records")
        self.model.train(training)
        totals: dict[str, float] = {}
        example_count = 0
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        step_count = 0
        total_batches = math.ceil(len(records) / self.config.batch_size)
        for step, records_batch in enumerate(
            self._batches(records, shuffle=training, seed=self.config.seed + epoch)
        ):
            batch = self.collator(records_batch).to(self.device)
            with torch.set_grad_enabled(training), self._autocast():
                outputs = self.model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    action_positions=batch.action_positions,
                )
                breakdown = compute_distillation_loss(
                    outputs,
                    batch,
                    self.config.loss_weights,
                    self.config.categorical_label_smoothing,
                )
            if training:
                (breakdown.total / self.config.gradient_accumulation_steps).backward()  # type: ignore[no-untyped-call]
                should_step = (
                    step + 1
                ) % self.config.gradient_accumulation_steps == 0 or step + 1 == total_batches
                if should_step:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in self.model.parameters()
                            if parameter.requires_grad
                        ],
                        self.config.gradient_clip_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    step_count += 1
                    if self.config.max_steps is not None and step_count >= self.config.max_steps:
                        break
            for name, value in breakdown.as_float_dict().items():
                totals[name] = totals.get(name, 0.0) + value * len(records_batch)
            for name, value in prediction_metrics(outputs, batch).items():
                totals[name] = totals.get(name, 0.0) + value * len(records_batch)
            example_count += len(records_batch)
        return {name: value / max(1, example_count) for name, value in totals.items()}

    def _save_checkpoint(self, tag: str, epoch: int, metrics: Mapping[str, Any]) -> Path:
        checkpoint = self.output_dir / "checkpoints" / tag
        checkpoint.mkdir(parents=True, exist_ok=True)
        backbone = getattr(self.model, "backbone", None)
        if hasattr(backbone, "save_pretrained"):
            cast(Any, backbone).save_pretrained(checkpoint / "backbone", safe_serialization=True)
            trainable_state = {
                name: parameter.detach().cpu()
                for name, parameter in self.model.action_head.named_parameters()
            }
            torch.save(trainable_state, checkpoint / "action_head.pt")
        else:
            state = {
                name: parameter.detach().cpu()
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            torch.save(state, checkpoint / "trainable_state.pt")
        torch.save(self.optimizer.state_dict(), checkpoint / "optimizer.pt")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(checkpoint / "tokenizer")
        metadata = {
            "epoch": epoch,
            "model_id": self.config.model_id,
            "model_revision": self.config.model_revision,
            "config": self.config.as_dict(),
            "trainable_parameter_count": self.model.trainable_parameter_count(),
            "metrics": dict(metrics),
        }
        (checkpoint / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return checkpoint

    def _log(self, payload: Mapping[str, Any]) -> None:
        with self._metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(payload), sort_keys=True) + "\n")

    def fit(
        self,
        train_records: Sequence[Mapping[str, Any]],
        validation_records: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Train until early stopping and return the reproducible history."""

        _seed_everything(self.config.seed)
        if not train_records or not validation_records:
            raise ValueError("both train and validation splits must be non-empty")
        history: list[dict[str, Any]] = []
        best_loss = math.inf
        best_epoch = -1
        epochs_without_improvement = 0
        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(train_records, epoch, training=True)
            validation_metrics = self._run_epoch(
                validation_records,
                epoch,
                training=False,
            )
            row: dict[str, Any] = {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
            history.append(row)
            self._log(row)
            validation_loss = float(validation_metrics["total"])
            self._save_checkpoint("last", epoch, row)
            if validation_loss < best_loss - 1.0e-8:
                best_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint("best", epoch, row)
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.early_stopping_patience:
                break
        summary = {
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "epochs_completed": len(history),
            "train_examples": len(train_records),
            "validation_examples": len(validation_records),
            "device": str(self.device),
            "history": history,
        }
        (self.output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary
