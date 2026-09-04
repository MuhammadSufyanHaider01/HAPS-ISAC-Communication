#!/usr/bin/env python3
"""Train the Gemma 4 E4B structured-action student by offline distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from haps_isac.data.dataset_loader import DatasetLoader
from haps_isac.models.student_actor import GemmaStructuredStudent
from haps_isac.training.distill_trainer import DistillationConfig, DistillationTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="demonstration-only view produced by prepare_distillation_view.py",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/distillation.yaml"),
        help="Gemma/PEFT and optimization YAML configuration",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-config", type=Path, default=Path("configs/system_v1.yaml"))
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--method",
        choices=("qlora", "lora"),
        help="override PEFT method; qlora is the planned default",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the dataset/config and print a training plan without loading Gemma",
    )
    return parser


def _load_system_num_pairs(path: Path) -> int:
    import yaml

    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    try:
        return int(payload["system"]["num_noma_pairs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"could not read system.num_noma_pairs from {path}") from error


def _infer_num_ris_actions(records: list[dict[str, Any]]) -> int:
    values = [
        int(target.get("action", {}).get("ris_code", 0))
        for record in records
        for target in (
            record.get("target_candidates")
            or [{"action": record.get("selected_action", {"ris_code": 0})}]
        )
    ]
    return max(values, default=0) + 1


def _infer_num_scheduling_actions(records: list[dict[str, Any]]) -> int:
    values = [
        int(target.get("action", {}).get("pair", 0))
        for record in records
        for target in (
            record.get("target_candidates")
            or [{"action": record.get("selected_action", {"pair": 0})}]
        )
    ]
    return max(values, default=0) + 1


def _apply_overrides(
    config: DistillationConfig,
    arguments: argparse.Namespace,
) -> DistillationConfig:
    updates: dict[str, Any] = {}
    if arguments.model_id:
        updates["model_id"] = arguments.model_id
    if arguments.model_revision:
        updates["model_revision"] = arguments.model_revision
    if arguments.batch_size:
        updates["batch_size"] = arguments.batch_size
    if arguments.epochs:
        updates["max_epochs"] = arguments.epochs
    if arguments.max_steps:
        updates["max_steps"] = arguments.max_steps
    if arguments.device:
        updates["device"] = arguments.device
    if arguments.method:
        updates["peft_method"] = arguments.method
        updates["load_in_4bit"] = arguments.method == "qlora"
    if not updates:
        return config
    from dataclasses import replace

    return replace(config, **updates)


def _training_plan(
    config: DistillationConfig,
    dataset: Path,
    train_count: int,
    validation_count: int,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset),
        "model_id": config.model_id,
        "model_revision": config.model_revision,
        "peft_method": config.peft_method,
        "load_in_4bit": config.load_in_4bit,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_examples": train_count,
        "validation_examples": validation_count,
        "device": config.device,
        "precision": config.precision,
        "trainable_policy": "LoRA adapters plus structured action/value/constraint heads",
    }


def _is_lora_compatible_module(module: torch.nn.Module) -> bool:
    """Return whether PEFT can replace the module with a LoRA layer."""

    # bitsandbytes quantized layers are not torch.nn.Linear subclasses, but
    # PEFT has dedicated dispatchers for these concrete layer types.
    return isinstance(module, torch.nn.Linear) or module.__class__.__name__ in {
        "Linear4bit",
        "Linear8bitLt",
    }


def _resolve_lora_target_modules(
    backbone: torch.nn.Module,
    configured_targets: tuple[str, ...] | list[str],
) -> tuple[str, ...]:
    """Resolve configured projection names through Gemma4 wrapper modules.

    Transformers 5.x wraps several Gemma4 projections in
    ``Gemma4ClippableLinear``.  PEFT cannot replace that wrapper directly,
    while its nested ``linear`` child is a supported ``Linear`` or
    ``Linear4bit`` layer.  Returning exact module paths avoids accidentally
    matching the unsupported wrapper elsewhere in the model.
    """

    targets = tuple(str(value) for value in configured_targets)
    resolved: set[str] = set()
    for name, module in backbone.named_modules():
        if not name:
            continue
        if not any(name == target or name.endswith(f".{target}") for target in targets):
            continue
        if _is_lora_compatible_module(module):
            resolved.add(name)
            continue
        nested_linear = getattr(module, "linear", None)
        if isinstance(nested_linear, torch.nn.Module) and _is_lora_compatible_module(nested_linear):
            resolved.add(f"{name}.linear")
    if not resolved:
        raise RuntimeError(
            "none of the configured LoRA target modules were found as supported linear layers; "
            f"requested={list(targets)}"
        )
    return tuple(sorted(resolved))


def _load_transformers_student(
    config: DistillationConfig,
) -> tuple[Any, GemmaStructuredStudent]:
    try:
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Transformers is required; install the [student] optional dependency"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    action_token = "<ACTION>"
    # Transformers 5.x GemmaTokenizer does not expose the legacy
    # ``additional_special_tokens`` attribute.  ``getattr`` keeps this
    # compatible with both tokenizer implementations; add_special_tokens is
    # idempotent when the token is already registered.
    additional_special_tokens = getattr(tokenizer, "additional_special_tokens", ())
    if action_token not in additional_special_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [action_token]})

    dtype = torch.bfloat16 if config.compute_dtype.lower() == "bfloat16" else torch.float16
    model_kwargs: dict[str, Any] = {
        "revision": config.model_revision,
        "torch_dtype": dtype,
    }
    if config.device == "auto" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
    if config.peft_method.lower() == "qlora":
        if not config.load_in_4bit:
            raise ValueError("QLoRA requires peft.load_in_4bit=true")
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError("Transformers was built without BitsAndBytesConfig") from error
        try:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as error:
            raise RuntimeError(
                "QLoRA requires a working bitsandbytes installation; use --method lora "
                "for an unquantized fallback"
            ) from error
    try:
        backbone = AutoModel.from_pretrained(config.model_id, **model_kwargs)
    except (OSError, ValueError, NotImplementedError):
        backbone = AutoModelForCausalLM.from_pretrained(config.model_id, **model_kwargs)
    if hasattr(backbone, "resize_token_embeddings"):
        backbone.resize_token_embeddings(len(tokenizer))

    if config.peft_method.lower() not in {"qlora", "lora"}:
        raise ValueError("peft.method must be either qlora or lora")
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise RuntimeError(
            "PEFT is required for Gemma distillation; install the [student] optional dependency"
        ) from error
    if config.peft_method.lower() == "qlora":
        backbone = prepare_model_for_kbit_training(
            backbone,
            use_gradient_checkpointing=config.gradient_checkpointing,
        )
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(_resolve_lora_target_modules(backbone, config.target_modules)),
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    backbone = get_peft_model(backbone, lora)
    if config.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "config"):
        backbone.config.use_cache = False

    student = GemmaStructuredStudent(
        backbone,
        num_scheduling_actions=config.num_scheduling_actions,
        num_ris_actions=config.num_ris_actions,
        constraint_dim=config.constraint_dim,
        action_token_id=tokenizer.convert_tokens_to_ids(action_token),
        bottleneck_size=config.bottleneck_size,
    )
    return tokenizer, student


def main() -> None:
    arguments = build_parser().parse_args()
    config = _apply_overrides(
        DistillationConfig.from_yaml(arguments.config),
        arguments,
    )
    loader = DatasetLoader(arguments.dataset)
    records = list(loader.iter_table("demonstrations"))
    if not records:
        raise ValueError(f"no demonstrations found in {arguments.dataset}")
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "validation"]
    from dataclasses import replace

    config = replace(
        config,
        num_scheduling_actions=max(
            config.num_scheduling_actions,
            _load_system_num_pairs(arguments.system_config) + 1,
            _infer_num_scheduling_actions(records),
        ),
        num_ris_actions=max(config.num_ris_actions, _infer_num_ris_actions(records)),
    )
    plan = _training_plan(config, arguments.dataset, len(train_records), len(validation_records))
    if arguments.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    tokenizer, student = _load_transformers_student(config)
    trainer = DistillationTrainer(student, tokenizer, config, arguments.output)
    summary = trainer.fit(train_records, validation_records)
    plan["trainable_parameter_count"] = student.trainable_parameter_count()
    plan["summary"] = summary
    (arguments.output / "run_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
