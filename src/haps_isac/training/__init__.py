"""Offline distillation, active correction, and constrained PPO training."""

from haps_isac.training.distill_trainer import (
    DistillationBatch,
    DistillationConfig,
    DistillationLossWeights,
    DistillationTrainer,
    compute_distillation_loss,
)

__all__ = [
    "DistillationBatch",
    "DistillationConfig",
    "DistillationLossWeights",
    "DistillationTrainer",
    "compute_distillation_loss",
]
