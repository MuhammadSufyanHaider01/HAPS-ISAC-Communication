# HAPS-ISAC Communication

Simulator-verified reasoning distillation and constrained reinforcement learning for causal, freshness-aware, secure HAPS-ISAC control.

## Current status

The repository is scaffolded according to the implementation roadmap. Simulator and learning behavior have not been implemented yet.

## Design documents

- [System and signal model](HAPS_ISAC_Freshness_System_Model.pdf)
- [Verified reasoning-distillation specification](Verified_Reasoning_Distillation_HAPS_ISAC.pdf)
- [Implementation plan](plan.md)

## Planned pipeline

```text
causal numerical state
  -> reasoning-teacher candidates
  -> physics completion and safety repair
  -> stochastic rollout verification
  -> verified demonstration dataset
  -> numerical student distillation
  -> active correction
  -> constrained PPO refinement
```

The large reasoning model is an offline candidate proposer. The simulator is the authoritative evaluator, and deployment uses only the numerical student, deterministic physics completion, and the safety layer.

## Repository layout

- `configs/`: versioned system, teacher, training, and evaluation configuration.
- `src/haps_isac/`: simulator, physics, control, verification, data, and learning packages.
- `scripts/`: command-line experiment entry points.
- `tests/`: unit, integration, oracle, and reproducibility tests.
- `datasets/`, `checkpoints/`, `results/`: generated artifacts, excluded from Git.
- `notebooks/`: exploratory analysis only; production logic belongs under `src/`.

## Development setup

Python 3.11 is the target runtime. Dependency locking and the first installable development environment are part of Phase 0 in `plan.md`.
