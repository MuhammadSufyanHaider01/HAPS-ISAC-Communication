# HAPS-ISAC Communication

Simulator-verified reasoning distillation and constrained reinforcement learning for causal, freshness-aware, secure HAPS-ISAC control.

## Current status

The validated Version 1 research environment is implemented. It includes fixed near/far NOMA pairs, causal communication and target sensing, target/eavesdropper reception, AoI/AoLI/AoSI, EKF tracking, CPU-delayed estimate availability, virtual queues, physical action completion, deterministic repair/fallback, reproducible state cloning, random and greedy baselines, and a reduced-grid one-step oracle.

Version 2 components—RIS, AAV jamming and mobility, residual self-interference, imperfect CSI/SIC, and stochastic blockage—remain disabled behind the configuration contract. Teacher integration, distillation, and constrained PPO have not started.

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

Python 3.11 is the target runtime.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
```

Run all quality gates:

```bash
.venv/bin/ruff check src tests scripts
.venv/bin/mypy src/haps_isac
.venv/bin/pytest -q
.venv/bin/python scripts/validate_environment.py --steps 5000
```

The validation command checks deterministic replay, candidate-evaluation isolation, observation/action invariants, random and urgency-greedy stress rollouts, hard feasibility, repair/fallback rates, and a common-seed one-step grid oracle.
