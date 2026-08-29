# HAPS-ISAC Communication

Simulator-verified reasoning distillation and constrained reinforcement learning for causal, freshness-aware, secure HAPS-ISAC control.

## Current status

The validated Version 1 research environment is implemented. It includes fixed near/far NOMA pairs, causal communication and target sensing, target/eavesdropper reception, AoI/AoLI/AoSI, EKF tracking, CPU-delayed estimate availability, virtual queues, physical action completion, deterministic repair/fallback, reproducible state cloning, random and greedy baselines, and a reduced-grid one-step oracle.

The offline teacher-generation phase is implemented through schema v5: Qwen/Gemma-compatible adapters, causal versioned prompts, strict parsing, content-addressed caching, adaptive common-random rollout verification, practical-equivalence soft labels, exact stratified sampling, plotting-ready linked logs, crash-resumable shards, validated merging, dataset auditing, and Slurm workflows. The complete path passes deterministic mock-teacher resume/merge tests; the final production Qwen dataset has not yet been generated.

Version 2 components—RIS, AAV jamming and mobility, residual self-interference, imperfect CSI/SIC, and stochastic blockage—remain disabled behind the configuration contract. Student distillation, active correction, and constrained PPO have not started. The planned primary student is the open-weight Gemma 4 E4B instruction model, adapted with QLoRA/LoRA; it will consume the causal state and learn verified numerical actions, not teacher reasoning text. A compact numerical policy remains an optional low-latency ablation.

## Design documents

- [System and signal model](HAPS_ISAC_Freshness_System_Model.pdf)
- [Verified reasoning-distillation specification](Verified_Reasoning_Distillation_HAPS_ISAC.pdf)
- [Implementation plan](plan.md)

## Planned pipeline

```text
causal numerical state
  -> reasoning-teacher candidates
  -> physics completion and safety repair
  -> adaptive stochastic rollout verification
  -> weighted verified demonstration dataset
  -> Gemma 4 E4B QLoRA/LoRA action distillation
  -> active correction
  -> constrained PPO refinement
```

The large reasoning model is an offline candidate proposer. The simulator is the authoritative evaluator. The primary student is a parameter-efficiently adapted Gemma 4 E4B model with a structured hybrid-action head; deployment uses that student, deterministic physics completion, and the safety layer. A smaller numerical-only policy may be evaluated separately for latency and memory.

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

When implementing student distillation, install the optional PEFT stack with `.venv/bin/python -m pip install -e '.[student]'`; it provides `peft` and `bitsandbytes` for QLoRA/LoRA alongside the existing transformer dependency.

The validation command checks deterministic replay, candidate-evaluation isolation, observation/action invariants, random and urgency-greedy stress rollouts, hard feasibility, repair/fallback rates, and a common-seed one-step grid oracle.

## Teacher dataset workflow

Validate the complete pipeline locally without contacting a model:

~~~bash
.venv/bin/python scripts/generate_demonstrations.py \
  --provider mock --states 3 --candidates 4 --rollouts 2 --horizon 3 \
  --shortlist 2 --no-parquet --run-id smoke-v1 \
  --output /tmp/haps-teacher-smoke
.venv/bin/python scripts/audit_demonstrations.py /tmp/haps-teacher-smoke
~~~

Compare live teachers on one frozen state bank by repeating the four-value teacher option:

~~~bash
.venv/bin/python scripts/benchmark_teachers.py \
  --teacher qwen qwen Qwen/Qwen3.5-27B http://qwen-host:8000/v1 \
  --teacher gemma gemma GEMMA_MODEL_ID http://gemma-host:8000/v1 \
  --states 2000 --output results/teacher-tournament
~~~

See [scripts/slurm/README.md](scripts/slurm/README.md) for non-launching ARC Slurm templates. Always audit a small pilot before requesting the full demonstration budget.
