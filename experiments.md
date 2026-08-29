# Experiment Plan

This file records the planned comparison suite for the simulator-verified teacher-to-student HAPS-ISAC optimization pipeline. It is intentionally a living document; additional experiments, ablations, robustness tests, and reporting decisions should be added here as the implementation progresses.

## Proposed Method

The proposed controller is the distilled 4B-class `google/gemma-4-E4B-it` student. The student receives the causal HAPS-ISAC state, uses a structured hybrid-action head for categorical and continuous decisions, and is adapted with QLoRA by default or BF16 LoRA when memory permits. The base model is frozen; the trainable parameters are the LoRA adapters and action/value/constraint heads. After offline action distillation, the same checkpoint may be refined with constrained PPO in the simulator.

The main proposed variants are:

- distilled student before PPO;
- distilled student with active teacher correction;
- distilled student after constrained PPO refinement.

## Baseline Families

The following five baseline families are planned for comparison against the proposed distilled student. Exact algorithm choices within each family will be finalized after the Version 1 simulator and action interface are frozen.

### 1. Exhaustive / Near-Optimal Reference

Use exhaustive enumeration or a reduced-grid search over the discrete and discretized continuous action space for small problem instances (small numbers of NOMA clusters). If tractable, add a short-horizon exact or rolling-horizon reference.

Purpose:

- provide an empirical optimality or near-optimality ceiling;
- quantify the gap of the teacher, distilled student, and PPO-refined student;
- validate that the simulator and objective implementation are internally consistent.

This is a reference method rather than a scalable deployment baseline. It should be restricted to instances for which the search is demonstrably complete or clearly labelled as a reduced-grid approximation.

### 2. Model-Based Wireless Optimization

Implement strong model-based methods using the same system model, action completion, safety repair, and constraints. Candidate methods include:

- Lyapunov drift-plus-penalty for long-term freshness, secrecy, energy, and reliability constraints;
- alternating optimization, successive convex approximation, WMMSE, or block-coordinate descent for per-slot wireless decisions;
- a rolling-horizon model-predictive-control variant where computationally feasible.

The preferred reference is an integrated long-horizon method such as Lyapunov drift-plus-penalty with an AO/SCA/WMMSE inner solver. A simple one-step optimizer may be retained as a secondary diagnostic, but should not be the primary model-based claim.

Purpose:

- compare against interpretable, physics/model-based optimization;
- test whether the learned policy captures long-horizon freshness and constraint effects;
- establish a strong wireless baseline rather than only comparing against generic learning methods.

### 3. AI-Based State-of-the-Art Approaches from Recent HAPS Papers

Select the strongest relevant AI-based methods from recent HAPS, HAPS-ISAC, NOMA, RIS, secrecy, or freshness-aware wireless-optimization papers. Reimplement or adapt them to the current simulator while preserving their algorithmic identity.

Possible categories include:

- supervised or imitation-based resource-allocation networks;
- centralized or multi-agent deep reinforcement learning;
- graph, set, attention, or permutation-equivariant policies;
- constrained or multi-objective learning methods used for comparable HAPS/ISAC decisions.

Selection criteria:

- relevance to the same decision variables and system constraints;
- sufficiently complete algorithmic details for a fair implementation;
- comparable observation information and training interaction budget;
- identical simulator dynamics, channel assumptions, completion, and repair rules.

The paper should identify these as literature baselines and clearly distinguish direct reimplementations from adaptations required by the present action space.

### 4. Reinforcement Learning

Use a strong RL baseline trained without teacher initialization:

- constrained PPO from scratch with the same structured hybrid-action architecture as the student;
- optionally, one suitable hybrid-action off-policy method such as SAC/TD3 with a clearly documented discrete-action treatment.

The mandatory comparison is PPO from scratch. It isolates the value of teacher distillation, while keeping the policy class, environment interaction budget, safety layer, and optimization framework comparable.

Additional offline-learning comparisons may be added later, such as quality-weighted behavior cloning versus CQL, IQL, or TD3+BC on the verified teacher dataset.

### 5. Teacher Model

Evaluate the Qwen3.5-27B teacher directly in the environment after the same physics completion and safety verification used for the dataset. Report at least:

- single proposal (`K=1`);
- verified best-of-`K` candidate selection;
- teacher inference cost, candidate count, tokens, and latency.

The teacher is an upper-quality reference and not a deployment-equivalent baseline. Its purpose is to measure how closely the distilled student preserves simulator-verified teacher performance and whether PPO can improve long-horizon performance beyond teacher imitation.

## Common Evaluation Protocol

All methods should use the same:

- HAPS-ISAC simulator version and enabled system components;
- state definition, action semantics, physics completion, safety repair, and fallback rules;
- train/validation/test scenario split;
- random scenario bank and common random numbers where applicable;
- evaluation episode lengths, seeds, and stopping criteria;
- hyperparameter-tuning and environment-interaction budget, reported transparently.

Primary system-level metrics are AoI, AoLI, AoSI, secrecy rate/outage, sensing accuracy or tracking uncertainty, reliability, energy, weighted system cost, hard-feasible rate, constraint-violation probability/CVaR, repair distance, and fallback frequency. Action agreement and training losses are secondary diagnostics.

Every main comparison should report confidence intervals across independent seeds and held-out episodes. Runtime, peak memory, trainable parameter count, and end-to-end per-slot latency should be included for the teacher and student comparisons.

## Planned Main Comparison Table

The initial main table is expected to contain:

1. exhaustive/reduced-grid reference (small instances only);
2. integrated model-based optimizer;
3. selected recent HAPS/ISAC AI baseline(s);
4. PPO from scratch;
5. Qwen3.5-27B direct teacher;
6. distilled Gemma student;
7. distilled Gemma plus constrained PPO.

Random and simple greedy policies remain useful simulator sanity checks, but are not intended to be headline comparisons. They may be reported in an appendix or diagnostic table.

## Open Items for Future Updates

- finalize the exact model-based solver and literature AI baselines;
- decide whether the optional off-policy hybrid RL baseline is sufficiently fair to implement;
- add ablations for soft versus hard teacher targets, verification horizon, candidate count, active correction, QLoRA versus LoRA, and the PPO distillation anchor;
- add scaling and OOD tests over cluster count, spatial dispersion, channel/CSI conditions, traffic, HAP geometry, and target/eavesdropper configurations;
- define the final paper figures, statistical tests, and compute budget.
