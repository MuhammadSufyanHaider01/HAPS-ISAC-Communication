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

## Training-Convergence Diagnostics

Optionally log and plot the total distillation loss during offline student training, together with its categorical, continuous-action, value, constraint, and auxiliary components when available. Report training and held-out validation curves, the step/epoch of the best checkpoint, and whether loss convergence corresponds to improved verified system-level performance and feasibility. These curves are convergence diagnostics, not substitutes for AoI, AoLI, AoSI, secrecy, sensing, energy, and constraint results.

## Core Wireless Knowledge-Creation Experiments

The central paper figures should establish physical-layer and cross-layer operating principles, rather than only rank learning algorithms. Use an exhaustive or strong model-based reference on small instances to validate each physical effect, then show that the proposed student reproduces it at deployment scale. The student is the scalable controller; the wireless-system insight is the contribution.

### 1. NOMA versus OMA versus Semi-NOMA Sum-Rate Study (Highest Priority)

Compare pure power-domain NOMA, OMA, and a formally defined downlink semi-NOMA mode under the same total time-bandwidth, HAPS transmit-power, sensing-resource, completion, and safety constraints. This initial study is intentionally restricted to the network sum rate; freshness, secrecy, sensing, energy, and feasibility metrics belong to separate experiments.

Vary the number of served NOMA clusters \(N\) while evaluating all three access modes on the same topology/channel scenario bank. Keep the total system bandwidth and HAPS power fixed, specify whether offered traffic is held per cluster or in total, and use the same near/far pairing rule or association policy wherever it applies.

The main result is average network sum rate versus \(N\), with confidence intervals. For semi-NOMA, optimize or sweep the orthogonal resource fraction \(\eta\), where \(\eta=1\) gives OMA and \(\eta=0\) gives pure NOMA; optionally report the sum-rate-optimal \(\eta^*\) versus \(N\).

The sum-rate crossover points identify the studied regimes in which OMA is no longer sum-rate preferred and where semi-NOMA or pure NOMA provides the larger gain. State this narrowly as a sum-rate result, not as a general dominance claim.

Prerequisite: add comparable OMA and downlink semi-NOMA control paths to the simulator/action-completion layer before final student training.

### 2. Sensing-Secrecy-Tracking Geometry Map

Study the coupling between sensing effort, target tracking, and physical-layer secrecy while keeping the total resource budget fixed.

Vary:

- target motion/process noise or tracking difficulty;
- target-to-eavesdropper angular or spatial separation; and
- legitimate-cluster/eavesdropper geometry.

Report tracking RMSE or covariance trace, secrecy rate/outage, AoLI/AoSI, energy, and the sensing-versus-communication allocation. The desired result is a geometry-dependent map that identifies when extra sensing improves mission freshness through better estimation and when it causes an unacceptable secrecy cost.

### 3. Cross-Layer Freshness Bottleneck Map

Study whether stale information is dominated by radio service, sensing/estimation, or CPU-delayed estimate availability. Jointly vary offered update load, CPU service delay/capacity, and radio quality.

Report tail AoLI/AoSI, virtual-queue stability, energy, tracking uncertainty, and the marginal benefit of additional radio versus computing resources. Classify each operating point as radio-limited, compute-limited, or sensing-limited; a key result is the regime where additional transmit power cannot improve AoLI because computing delay is dominant.

### 4. HAPS Freshness-Security Feasibility Region

Construct an empirical service/feasibility region by jointly varying offered update load and a HAPS provisioning dimension, such as coverage radius, transmit-power budget, or computing capacity. Use secrecy/reliability requirements as separate panels or constraints.

For every operating point, report bounded virtual queues, hard-feasibility, and tail freshness/security targets. Compare the proposed controller with the strongest model-based and RL references. Describe this as an empirical stable-feasible region, not as an information-theoretic capacity region unless it is analytically derived. The map should yield deployment guidance for meeting a stated freshness and secrecy service-level agreement.

### 5. Version 2 Robustness Boundary (Future Extension)

After imperfect CSI, residual SIC, blockage, or mobility is enabled, map the boundary over CSI error and interference/blockage severity. Report tail freshness, secrecy outage, feasibility, repair distance, and fallback frequency to identify where nominal NOMA-ISAC control becomes unreliable.

## Supporting Algorithmic Experiments

These studies are important evidence for the teacher-to-student method, but are secondary to the wireless knowledge-creation experiments above.

### 1. Distillation versus Reinforcement Learning

Compare PPO from scratch, quality-weighted distillation, and distillation followed by constrained PPO using the same policy interface, safety layer, scenario split, and environment-interaction budget. Report verified system performance against training interactions, teacher-data budget, convergence behavior, and constraint outcomes. This isolates the value of verified teacher supervision and PPO refinement.

### 2. Deployment Efficiency

Compare direct Qwen teacher control, the strongest model-based optimizer, the Gemma student, and the optional numerical-only policy. Report end-to-end per-slot p50/p95 latency, memory, trainable and total parameter counts, throughput, deadline-miss rate, and verified system performance. Present this as a performance-latency-memory Pareto result.

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
