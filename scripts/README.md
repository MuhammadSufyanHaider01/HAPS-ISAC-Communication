# Experiment entry points

Implemented and tested:

- `validate_environment.py`: Version 1 simulator and baseline stress validation.
- `generate_demonstrations.py`: cached teacher querying, verification, selection, and linked logs.
- `audit_demonstrations.py`: dataset integrity and teacher-quality gate.
- `benchmark_teachers.py`: frozen-state, common-random-number teacher tournament.
- `slurm/`: non-launching vLLM and combined dataset-generation job templates.

Reserved for later phases:

- `train_student.py`
- `collect_active_queries.py`
- `train_constrained_ppo.py`
- `evaluate_all.py`

Reserved commands remain placeholders until their underlying learning modules are implemented and tested.
