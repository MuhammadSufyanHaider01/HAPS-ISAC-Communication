# Experiment entry points

Implemented and tested:

- `validate_environment.py`: Version 1 simulator and baseline stress validation.
- `generate_demonstrations.py`: cached teacher querying, adaptive verification, soft labels, stratified states, and resumable shards.
- `merge_teacher_shards.py`: validated shard merge, aggregate audit, and scale-up gate enforcement.
- `audit_demonstrations.py`: dataset integrity and teacher-quality gate.
- `report_teacher_quality.py`: repair, confidence, state-difficulty, and baseline diagnostics.
- `prepare_distillation_view.py`: filter the merged logs to demonstrations with valid soft targets without new teacher queries.
- `benchmark_teachers.py`: frozen-state, common-random-number teacher tournament.
- `slurm/`: non-launching teacher serving, resumable GPU arrays, and dependent CPU merge templates.

Reserved for later phases:

- `train_student.py` (Gemma 4 E4B action distillation with QLoRA/LoRA adapters)
- `collect_active_queries.py`
- `train_constrained_ppo.py`
- `evaluate_all.py`

Reserved commands remain placeholders until their underlying learning modules are implemented and tested.
