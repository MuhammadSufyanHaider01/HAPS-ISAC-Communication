# Slurm teacher jobs

These templates target the GPU partitions visible on the ARC cluster. They do not launch automatically.

The default generation backend is direct Transformers inference with a CUDA 12.6 PyTorch wheel.
Create the environment once from the repository root:

~~~bash
module load python/3.12.5
python3 -m venv --copies .venv-gpu
.venv-gpu/bin/pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.12.1
.venv-gpu/bin/pip install --index-url https://download.pytorch.org/whl/cu126 --no-deps torchvision==0.27.1
.venv-gpu/bin/pip install -e '.[data,teacher]'
~~~

The job keeps Hugging Face and PyTorch caches under
`/home/muhammadsufyan.haide/.cache/haps-isac` by default, outside the repository.

Serve a model for interactive testing:

~~~bash
HAPS_TEACHER_MODEL=Qwen/Qwen3.5-27B \
  sbatch scripts/slurm/serve_teacher.sbatch
~~~

Generate a bounded diagnostic shard in one allocation:

~~~bash
HAPS_TEACHER_BACKEND=transformers \
HAPS_PYTHON_ENVIRONMENT=.venv-gpu \
HAPS_DATASET_STATES=2 \
HAPS_RUN_ID=qwen-smoke-001 \
  scripts/slurm/submit_teacher_dataset.sh
~~~

The single-job wrapper permits a dirty tree for diagnostics. It records the commit and dirty state
on the login node so queued jobs retain reproducible source metadata even when Git is unavailable
on compute nodes. Its default output is
`datasets/qwen-smoke-001/shards/shard-000`.

## Qualification and production workflow

The workflow wrapper requires a clean commit, submits a resumable GPU array, then submits a CPU
merge job with an `afterok` dependency. The merge validates that every shard used the same clean
commit, configuration hash, master seed, and complete non-overlapping global state range. It then
runs the integrity audit and aggregate scale-up gates.

Run the next 200-state qualification before production:

~~~bash
HAPS_TEACHER_BACKEND=transformers \
HAPS_PYTHON_ENVIRONMENT=.venv-gpu \
HAPS_DATASET_STATES=200 \
HAPS_SHARD_COUNT=1 \
HAPS_RUN_ID=qwen-v1.4-qualification-200 \
  scripts/slurm/submit_teacher_dataset_workflow.sh
~~~

Only if the merged qualification report has `audit_passed=true` and `scale_up_passed=true`, submit
the planned 5,000-state dataset. Ten 500-state shards fit independently within the 24-hour GPU
limit; the concurrency cap can be adjusted to match available GPUs.

~~~bash
HAPS_TEACHER_BACKEND=transformers \
HAPS_PYTHON_ENVIRONMENT=.venv-gpu \
HAPS_DATASET_STATES=5000 \
HAPS_SHARD_COUNT=10 \
HAPS_MAX_PARALLEL_SHARDS=2 \
HAPS_RUN_ID=qwen-v1.4-production-5000 \
  scripts/slurm/submit_teacher_dataset_workflow.sh
~~~

The wrapper prints both Slurm job IDs. Closing the SSH window does not stop submitted jobs. Every
array task defaults to `HAPS_RESUME=1`: resubmitting the same clean code, run ID, state count, seed,
and shard plan trims an interrupted shard to its last complete state and continues. A resume is
rejected if any provenance field differs. Do not set one shared `HAPS_OUTPUT_DIRECTORY` for a
multi-task array; set `HAPS_DATASET_ROOT` if the whole dataset needs a custom location.

The default layout is:

~~~text
datasets/<run-id>/
  shards/shard-000/ ... shard-NNN/
  merged/
    manifest.json
    shards.json
    audit.json
    teacher_quality_report.json
~~~

For a completed run whose raw shards are valid but whose scale-up gate is below
the production threshold, the merge can be retained for diagnosis by setting
`HAPS_ALLOW_FAILED_GATES=1`. The canonical JSONL tables and quality report are
then preserved; the flag never changes the reported metrics. To create a
distillation-only view without issuing new teacher queries, run:

~~~bash
.venv/bin/python scripts/prepare_distillation_view.py \
  datasets/<run-id>/merged \
  datasets/<run-id>/distillation
~~~

The view contains only demonstrations with valid normalized soft targets and
records the excluded-state count in `distillation_manifest.json`.

## Gemma 4 E4B distillation

train_student.py serializes the causal numerical observation, appends the
ACTION sentinel, and trains only LoRA adapters plus the structured action,
value, and constraint heads. Teacher reasoning is never loaded into the
student input. The default configuration uses QLoRA NF4, BF16 compute, batch
size 8, and gradient accumulation of 8.

Validate the dataset and resolved training plan without downloading Gemma:

~~~bash
.venv/bin/python scripts/train_student.py \
  --dataset datasets/qwen3.5-27b-production-5000-v2.2-001/distillation \
  --config configs/distillation.yaml \
  --output /tmp/haps-gemma-plan \
  --dry-run
~~~

Submit the 24-hour GPU training job after installing the student extra
(peft and bitsandbytes):

~~~bash
HAPS_DISTILL_DATASET=datasets/qwen3.5-27b-production-5000-v2.2-001/distillation \
HAPS_PYTHON_ENVIRONMENT=.venv-gpu \
sbatch scripts/slurm/train_student.sbatch
~~~

Override the Slurm partition with sbatch --partition=<available-gpu-partition>
when the default H100 partition is occupied. Check metrics.jsonl,
training_summary.json, and checkpoints/best/ in the output directory.

Runtime overrides include
`HAPS_TEACHER_MODEL`, `HAPS_TEACHER_REVISION`, `HAPS_TEACHER_BACKEND`,
`HAPS_DATASET_STATES`, `HAPS_SHARD_COUNT`, `HAPS_MAX_PARALLEL_SHARDS`,
`HAPS_DATASET_ROOT`, `HAPS_LOG_FLUSH_EVERY`, and `HAPS_RESUME`.
Never place Hugging Face or API tokens in an sbatch file.

Each generation job retains:

- six linked JSONL tables and their Parquet mirrors for states, requests, all K candidates,
  rollouts, selections, and demonstrations. Rollouts are labeled as teacher candidates or
  matched greedy/random baselines and retain their common seed for paired analysis. Demonstrations
  include normalized multi-candidate soft targets for practically equivalent actions;
- `teacher_server_metrics.jsonl` for request latency, token throughput, failures, and peak memory;
- `teacher_server.stderr.log` or `vllm.log` for inference diagnostics;
- timestamped `gpu_metrics.csv` utilization, memory, power, and temperature samples;
- `audit.json` with linked-table, global-index, split, soft-target, and rollout-transaction checks;
- `teacher_quality_report.json` with proposed/executed diversity, selected-action repair and
  fallback rates, raw confidence overlap, decisive/equivalent/unresolved decisions, adaptive
  rollout counts, target entropy, state/split quota error, duplication, difficulty, and
  matched-horizon baseline diagnostics; and
- the Slurm stdout log under `results/`.

The aggregate scale-up decision gates on unresolved selections rather than treating statistically
overlapping but practically equivalent actions as failures. It also requires valid normalized
soft targets, no duplicate causal states, exact sampling within tolerance, 100% hard feasibility,
bounded repair/fallback rates, and verified baseline wins. Raw selection uncertainty remains in
the report for plotting.

All canonical table records include UTC timestamps. Qualification jobs should flush after every
state; production can raise `HAPS_LOG_FLUSH_EVERY` to reduce filesystem overhead.
