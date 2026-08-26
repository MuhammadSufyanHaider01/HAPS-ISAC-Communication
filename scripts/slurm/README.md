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

Generate a bounded, fully verified smoke dataset in one allocation:

~~~bash
HAPS_TEACHER_BACKEND=transformers \
HAPS_PYTHON_ENVIRONMENT=.venv-gpu \
HAPS_DATASET_STATES=2 \
HAPS_RUN_ID=qwen-smoke-001 \
  scripts/slurm/submit_teacher_dataset.sh
~~~

The wrapper records the commit and dirty state on the login node so queued jobs retain
reproducible source metadata even when Git is unavailable on compute nodes.

Scale only after the smoke dataset passes its audit. Runtime overrides include
`HAPS_TEACHER_MODEL`, `HAPS_TEACHER_REVISION`, `HAPS_TEACHER_BACKEND`,
`HAPS_DATASET_STATES`, `HAPS_OUTPUT_DIRECTORY`, and `HAPS_LOG_FLUSH_EVERY`.
Never place Hugging Face or API tokens in an sbatch file.

Each generation job retains:

- six linked JSONL tables and their Parquet mirrors for states, requests, all K candidates,
  rollouts, selections, and demonstrations. Rollouts are labeled as teacher candidates or
  matched greedy/random baselines and retain their common seed for paired analysis;
- `teacher_server_metrics.jsonl` for request latency, token throughput, failures, and peak memory;
- `teacher_server.stderr.log` or `vllm.log` for inference diagnostics;
- timestamped `gpu_metrics.csv` utilization, memory, power, and temperature samples;
- `audit.json` with quality gates and aggregate metrics;
- `teacher_quality_report.json` with proposed/executed diversity, selected-action repair and
  fallback rates, paired selection confidence intervals, difficulty, and matched-horizon
  baseline diagnostics; and
- the Slurm stdout log under `results/`.

All canonical table records include UTC timestamps. Smoke jobs flush after every state by default;
