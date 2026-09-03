#!/usr/bin/env bash
#SBATCH --job-name=haps-merge
#SBATCH --partition=cpu2019
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/haps-merge-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load python/3.12.5
python_environment="${HAPS_PYTHON_ENVIRONMENT:-.venv-gpu}"
source "${python_environment}/bin/activate"

run_id="${HAPS_RUN_ID:?HAPS_RUN_ID is required}"
shard_count="${HAPS_SHARD_COUNT:?HAPS_SHARD_COUNT is required}"
dataset_root="${HAPS_DATASET_ROOT:-datasets/${run_id}}"
merged_run_id="${HAPS_MERGED_RUN_ID:-${run_id}-merged}"
merged_output="${HAPS_MERGED_OUTPUT:-${dataset_root}/merged}"

merge_args=(
  --shard-root "${dataset_root}/shards"
  --output "${merged_output}"
  --run-id "${merged_run_id}"
  --expected-shards "${shard_count}"
)
if [[ "${HAPS_ALLOW_FAILED_GATES:-0}" == "1" ]]; then
  merge_args+=(--allow-failed-gates)
fi

"${python_environment}/bin/python" scripts/merge_teacher_shards.py "${merge_args[@]}"
