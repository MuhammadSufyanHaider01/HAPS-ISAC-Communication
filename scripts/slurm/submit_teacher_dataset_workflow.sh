#!/usr/bin/env bash
set -euo pipefail

repository_directory="$(git rev-parse --show-toplevel)"
cd "${repository_directory}"

: "${HAPS_RUN_ID:?HAPS_RUN_ID is required}"
: "${HAPS_SHARD_COUNT:?HAPS_SHARD_COUNT is required}"
if ((HAPS_SHARD_COUNT <= 0)); then
  echo "HAPS_SHARD_COUNT must be positive" >&2
  exit 2
fi

export HAPS_GIT_COMMIT="${HAPS_GIT_COMMIT:-$(git rev-parse HEAD)}"
if [[ -z "${HAPS_GIT_DIRTY:-}" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    export HAPS_GIT_DIRTY=1
  else
    export HAPS_GIT_DIRTY=0
  fi
fi
if [[ "${HAPS_GIT_DIRTY}" != "0" ]]; then
  echo "Refusing production workflow submission from a dirty Git tree" >&2
  exit 2
fi

maximum_parallel="${HAPS_MAX_PARALLEL_SHARDS:-${HAPS_SHARD_COUNT}}"
if ((maximum_parallel <= 0)); then
  echo "HAPS_MAX_PARALLEL_SHARDS must be positive" >&2
  exit 2
fi
array_stop=$((HAPS_SHARD_COUNT - 1))
generation_job_id="$(
  sbatch --parsable \
    --array="0-${array_stop}%${maximum_parallel}" \
    "$@" \
    scripts/slurm/generate_teacher_dataset.sbatch
)"
generation_job_id="${generation_job_id%%;*}"
merge_job_id="$(
  sbatch --parsable \
    --dependency="afterok:${generation_job_id}" \
    scripts/slurm/merge_teacher_dataset.sbatch
)"
merge_job_id="${merge_job_id%%;*}"

printf 'generation_job_id=%s\n' "${generation_job_id}"
printf 'merge_job_id=%s\n' "${merge_job_id}"
printf 'dataset_root=%s\n' "${HAPS_DATASET_ROOT:-datasets/${HAPS_RUN_ID}}"
