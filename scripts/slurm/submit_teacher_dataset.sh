#!/usr/bin/env bash
set -euo pipefail

repository_directory="$(git rev-parse --show-toplevel)"
cd "${repository_directory}"

export HAPS_GIT_COMMIT="${HAPS_GIT_COMMIT:-$(git rev-parse HEAD)}"
if [[ -z "${HAPS_GIT_DIRTY:-}" ]]; then
  if [[ -n "$(git status --porcelain)" ]]; then
    export HAPS_GIT_DIRTY=1
  else
    export HAPS_GIT_DIRTY=0
  fi
fi

shard_count="${HAPS_SHARD_COUNT:-1}"
if ((shard_count <= 0)); then
  echo "HAPS_SHARD_COUNT must be positive" >&2
  exit 2
fi
if ((shard_count == 1)); then
  exec sbatch "$@" scripts/slurm/generate_teacher_dataset.sbatch
fi

maximum_parallel="${HAPS_MAX_PARALLEL_SHARDS:-${shard_count}}"
if ((maximum_parallel <= 0)); then
  echo "HAPS_MAX_PARALLEL_SHARDS must be positive" >&2
  exit 2
fi
array_stop=$((shard_count - 1))
exec sbatch \
  --array="0-${array_stop}%${maximum_parallel}" \
  "$@" \
  scripts/slurm/generate_teacher_dataset.sbatch
