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

exec sbatch "$@" scripts/slurm/generate_teacher_dataset.sbatch
