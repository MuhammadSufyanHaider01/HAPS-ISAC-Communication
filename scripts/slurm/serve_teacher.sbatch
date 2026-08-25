#!/usr/bin/env bash
#SBATCH --job-name=haps-teacher
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results/haps-teacher-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load cuda/12.6.2 python/3.12.5
python_environment="${HAPS_PYTHON_ENVIRONMENT:-.venv-gpu}"
source "${python_environment}/bin/activate"

teacher_model="${HAPS_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
teacher_revision="${HAPS_TEACHER_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}"
teacher_host="${HAPS_TEACHER_HOST:-127.0.0.1}"
teacher_port="${HAPS_TEACHER_PORT:-8000}"
cache_root="${HAPS_CACHE_ROOT:-/home/muhammadsufyan.haide/.cache/haps-isac}"

export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export HF_HUB_OFFLINE="${HAPS_HF_HUB_OFFLINE:-1}"
mkdir -p "${HF_HOME}"

exec "${python_environment}/bin/python" scripts/serve_transformers_teacher.py \
  --model "${teacher_model}" \
  --revision "${teacher_revision}" \
  --host "${teacher_host}" \
  --port "${teacher_port}"
