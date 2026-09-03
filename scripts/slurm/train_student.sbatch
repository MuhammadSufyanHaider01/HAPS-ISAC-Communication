#!/usr/bin/env bash
#SBATCH --job-name=haps-gemma-distill
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/haps-gemma-distill-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load cuda/12.6.2 python/3.12.5
python_environment="${HAPS_PYTHON_ENVIRONMENT:-.venv-gpu}"
source "${python_environment}/bin/activate"

cache_root="${HAPS_CACHE_ROOT:-/home/muhammadsufyan.haide/.cache/haps-isac}"
dataset="${HAPS_DISTILL_DATASET:-datasets/qwen3.5-27b-production-5000-v2.2-001/distillation}"
config="${HAPS_DISTILL_CONFIG:-configs/distillation.yaml}"
output="${HAPS_DISTILL_OUTPUT:-checkpoints/gemma4-e4b-distillation-${SLURM_JOB_ID}}"
system_config="${HAPS_SYSTEM_CONFIG:-configs/system_v1.yaml}"

export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${output}"

exec "${python_environment}/bin/python" scripts/train_student.py \
  --dataset "${dataset}" \
  --config "${config}" \
  --system-config "${system_config}" \
  --output "${output}"

