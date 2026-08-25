#!/usr/bin/env bash
#SBATCH --job-name=haps-teacher
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

module load cuda/12.6.2
source .venv/bin/activate

teacher_model="${HAPS_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
teacher_revision="${HAPS_TEACHER_REVISION:-main}"
teacher_provider="${HAPS_TEACHER_PROVIDER:-qwen}"
teacher_host="${HAPS_TEACHER_HOST:-0.0.0.0}"
teacher_port="${HAPS_TEACHER_PORT:-8000}"
tensor_parallel="${HAPS_TENSOR_PARALLEL_SIZE:-1}"
max_model_length="${HAPS_MAX_MODEL_LENGTH:-4096}"
gpu_utilization="${HAPS_GPU_MEMORY_UTILIZATION:-0.90}"

server_args=(
  serve "${teacher_model}"
  --revision "${teacher_revision}"
  --host "${teacher_host}"
  --port "${teacher_port}"
  --tensor-parallel-size "${tensor_parallel}"
  --max-model-len "${max_model_length}"
  --gpu-memory-utilization "${gpu_utilization}"
)

if [[ "${teacher_provider}" == "qwen" ]]; then
  server_args+=(--enable-reasoning --reasoning-parser qwen3)
fi

exec vllm "${server_args[@]}"
