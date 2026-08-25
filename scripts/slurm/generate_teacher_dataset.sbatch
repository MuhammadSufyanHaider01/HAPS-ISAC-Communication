#!/usr/bin/env bash
#SBATCH --job-name=haps-dataset
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00

set -euo pipefail

module load cuda/12.6.2
source .venv/bin/activate

teacher_model="${HAPS_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
teacher_revision="${HAPS_TEACHER_REVISION:-main}"
teacher_provider="${HAPS_TEACHER_PROVIDER:-qwen}"
teacher_port="${HAPS_TEACHER_PORT:-8000}"
tensor_parallel="${HAPS_TENSOR_PARALLEL_SIZE:-1}"
max_model_length="${HAPS_MAX_MODEL_LENGTH:-4096}"
gpu_utilization="${HAPS_GPU_MEMORY_UTILIZATION:-0.90}"
dataset_states="${HAPS_DATASET_STATES:-100}"
run_id="${HAPS_RUN_ID:-teacher-${SLURM_JOB_ID}}"
output_directory="${HAPS_OUTPUT_DIRECTORY:-datasets/${run_id}}"

mkdir -p "${output_directory}"

server_args=(
  serve "${teacher_model}"
  --revision "${teacher_revision}"
  --host 127.0.0.1
  --port "${teacher_port}"
  --tensor-parallel-size "${tensor_parallel}"
  --max-model-len "${max_model_length}"
  --gpu-memory-utilization "${gpu_utilization}"
)
if [[ "${teacher_provider}" == "qwen" ]]; then
  server_args+=(--enable-reasoning --reasoning-parser qwen3)
fi

vllm "${server_args[@]}" >"${output_directory}/vllm.log" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

health_url="http://127.0.0.1:${teacher_port}/health"
server_ready=0
for _ in {1..180}; do
  if python -c "import urllib.request; urllib.request.urlopen('${health_url}', timeout=2)"; then
    server_ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    break
  fi
  sleep 5
done
if [[ "${server_ready}" -ne 1 ]]; then
  echo "Teacher server did not become ready; inspect ${output_directory}/vllm.log" >&2
  exit 1
fi

python scripts/generate_demonstrations.py \
  --system-config configs/system_v1.yaml \
  --teacher-config configs/teacher.yaml \
  --provider "${teacher_provider}" \
  --model-id "${teacher_model}" \
  --base-url "http://127.0.0.1:${teacher_port}/v1" \
  --states "${dataset_states}" \
  --run-id "${run_id}" \
  --output "${output_directory}"
