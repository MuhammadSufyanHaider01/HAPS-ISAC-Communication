#!/usr/bin/env bash
#SBATCH --job-name=haps-dataset
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=results/haps-dataset-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load cuda/12.6.2 python/3.12.5
python_environment="${HAPS_PYTHON_ENVIRONMENT:-.venv-gpu}"
source "${python_environment}/bin/activate"

teacher_model="${HAPS_TEACHER_MODEL:-Qwen/Qwen3.5-27B}"
teacher_revision="${HAPS_TEACHER_REVISION:-fc05daec18b0a78c049392ed2e771dde82bdf654}"
teacher_provider="${HAPS_TEACHER_PROVIDER:-qwen}"
teacher_backend="${HAPS_TEACHER_BACKEND:-transformers}"
teacher_port="${HAPS_TEACHER_PORT:-8000}"
tensor_parallel="${HAPS_TENSOR_PARALLEL_SIZE:-1}"
max_model_length="${HAPS_MAX_MODEL_LENGTH:-4096}"
gpu_utilization="${HAPS_GPU_MEMORY_UTILIZATION:-0.90}"
dataset_states="${HAPS_DATASET_STATES:-100}"
run_id="${HAPS_RUN_ID:-teacher-${SLURM_JOB_ID}}"
output_directory="${HAPS_OUTPUT_DIRECTORY:-datasets/${run_id}}"
cache_root="${HAPS_CACHE_ROOT:-/home/muhammadsufyan.haide/.cache/haps-isac}"
flush_every="${HAPS_LOG_FLUSH_EVERY:-1}"
startup_timeout_s="${HAPS_SERVER_STARTUP_TIMEOUT_S:-3600}"

export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
export HF_HUB_OFFLINE="${HAPS_HF_HUB_OFFLINE:-1}"
export TORCH_HOME="${TORCH_HOME:-${cache_root}/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${cache_root}/vllm}"
export HAPS_TEACHER_BACKEND="${teacher_backend}"
export HAPS_PYTHON_ENVIRONMENT="${python_environment}"
mkdir -p "${output_directory}" "${HF_HOME}" "${TORCH_HOME}" "${VLLM_CACHE_ROOT}"

gpu_metrics_file="${output_directory}/gpu_metrics.csv"
printf '%s\n' 'timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w,temperature_c' >"${gpu_metrics_file}"

if [[ "${teacher_backend}" == "transformers" ]]; then
  server_log="${output_directory}/teacher_server.stderr.log"
  "${python_environment}/bin/python" scripts/serve_transformers_teacher.py \
    --model "${teacher_model}" \
    --revision "${teacher_revision}" \
    --host 127.0.0.1 \
    --port "${teacher_port}" \
    >"${output_directory}/teacher_server_metrics.jsonl" \
    2>"${server_log}" &
elif [[ "${teacher_backend}" == "vllm" ]]; then
  server_log="${output_directory}/vllm.log"
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
  "${python_environment}/bin/vllm" "${server_args[@]}" >"${server_log}" 2>&1 &
else
  printf 'Unsupported HAPS_TEACHER_BACKEND: %s\n' "${teacher_backend}" >&2
  exit 2
fi
server_pid=$!
(
  while kill -0 "${server_pid}" 2>/dev/null; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >>"${gpu_metrics_file}"
    sleep 5
  done
) &
metrics_pid=$!
cleanup() {
  kill "${metrics_pid}" 2>/dev/null || true
  wait "${metrics_pid}" 2>/dev/null || true
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

health_url="http://127.0.0.1:${teacher_port}/health"
server_ready=0
health_attempts=$((startup_timeout_s / 5))
for ((attempt = 0; attempt < health_attempts; attempt++)); do
  if "${python_environment}/bin/python" -c "import urllib.request; urllib.request.urlopen('${health_url}', timeout=2)"; then
    server_ready=1
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    break
  fi
  sleep 5
done
if [[ "${server_ready}" -ne 1 ]]; then
  echo "Teacher server did not become ready; inspect ${server_log}" >&2
  exit 1
fi

"${python_environment}/bin/python" scripts/generate_demonstrations.py \
  --system-config configs/system_v1.yaml \
  --teacher-config configs/teacher.yaml \
  --provider "${teacher_provider}" \
  --model-id "${teacher_model}" \
  --model-revision "${teacher_revision}" \
  --base-url "http://127.0.0.1:${teacher_port}/v1" \
  --states "${dataset_states}" \
  --run-id "${run_id}" \
  --output "${output_directory}" \
  --flush-every "${flush_every}"

"${python_environment}/bin/python" scripts/audit_demonstrations.py "${output_directory}" \
  | tee "${output_directory}/audit.json"
