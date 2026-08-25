#!/usr/bin/env bash
#SBATCH --job-name=haps-gpu-preflight
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=results/haps-gpu-preflight-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

module load cuda/12.6.2 python/3.12.5
python_environment="${HAPS_PYTHON_ENVIRONMENT:-.venv}"
source "${python_environment}/bin/activate"
pwd
command -v python
"${python_environment}/bin/python" -c 'import sys; print(sys.executable)'

nvidia-smi --query-gpu=timestamp,index,name,driver_version,memory.total,compute_cap --format=csv

"${python_environment}/bin/python" -c '
import importlib.metadata as metadata
import json
import torch
packages = {}
for package in ("vllm", "transformers"):
    try:
        packages[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        packages[package] = None
properties = torch.cuda.get_device_properties(0)
print(json.dumps({"cuda_available": torch.cuda.is_available(), "device": properties.name, "memory_bytes": properties.total_memory, "compute_capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "packages": packages}, sort_keys=True))
'
