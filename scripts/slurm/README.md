# Slurm teacher jobs

These templates target the GPU partitions visible on the ARC cluster. They do not launch automatically.

Before submission, create a GPU environment containing the project data extras and a CUDA-compatible vLLM installation. Keep model caches outside the repository.

Serve a model for interactive testing:

~~~bash
HAPS_TEACHER_MODEL=Qwen/Qwen3.5-27B \
  sbatch scripts/slurm/serve_teacher.sbatch
~~~

Generate a small fully verified pilot in one allocation:

~~~bash
HAPS_DATASET_STATES=100 \
HAPS_RUN_ID=qwen-pilot-001 \
  sbatch scripts/slurm/generate_teacher_dataset.sbatch
~~~

Scale only after the pilot passes the dataset audit. Override HAPS_TEACHER_MODEL, HAPS_TEACHER_PROVIDER, HAPS_TENSOR_PARALLEL_SIZE, HAPS_MAX_MODEL_LENGTH, and HAPS_OUTPUT_DIRECTORY through exported environment variables. Never place Hugging Face or API tokens in an sbatch file.
