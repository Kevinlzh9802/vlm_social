#!/bin/bash
#SBATCH --job-name="qwen2.5-omni_inference_test"
#SBATCH --partition=gpu-a100
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/qwen2.5-omni/inference_test_%j.out
#SBATCH --error=logs/qwen2.5-omni/inference_test_%j.err
# Submit from the model project root; ensure logs/qwen2.5-omni exists before sbatch.
# User paths to set: export QWEN_PROJECT_ROOT=/path/to/Qwen2.5-Omni DATA_ROOT=/path/to/data/gestalt_bench MODEL_ROOT=/path/to/models APPTAINER_ROOT=/path/to/apptainers
# Optional cache/log paths: export HF_CACHE=/path/to/huggingface-cache LOG_DIR=logs/qwen2.5-omni


# Simple Qwen2.5-Omni inference test via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_test.sh
#   sbatch job_scripts/inference_test.sh --model ${MODEL_ROOT:-/path/to/models}/Qwen2.5-Omni-7B

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
QWEN_PROJECT_ROOT="${QWEN_PROJECT_ROOT:-${PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
MODEL_ROOT="${MODEL_ROOT:-/path/to/models}"
APPTAINER_ROOT="${APPTAINER_ROOT:-/path/to/apptainers}"
HF_CACHE="${HF_CACHE:-${MODEL_ROOT}/.cache/huggingface}"
LOG_DIR="${LOG_DIR:-logs/qwen2.5-omni}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Under Slurm, BASH_SOURCE may point to a spool location; prefer submission dir.
project_dir=${QWEN_PROJECT_ROOT}
sif_file=${APPTAINER_ROOT}/qwen2.5-omni-inference.sif
hf_cache_host=${HF_CACHE}
data_root_host=${DATA_ROOT}
model_root_host=${MODEL_ROOT}

# Always run workspace infer_test.py; CLI args are forwarded to the script.
test_script="infer_test.py"
script_args=("$@")

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build/copy the SIF first." >&2
    exit 1
fi

if [ ! -f "$project_dir/$test_script" ]; then
    echo "[ERROR] Test script not found: $project_dir/$test_script" >&2
    exit 1
fi

# Ensure slurm output directory exists
mkdir -p "$LOG_DIR"
mkdir -p "$hf_cache_host"

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file   = $sif_file"
echo "[INFO] project_dir= $project_dir"
echo "[INFO] test_script= $test_script"
echo "[INFO] hf_cache   = $hf_cache_host"
echo "[INFO] data_root  = $data_root_host"
echo "[INFO] model_root = $model_root_host"
if [ "${#script_args[@]}" -gt 0 ]; then
    echo "[INFO] script_args= ${script_args[*]}"
fi
echo ""

apptainer exec --nv \
    --bind "$project_dir":/workspace \
    --bind "$hf_cache_host":/opt/huggingface \
    --bind "$data_root_host":"$data_root_host" \
    --bind "$model_root_host":"$model_root_host" \
    --env HF_HOME=/opt/huggingface \
    --env TRANSFORMERS_CACHE=/opt/huggingface \
    --pwd /workspace \
    "$sif_file" \
    python "$test_script" "${script_args[@]}"

echo ""
echo "[INFO] Inference test completed successfully."
