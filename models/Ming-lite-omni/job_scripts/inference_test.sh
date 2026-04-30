#!/bin/bash
#SBATCH --job-name="ming-lite-omni_inference_test"
#SBATCH --partition=gpu-a100
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/ming-lite-omni/inference_test_%j.out
#SBATCH --error=logs/ming-lite-omni/inference_test_%j.err
# Submit from the model project root; ensure logs/ming-lite-omni exists before sbatch.
# User paths to set: export MING_PROJECT_ROOT=/path/to/Ming-lite-omni DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers
# Optional log path: export LOG_DIR=logs/ming-lite-omni


# Simple Ming-Lite-Omni inference test via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_test.sh
#   sbatch job_scripts/inference_test.sh test_infer_gen_image.py
#   sbatch job_scripts/inference_test.sh test_audio_tasks.py

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MING_PROJECT_ROOT="${MING_PROJECT_ROOT:-${PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
APPTAINER_ROOT="${APPTAINER_ROOT:-/path/to/apptainers}"
LOG_DIR="${LOG_DIR:-logs/ming-lite-omni}"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="${MING_PROJECT_ROOT}"
sif_file=${APPTAINER_ROOT}/ming-lite-omni.sif

# Default test script; override via first positional argument
test_script="${1:-test_infer.py}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build it first: bash apptainer/build.sh" >&2
    exit 1
fi

if [ ! -f "$project_dir/$test_script" ]; then
    echo "[ERROR] Test script not found: $project_dir/$test_script" >&2
    exit 1
fi

# Ensure slurm output directory exists
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file   = $sif_file"
echo "[INFO] project_dir= $project_dir"
echo "[INFO] test_script= $test_script"
echo ""

echo "[HOST] nvidia-smi:"
nvidia-smi || true

# Triton's JIT compiler needs the unversioned "libcuda.so" to link GPU
# kernels.  Apptainer --nv injects the host driver into /.singularity.d/libs
# (which may include libcuda.so), but Triton's libcuda_dirs() searches
# ldconfig and hardcoded paths like /usr/lib64 — NOT LD_LIBRARY_PATH.
#
# Fix: bind-mount a directory containing a libcuda.so symlink (pointing to
# the --nv-injected library) over /usr/lib64 would be destructive, so
# instead we create the symlink inside the container via the bash wrapper.
apptainer exec --nv --writable-tmpfs \
  --bind "$project_dir":/workspace \
  --bind "$DATA_ROOT":"$DATA_ROOT" \
  --pwd /workspace \
  "$sif_file" \
  bash -lc '
    set -euo pipefail

    real=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
    test -n "$real"

    mkdir -p /tmp/triton-libcuda
    ln -sf "$real" /tmp/triton-libcuda/libcuda.so
    ln -sf "$real" /tmp/triton-libcuda/libcuda.so.1

    export TRITON_LIBCUDA_PATH=/tmp/triton-libcuda
    export LD_LIBRARY_PATH=/tmp/triton-libcuda:${LD_LIBRARY_PATH:-}

    python - <<PY
from pathlib import Path
p = Path("/opt/conda/lib/python3.10/site-packages/triton/common/build.py")
txt = p.read_text()
needle = "@functools.lru_cache()\ndef libcuda_dirs():\n"
patch = """@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv(\"TRITON_LIBCUDA_PATH\")
    if env_libcuda_path:
        return [env_libcuda_path]
"""
if "TRITON_LIBCUDA_PATH" not in txt:
    txt = txt.replace(needle, patch)
    p.write_text(txt)
    print("Patched Triton build.py")
else:
    print("Triton build.py already patched")
PY

    python "'"$test_script"'"
  '
