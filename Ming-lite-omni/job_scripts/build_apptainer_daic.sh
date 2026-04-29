#!/bin/bash
#SBATCH --job-name="ming-lite-omni_build_daic"
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40
#SBATCH --account ewi-insy-prb
#SBATCH --partition insy,general
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/ming-lite-omni/build_apptainer_daic_%j.out
#SBATCH --error=logs/ming-lite-omni/build_apptainer_daic_%j.err
# Submit from the model project root; ensure logs/ming-lite-omni exists before sbatch.
# User paths to set: export MING_PROJECT_ROOT=/path/to/Ming-lite-omni APPTAINER_ROOT=/path/to/apptainers LOG_DIR=logs/ming-lite-omni
# Optional build paths: export SIF_PATH=/path/to/ming-lite-omni.sif APPTAINER_TMPDIR=/path/to/tmp


set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
MING_PROJECT_ROOT="${MING_PROJECT_ROOT:-${PROJECT_DIR:-${PROJECT_ROOT}}}"
APPTAINER_ROOT="${APPTAINER_ROOT:-/path/to/apptainers}"
LOG_DIR="${LOG_DIR:-logs/ming-lite-omni}"

# Build the Apptainer image on DAIC.
#
# Submit from the project folder:
#   sbatch job_scripts/build_apptainer_daic.sh
#
# Optional overrides:
#   sbatch --export=ALL,MING_PROJECT_ROOT=/path/to/Ming-lite-omni,SIF_PATH=/path/to/ming-lite-omni.sif job_scripts/build_apptainer_daic.sh

project_dir="${MING_PROJECT_ROOT}"
log_dir="${LOG_DIR}"

mkdir -p "$log_dir"

if [ ! -d "$project_dir" ]; then
    echo "[ERROR] Project directory not found: $project_dir" >&2
    exit 1
fi

cd "$project_dir"

build_script="$project_dir/apptainer/build.sh"
def_file="$project_dir/apptainer/Ming-lite-omni.def"
sif_file="${SIF_PATH:-${APPTAINER_ROOT}/ming-lite-omni.sif}"

if [ ! -f "$build_script" ]; then
    echo "[ERROR] Build script not found: $build_script" >&2
    exit 1
fi

if [ ! -f "$def_file" ]; then
    echo "[ERROR] Definition file not found: $def_file" >&2
    exit 1
fi

echo "[INFO] host         = $(hostname)"
echo "[INFO] project_dir  = $project_dir"
echo "[INFO] build_script = $build_script"
echo "[INFO] def_file     = $def_file"
echo "[INFO] sif_file     = $sif_file"
echo "[INFO] job_id       = ${SLURM_JOB_ID:-<none>}"
echo

module purge || true

if ! command -v apptainer >/dev/null 2>&1; then
    echo "[ERROR] apptainer is not installed or not in PATH on this node" >&2
    exit 1
fi

export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/$USER/apptainer-build-${SLURM_JOB_ID:-manual}}"
export TMPDIR="${TMPDIR:-$APPTAINER_TMPDIR}"
mkdir -p "$APPTAINER_TMPDIR"

echo "[INFO] APPTAINER_TMPDIR = $APPTAINER_TMPDIR"
echo

mkdir -p "$(dirname "$sif_file")"

if [ "$sif_file" = "$project_dir/apptainer/ming-lite-omni.sif" ]; then
    bash "$build_script"
else
    BUILD_FLAGS=()
    if [ "$(id -u)" -ne 0 ]; then
        BUILD_FLAGS+=(--fakeroot)
    fi
    apptainer build "${BUILD_FLAGS[@]}" "$sif_file" "$def_file"
fi

echo
echo "[INFO] Build completed: $sif_file"
