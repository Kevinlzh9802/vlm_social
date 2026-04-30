#!/bin/bash
#SBATCH --job-name="dataset_info"
#SBATCH --time=3:59:00
#SBATCH --partition=insy,general # Request partition. Default is 'general' 
#SBATCH --qos=short         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --output=logs/dataset_info_daic_%j.out
#SBATCH --error=logs/dataset_info_daic_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench RESULTS_ROOT=/path/to/results/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/gesbench.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_INPUT_PATH="mintrec2/raw"
DEFAULT_OUTPUT_ROOT="${RESULTS_ROOT:-/path/to/results/gestalt_bench}"

usage() {
    echo "Usage: sbatch $0 [input_path] [output_root]" >&2
    echo "  input_path: 2-level path under ${DEFAULT_DATA_ROOT}, e.g. mintrec2/raw" >&2
    echo "  output_root: base results folder; final output becomes output_root/<input_path>" >&2
}

INPUT_PATH="${1:-${DEFAULT_INPUT_PATH}}"
OUTPUT_ROOT="${2:-${DEFAULT_OUTPUT_ROOT}}"
INPUT_DIR="${DEFAULT_DATA_ROOT}/${INPUT_PATH}"
OUTPUT_DIR="${OUTPUT_ROOT}/${INPUT_PATH}"

if [[ -z "${INPUT_PATH}" || "${INPUT_PATH}" == "." ]]; then
    usage
    echo "Invalid input path: ${INPUT_PATH}" >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build gesbench.sif gesbench.def" >&2
    exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    usage
    echo "Input directory does not exist: ${INPUT_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Project root: ${PROJECT_ROOT}"
echo "Apptainer image: ${SIF_PATH}"
echo "Data root: ${DEFAULT_DATA_ROOT}"
echo "Input path: ${INPUT_PATH}"
echo "Input dir: ${INPUT_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    --bind "${OUTPUT_ROOT}:${OUTPUT_ROOT}" \
    "${SIF_PATH}" \
    python /workspace/dataset/dataset_info.py \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    --recursive
