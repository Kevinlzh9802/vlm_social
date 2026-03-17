#!/bin/bash
#SBATCH --job-name="dataset_info"
#SBATCH --time=3:59:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${PROJECT_ROOT}/apptainer/vlm_social.sif"
DEFAULT_INPUT_ROOT="/scratch/zli33/data/gestalt_bench/mintrec2"
DEFAULT_INPUT_FOLDER="raw"
DEFAULT_OUTPUT_ROOT="/scratch/zli33/data/gestalt_bench/results/gesbench"

usage() {
    echo "Usage: sbatch $0 [input_folder] [output_root]" >&2
    echo "  input_folder: folder name under ${DEFAULT_INPUT_ROOT}" >&2
    echo "  output_root: base results folder; final output becomes output_root/<input_folder>" >&2
}

INPUT_FOLDER_NAME="${1:-${DEFAULT_INPUT_FOLDER}}"
OUTPUT_ROOT="${2:-${DEFAULT_OUTPUT_ROOT}}"
INPUT_DIR="${DEFAULT_INPUT_ROOT}/${INPUT_FOLDER_NAME}"
OUTPUT_DIR="${OUTPUT_ROOT}/${INPUT_FOLDER_NAME}"

if [[ -z "${INPUT_FOLDER_NAME}" || "${INPUT_FOLDER_NAME}" == "." ]]; then
    usage
    echo "Invalid input folder name: ${INPUT_FOLDER_NAME}" >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build vlm_social.sif vlm_social.def" >&2
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
echo "Input root: ${DEFAULT_INPUT_ROOT}"
echo "Input dir: ${INPUT_DIR}"
echo "Input folder name: ${INPUT_FOLDER_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /scratch:/scratch \
    "${SIF_PATH}" \
    python /workspace/dataset/dataset_info.py \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    --recursive
