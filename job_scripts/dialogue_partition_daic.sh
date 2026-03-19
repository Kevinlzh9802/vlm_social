#!/bin/bash
#SBATCH --job-name="dialogue_partition"
#SBATCH --time=34:00:00
#SBATCH --partition=insy,general # Request partition. Default is 'general'
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/vlm_social.sif"
DEFAULT_INPUT_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/mintrec2"
DEFAULT_INPUT_FOLDER="raw"
DEFAULT_OUTPUT_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/mintrec2/dialogue_partition"
DEFAULT_CLIP_LENGTH="0.5"

usage() {
    echo "Usage: sbatch $0 [input_folder] [dialogue_range] [output_root] [clip_length]" >&2
    echo "  input_folder: folder name under ${DEFAULT_INPUT_ROOT}" >&2
    echo "  dialogue_range: 1-based hundred-range index, e.g. 1 → [0,100), 4 → [300,400). Empty for all." >&2
    echo "  output_root: base results folder; final output becomes output_root/<input_folder>" >&2
    echo "  clip_length: clip length in seconds for cumulative clips of the last utterance" >&2
}

INPUT_FOLDER_NAME="${1:-${DEFAULT_INPUT_FOLDER}}"
DIALOGUE_RANGE="${2:-}"
OUTPUT_ROOT="${3:-${DEFAULT_OUTPUT_ROOT}}"
CLIP_LENGTH="${4:-${DEFAULT_CLIP_LENGTH}}"
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
echo "Clip length: ${CLIP_LENGTH}"
echo "Dialogue range: ${DIALOGUE_RANGE:-all}"

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    python /workspace/dataset/dialogue_partition.py \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    --clip-length "${CLIP_LENGTH}" \
    --recursive \
    ${DIALOGUE_RANGE:+--dialogue-range "${DIALOGUE_RANGE}"}
