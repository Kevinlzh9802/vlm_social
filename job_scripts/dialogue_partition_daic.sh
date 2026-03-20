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
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_INPUT_PATH="mintrec2/raw"
DEFAULT_CLIP_LENGTH="0.5"
DEFAULT_MODE="nested"

usage() {
    echo "Usage: sbatch $0 [input_path] [dialogue_range] [clip_length] [mode]" >&2
    echo "  input_path: 2-level path under ${DEFAULT_DATA_ROOT}, e.g. mintrec2/raw" >&2
    echo "  dialogue_range: 1-based hundred-range index, e.g. 1 → [0,100), 4 → [300,400). Empty for all." >&2
    echo "  clip_length: clip length in seconds for cumulative clips of the last utterance" >&2
    echo "  mode: nested (current layout) or context (prepend prior utterances to every clip)" >&2
    echo "  output is <data_root>/<dataset>/dialogue_partition/<subfolder>" >&2
}

INPUT_PATH="${1:-${DEFAULT_INPUT_PATH}}"
DIALOGUE_RANGE="${2:-}"
CLIP_LENGTH="${3:-${DEFAULT_CLIP_LENGTH}}"
MODE="${4:-${DEFAULT_MODE}}"

# Split input_path into dataset and subfolder: mintrec2/raw -> dataset=mintrec2, subfolder=raw
DATASET="${INPUT_PATH%%/*}"
SUBFOLDER="${INPUT_PATH#*/}"

INPUT_DIR="${DEFAULT_DATA_ROOT}/${INPUT_PATH}"

case "${MODE}" in
    nested)
        OUTPUT_DIR="${DEFAULT_DATA_ROOT}/${DATASET}/dialogue_partition/${SUBFOLDER}"
        ;;
    context)
        OUTPUT_DIR="${DEFAULT_DATA_ROOT}/${DATASET}/dialogue_partition_cont/${SUBFOLDER}"
        ;;
    *)
        usage
        echo "Invalid mode: ${MODE} (must be 'nested' or 'context')" >&2
        exit 1
        ;;
esac

if [[ -z "${INPUT_PATH}" || "${INPUT_PATH}" == "." || "${DATASET}" == "${SUBFOLDER}" ]]; then
    usage
    echo "Invalid input path: ${INPUT_PATH} (must be 2-level like dataset/subfolder)" >&2
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
echo "Data root: ${DEFAULT_DATA_ROOT}"
echo "Input path: ${INPUT_PATH}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Clip length: ${CLIP_LENGTH}"
echo "Dialogue range: ${DIALOGUE_RANGE:-all}"
echo "Mode: ${MODE}"

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    python /workspace/dataset/dialogue_partition.py \
    "${INPUT_DIR}" \
    "${OUTPUT_DIR}" \
    --clip-length "${CLIP_LENGTH}" \
    --mode "${MODE}" \
    --recursive \
    ${DIALOGUE_RANGE:+--dialogue-range "${DIALOGUE_RANGE}"}
