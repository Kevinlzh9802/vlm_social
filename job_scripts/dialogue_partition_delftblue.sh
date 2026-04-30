#!/bin/bash
#SBATCH --job-name="seamless_utt"
#SBATCH --time=16:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/dialogue_partition_delftblue_%j.out
#SBATCH --error=logs/dialogue_partition_delftblue_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/vlm_social DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/vlm_social.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_INPUT_PATH="mintrec2/raw"
DEFAULT_CLIP_LENGTH="0.5"
DEFAULT_MODE="context"
CUT=""
UTT=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [input_path] [dialogue_range] [clip_length] [mode] [cut] [utt]" >&2
    echo "  sbatch $0 [--input-path PATH] [--dialogue-range N] [--clip-length SEC] [--mode nested|context] [--cut SEC] [--utt 1,2,3] [--overwrite-1utt]" >&2
    echo "  input_path: 2-level path under ${DEFAULT_DATA_ROOT}, e.g. mintrec2/raw" >&2
    echo "  dialogue_range: 1-based hundred-range index, e.g. 1 → [0,100), 4 → [300,400). Empty for all." >&2
    echo "  clip_length: clip length in seconds for cumulative clips of the last utterance" >&2
    echo "  mode: nested (current layout) or context (prepend prior utterances to every clip)" >&2
    echo "  cut: optional max length in seconds for the final utterance before clipping" >&2
    echo "  utt: comma-separated group sizes to write, e.g. 1,2 or 3 (default: all)" >&2
    echo "  --overwrite-1utt: overwrite existing 1-utt outputs while reusing existing 2/3-utt outputs" >&2
    echo "  output is <data_root>/<dataset>/<mode>/<subfolder>" >&2
}

INPUT_PATH="${DEFAULT_INPUT_PATH}"
DIALOGUE_RANGE=""
CLIP_LENGTH="${DEFAULT_CLIP_LENGTH}"
MODE="${DEFAULT_MODE}"
OVERWRITE_1UTT=0
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-path)
            INPUT_PATH="${2:?Missing value for --input-path}"
            shift 2
            ;;
        --dialogue-range)
            DIALOGUE_RANGE="${2:?Missing value for --dialogue-range}"
            shift 2
            ;;
        --clip-length)
            CLIP_LENGTH="${2:?Missing value for --clip-length}"
            shift 2
            ;;
        --mode)
            MODE="${2:?Missing value for --mode}"
            shift 2
            ;;
        --cut)
            CUT="${2:?Missing value for --cut}"
            shift 2
            ;;
        --utt)
            UTT="${2:?Missing value for --utt}"
            shift 2
            ;;
        --overwrite-1utt)
            OVERWRITE_1UTT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            usage
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    INPUT_PATH="${POSITIONAL_ARGS[0]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    DIALOGUE_RANGE="${POSITIONAL_ARGS[1]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
    CLIP_LENGTH="${POSITIONAL_ARGS[2]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 3 ]]; then
    MODE="${POSITIONAL_ARGS[3]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 4 ]]; then
    CUT="${POSITIONAL_ARGS[4]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 5 ]]; then
    UTT="${POSITIONAL_ARGS[5]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 6 ]]; then
    usage
    echo "Too many positional arguments." >&2
    exit 1
fi

# Split input_path into dataset and subfolder: mintrec2/raw -> dataset=mintrec2, subfolder=raw
DATASET="${INPUT_PATH%%/*}"
SUBFOLDER="${INPUT_PATH#*/}"
INPUT_DIR="${DEFAULT_DATA_ROOT}/${INPUT_PATH}"

if [[ "${MODE}" != "nested" && "${MODE}" != "context" ]]; then
    usage
    echo "Invalid mode: ${MODE} (must be 'nested' or 'context')" >&2
    exit 1
fi

OUTPUT_DIR="${DEFAULT_DATA_ROOT}/${DATASET}/${MODE}/${SUBFOLDER}"

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
echo "Cut final utterance to first N seconds: ${CUT:-none}"
echo "Selected group sizes: ${UTT:-all}"
echo "Overwrite 1-utt groups: ${OVERWRITE_1UTT}"

PYTHON_ARGS=(
    python /workspace/dataset/dialogue_partition.py
    "${INPUT_DIR}"
    "${OUTPUT_DIR}"
    --clip-length "${CLIP_LENGTH}"
    --mode "${MODE}"
    --recursive
)

if [[ -n "${DIALOGUE_RANGE}" ]]; then
    PYTHON_ARGS+=(--dialogue-range "${DIALOGUE_RANGE}")
fi

if [[ -n "${CUT}" ]]; then
    PYTHON_ARGS+=(--cut "${CUT}")
fi

if [[ -n "${UTT}" ]]; then
    PYTHON_ARGS+=(--utt "${UTT}")
fi

if [[ "${OVERWRITE_1UTT}" == "1" ]]; then
    PYTHON_ARGS+=(--overwrite-1utt)
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
