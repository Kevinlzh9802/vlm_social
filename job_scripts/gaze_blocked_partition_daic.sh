#!/bin/bash
#SBATCH --job-name="gaze_block"
#SBATCH --time=08:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=logs/gaze_blocked_partition_daic_%j.out
#SBATCH --error=logs/gaze_blocked_partition_daic_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/eyetrack.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_PUPIL_PARENT="${DEFAULT_DATA_ROOT}/human_eval/pupil"
DEFAULT_ANNOTATION_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/results"
DEFAULT_VIDEO_JSON="${DEFAULT_DATA_ROOT}/human_eval/task2/task2.json"
DEFAULT_SOURCE_VIDEOS="${DEFAULT_DATA_ROOT}/human_eval/videos"
DEFAULT_FULL_CORRUPTION_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/data"
DEFAULT_MEDIA_URL_PREFIX="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_EFFECT="block"
DEFAULT_CLIP_LENGTH="0.5"
DEFAULT_FOCUS_REGION_RATIO="0.3"
DEFAULT_FOCUS_REGION_SHAPE="circle"
DEFAULT_MAX_GAZE_GAP="0.5"
DEFAULT_GAZE_MAPPING="measured-player"
DEFAULT_RESPONSE_SELECTION="latest-submitted"
DEFAULT_FULL_CORRUPTION_LOCALIZATION_SOURCE="annotation-media"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--annotation-dir PATH] [--video-json PATH] [--source-videos PATH] [--output-dir PATH] [--media-url-prefix URL] [--effect blur|block] [--clip-length SEC] [--focus-region-ratio RATIO] [--focus-region-shape circle|square] [--max-gaze-gap SEC] [--gaze-mapping legacy-extraction|measured-player] [--response-selection latest-submitted|first-response] [--full-corruption-localization-source annotation-media|video-json] [--utt 1,2,3] [--comparison] [--no-overwrite]" >&2
    echo "  pupil-parent: parent folder with T{x}_{y}_annotator{n} Pupil recording folders (default: ${DEFAULT_PUPIL_PARENT})" >&2
    echo "  annotation-dir: folder with T{x}_{y}.json annotation files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  video-json: fallback ordered video list JSON (default: ${DEFAULT_VIDEO_JSON})" >&2
    echo "  source-videos: local path replacing the media URL prefix (default: ${DEFAULT_SOURCE_VIDEOS})" >&2
    echo "  output-dir: whole-clip gaze-corrupted output parent (default: ${DEFAULT_FULL_CORRUPTION_OUTPUT_DIR})" >&2
    echo "  media-url-prefix: media URL prefix to replace; annotation1 and annotation2 are both accepted (default: ${DEFAULT_MEDIA_URL_PREFIX})" >&2
    echo "  effect: manipulation effect, default ${DEFAULT_EFFECT}" >&2
    echo "  gaze-mapping: screen-to-video transform, default ${DEFAULT_GAZE_MAPPING}" >&2
    echo "  response-selection: annotation response selector, default ${DEFAULT_RESPONSE_SELECTION}" >&2
    echo "  comparison: write under data_comparison and use shifted non-gaze mask centers." >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
VIDEO_JSON="${DEFAULT_VIDEO_JSON}"
SOURCE_VIDEOS="${DEFAULT_SOURCE_VIDEOS}"
FULL_CORRUPTION_OUTPUT_DIR="${DEFAULT_FULL_CORRUPTION_OUTPUT_DIR}"
MEDIA_URL_PREFIX="${DEFAULT_MEDIA_URL_PREFIX}"
EFFECT="${DEFAULT_EFFECT}"
CLIP_LENGTH="${DEFAULT_CLIP_LENGTH}"
FOCUS_REGION_RATIO="${DEFAULT_FOCUS_REGION_RATIO}"
FOCUS_REGION_SHAPE="${DEFAULT_FOCUS_REGION_SHAPE}"
MAX_GAZE_GAP="${DEFAULT_MAX_GAZE_GAP}"
GAZE_MAPPING="${DEFAULT_GAZE_MAPPING}"
RESPONSE_SELECTION="${DEFAULT_RESPONSE_SELECTION}"
FULL_CORRUPTION_LOCALIZATION_SOURCE="${DEFAULT_FULL_CORRUPTION_LOCALIZATION_SOURCE}"
UTT=""
COMPARISON=0
OVERWRITE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pupil-parent)
            PUPIL_PARENT="${2:?Missing value for --pupil-parent}"
            shift 2
            ;;
        --annotation-dir)
            ANNOTATION_DIR="${2:?Missing value for --annotation-dir}"
            shift 2
            ;;
        --video-json)
            VIDEO_JSON="${2:?Missing value for --video-json}"
            shift 2
            ;;
        --source-videos|--local-path-prefix)
            SOURCE_VIDEOS="${2:?Missing value for $1}"
            shift 2
            ;;
        --output-dir|--full-corruption-output-dir)
            FULL_CORRUPTION_OUTPUT_DIR="${2:?Missing value for $1}"
            shift 2
            ;;
        --media-url-prefix)
            MEDIA_URL_PREFIX="${2:?Missing value for --media-url-prefix}"
            shift 2
            ;;
        --effect)
            EFFECT="${2:?Missing value for --effect}"
            shift 2
            ;;
        --clip-length)
            CLIP_LENGTH="${2:?Missing value for --clip-length}"
            shift 2
            ;;
        --focus-region-ratio|--focus-box-ratio)
            FOCUS_REGION_RATIO="${2:?Missing value for $1}"
            shift 2
            ;;
        --focus-region-shape)
            FOCUS_REGION_SHAPE="${2:?Missing value for --focus-region-shape}"
            shift 2
            ;;
        --max-gaze-gap)
            MAX_GAZE_GAP="${2:?Missing value for --max-gaze-gap}"
            shift 2
            ;;
        --gaze-mapping)
            GAZE_MAPPING="${2:?Missing value for --gaze-mapping}"
            shift 2
            ;;
        --response-selection)
            RESPONSE_SELECTION="${2:?Missing value for --response-selection}"
            shift 2
            ;;
        --full-corruption-localization-source)
            FULL_CORRUPTION_LOCALIZATION_SOURCE="${2:?Missing value for --full-corruption-localization-source}"
            shift 2
            ;;
        --utt)
            UTT="${2:?Missing value for --utt}"
            shift 2
            ;;
        --comparison)
            COMPARISON=1
            shift
            ;;
        --no-overwrite)
            OVERWRITE=0
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
            usage
            echo "Unexpected positional argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${EFFECT}" != "blur" && "${EFFECT}" != "block" ]]; then
    echo "Invalid --effect: ${EFFECT}. Expected blur or block." >&2
    exit 1
fi

if [[ "${GAZE_MAPPING}" != "legacy-extraction" && "${GAZE_MAPPING}" != "measured-player" ]]; then
    echo "Invalid --gaze-mapping: ${GAZE_MAPPING}. Expected legacy-extraction or measured-player." >&2
    exit 1
fi

if [[ "${RESPONSE_SELECTION}" != "latest-submitted" && "${RESPONSE_SELECTION}" != "first-response" ]]; then
    echo "Invalid --response-selection: ${RESPONSE_SELECTION}. Expected latest-submitted or first-response." >&2
    exit 1
fi

if [[ "${FULL_CORRUPTION_LOCALIZATION_SOURCE}" != "annotation-media" && "${FULL_CORRUPTION_LOCALIZATION_SOURCE}" != "video-json" ]]; then
    echo "Invalid --full-corruption-localization-source: ${FULL_CORRUPTION_LOCALIZATION_SOURCE}. Expected annotation-media or video-json." >&2
    exit 1
fi

if [[ "${FOCUS_REGION_SHAPE}" != "circle" && "${FOCUS_REGION_SHAPE}" != "square" ]]; then
    echo "Invalid --focus-region-shape: ${FOCUS_REGION_SHAPE}. Expected circle or square." >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build or copy the eyetrack image to: ${SIF_PATH}" >&2
    exit 1
fi

if [[ ! -d "${PUPIL_PARENT}" ]]; then
    echo "Pupil parent folder does not exist: ${PUPIL_PARENT}" >&2
    exit 1
fi

if [[ ! -d "${ANNOTATION_DIR}" ]]; then
    echo "Annotation directory does not exist: ${ANNOTATION_DIR}" >&2
    exit 1
fi

if [[ ! -f "${VIDEO_JSON}" ]]; then
    echo "Video JSON does not exist: ${VIDEO_JSON}" >&2
    exit 1
fi

if [[ ! -d "${SOURCE_VIDEOS}" ]]; then
    echo "Source video path prefix does not exist: ${SOURCE_VIDEOS}" >&2
    exit 1
fi

if [[ "${COMPARISON}" == "1" ]]; then
    FULL_CORRUPTION_OUTPUT_DIR="${FULL_CORRUPTION_OUTPUT_DIR%/}"
    if [[ "${FULL_CORRUPTION_OUTPUT_DIR##*/}" == "data" ]]; then
        FULL_CORRUPTION_OUTPUT_DIR="${FULL_CORRUPTION_OUTPUT_DIR%/*}/data_comparison"
    else
        FULL_CORRUPTION_OUTPUT_DIR="${FULL_CORRUPTION_OUTPUT_DIR}_comparison"
    fi
fi

mkdir -p "${FULL_CORRUPTION_OUTPUT_DIR}"

echo "Project root:         ${PROJECT_ROOT}"
echo "Apptainer image:      ${SIF_PATH}"
echo "Data root:            ${DEFAULT_DATA_ROOT}"
echo "Pupil parent:         ${PUPIL_PARENT}"
echo "Annotation dir:       ${ANNOTATION_DIR}"
echo "Video JSON:           ${VIDEO_JSON}"
echo "Source videos:        ${SOURCE_VIDEOS}"
echo "Full corruption out:  ${FULL_CORRUPTION_OUTPUT_DIR}"
echo "Media URL prefix:     ${MEDIA_URL_PREFIX}"
echo "Effect:               ${EFFECT}"
echo "Clip length:          ${CLIP_LENGTH}"
echo "Focus region ratio:   ${FOCUS_REGION_RATIO}"
echo "Focus region shape:   ${FOCUS_REGION_SHAPE}"
echo "Max gaze gap:         ${MAX_GAZE_GAP}"
echo "Gaze mapping:         ${GAZE_MAPPING}"
echo "Response selection:   ${RESPONSE_SELECTION}"
echo "Full corr source:     ${FULL_CORRUPTION_LOCALIZATION_SOURCE}"
echo "Utterance groups:     ${UTT:-all}"
echo "Comparison:           ${COMPARISON}"
echo "Overwrite:            ${OVERWRITE}"
echo "Final-segment output: disabled"
echo "Focus plots:          disabled"
echo "Debug overlays:       disabled"
echo "Original copies:      disabled"

PYTHON_ARGS=(
    python /workspace/eyetrack/gaze_blocked_partition.py
    "${FULL_CORRUPTION_OUTPUT_DIR}"
    "${PUPIL_PARENT}"
    "${ANNOTATION_DIR}"
    "${SOURCE_VIDEOS}"
    --video-json "${VIDEO_JSON}"
    --media-url-prefix "${MEDIA_URL_PREFIX}"
    --full-corruption-output-parent "${FULL_CORRUPTION_OUTPUT_DIR}"
    --effect "${EFFECT}"
    --clip-length "${CLIP_LENGTH}"
    --focus-region-ratio "${FOCUS_REGION_RATIO}"
    --focus-region-shape "${FOCUS_REGION_SHAPE}"
    --max-gaze-gap "${MAX_GAZE_GAP}"
    --gaze-mapping "${GAZE_MAPPING}"
    --response-selection "${RESPONSE_SELECTION}"
    --full-corruption-localization-source "${FULL_CORRUPTION_LOCALIZATION_SOURCE}"
    --skip-final-segment-output
)

if [[ -n "${UTT}" ]]; then
    PYTHON_ARGS+=(--utt "${UTT}")
fi

if [[ "${COMPARISON}" == "1" ]]; then
    PYTHON_ARGS+=(--comparison)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    PYTHON_ARGS+=(--overwrite)
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
