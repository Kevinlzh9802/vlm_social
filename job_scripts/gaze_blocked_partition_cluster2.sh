#!/bin/bash
#SBATCH --job-name="gaze_block"
#SBATCH --time=08:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-type=END
#SBATCH --account=<account>
#SBATCH --output=logs/gaze_blocked_partition_<cluster2>_%j.out
#SBATCH --error=logs/gaze_blocked_partition_<cluster2>_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/eyetrack.sif"
DEFAULT_PUPIL_PARENT="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/pupil"
DEFAULT_ANNOTATION_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/results"
DEFAULT_VIDEO_JSON="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/task2.json"
DEFAULT_SOURCE_VIDEOS="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/videos"
DEFAULT_OUTPUT_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/manipulation"
DEFAULT_FULL_CORRUPTION_OUTPUT_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/manipulation_full"
DEFAULT_ORIGINAL_OUTPUT_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/original"
DEFAULT_FOCUS_PLOT_OUTPUT_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/extraction_focus_from_partition"
DEFAULT_DEBUG_OVERLAY_OUTPUT_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/gaze_overlay_debug"
DEFAULT_MEDIA_URL_PREFIX="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_EFFECT="block"
DEFAULT_CLIP_LENGTH="0.5"
DEFAULT_FOCUS_REGION_RATIO="0.18"
DEFAULT_FOCUS_REGION_SHAPE="circle"
DEFAULT_MAX_GAZE_GAP="0.5"
DEFAULT_GAZE_MAPPING="legacy-extraction"
DEFAULT_RESPONSE_SELECTION="latest-submitted"
DEFAULT_FULL_CORRUPTION_LOCALIZATION_SOURCE="annotation-media"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--annotation-dir PATH] [--video-json PATH] [--source-videos PATH] [--output-dir PATH] [--full-corruption-output-dir PATH] [--original-output-dir PATH] [--focus-plot-output-dir PATH] [--debug-overlay-output-dir PATH] [--media-url-prefix URL] [--effect blur|block] [--clip-length SEC] [--focus-region-ratio RATIO] [--focus-region-shape circle|square] [--max-gaze-gap SEC] [--gaze-mapping legacy-extraction|measured-player] [--response-selection latest-submitted|first-response] [--full-corruption-localization-source annotation-media|video-json] [--utt 1,2,3] [--with-final-segment-output] [--no-focus-plots] [--no-debug-overlay] [--no-overwrite]" >&2
    echo "  pupil-parent: parent folder with T{x}_{y}_annotator1/2 Pupil recordings (default: ${DEFAULT_PUPIL_PARENT})" >&2
    echo "  annotation-dir: folder with T{x}_{y}.json annotation files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  video-json: fallback ordered video list JSON (default: ${DEFAULT_VIDEO_JSON})" >&2
    echo "  source-videos: local path replacing the media URL prefix (default: ${DEFAULT_SOURCE_VIDEOS})" >&2
    echo "  output-dir: final-segment manipulated output parent (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  full-corruption-output-dir: whole-clip gaze-corrupted output parent (default: ${DEFAULT_FULL_CORRUPTION_OUTPUT_DIR})" >&2
    echo "  original-output-dir: unmanipulated copied output parent (default: ${DEFAULT_ORIGINAL_OUTPUT_DIR})" >&2
    echo "  focus-plot-output-dir: partition-pipeline gaze focus plots (default: ${DEFAULT_FOCUS_PLOT_OUTPUT_DIR})" >&2
    echo "  debug-overlay-output-dir: per-frame gaze overlay videos and CSVs (default: ${DEFAULT_DEBUG_OVERLAY_OUTPUT_DIR})" >&2
    echo "  --with-final-segment-output: also generate final-0.5s corrupted cumulative clips (default: skipped)" >&2
    echo "  --no-focus-plots: skip partition-pipeline static gaze focus plots" >&2
    echo "  --no-debug-overlay: skip per-frame gaze debug overlay videos" >&2
    echo "  media-url-prefix: media URL prefix to replace (default: ${DEFAULT_MEDIA_URL_PREFIX})" >&2
    echo "  effect: manipulation effect, default ${DEFAULT_EFFECT}" >&2
    echo "  gaze-mapping: screen-to-video transform, default ${DEFAULT_GAZE_MAPPING}" >&2
    echo "  response-selection: annotation response selector, default ${DEFAULT_RESPONSE_SELECTION}" >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
VIDEO_JSON="${DEFAULT_VIDEO_JSON}"
SOURCE_VIDEOS="${DEFAULT_SOURCE_VIDEOS}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
FULL_CORRUPTION_OUTPUT_DIR="${DEFAULT_FULL_CORRUPTION_OUTPUT_DIR}"
ORIGINAL_OUTPUT_DIR="${DEFAULT_ORIGINAL_OUTPUT_DIR}"
FOCUS_PLOT_OUTPUT_DIR="${DEFAULT_FOCUS_PLOT_OUTPUT_DIR}"
DEBUG_OVERLAY_OUTPUT_DIR="${DEFAULT_DEBUG_OVERLAY_OUTPUT_DIR}"
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
OVERWRITE=1
SKIP_FINAL_SEGMENT_OUTPUT=1
WRITE_FOCUS_PLOTS=1
WRITE_DEBUG_OVERLAY=1

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
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --full-corruption-output-dir)
            FULL_CORRUPTION_OUTPUT_DIR="${2:?Missing value for --full-corruption-output-dir}"
            shift 2
            ;;
        --original-output-dir)
            ORIGINAL_OUTPUT_DIR="${2:?Missing value for --original-output-dir}"
            shift 2
            ;;
        --focus-plot-output-dir)
            FOCUS_PLOT_OUTPUT_DIR="${2:?Missing value for --focus-plot-output-dir}"
            shift 2
            ;;
        --debug-overlay-output-dir)
            DEBUG_OVERLAY_OUTPUT_DIR="${2:?Missing value for --debug-overlay-output-dir}"
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
        --no-overwrite)
            OVERWRITE=0
            shift
            ;;
        --with-final-segment-output)
            SKIP_FINAL_SEGMENT_OUTPUT=0
            shift
            ;;
        --no-focus-plots)
            WRITE_FOCUS_PLOTS=0
            shift
            ;;
        --no-debug-overlay)
            WRITE_DEBUG_OVERLAY=0
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
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build eyetrack.sif eyetrack.def" >&2
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

mkdir -p "${OUTPUT_DIR}" "${FULL_CORRUPTION_OUTPUT_DIR}" "${ORIGINAL_OUTPUT_DIR}"
if [[ "${WRITE_FOCUS_PLOTS}" == "1" ]]; then
    mkdir -p "${FOCUS_PLOT_OUTPUT_DIR}"
fi
if [[ "${WRITE_DEBUG_OVERLAY}" == "1" ]]; then
    mkdir -p "${DEBUG_OVERLAY_OUTPUT_DIR}"
fi

echo "Project root:        ${PROJECT_ROOT}"
echo "Apptainer image:     ${SIF_PATH}"
echo "Pupil parent:        ${PUPIL_PARENT}"
echo "Annotation dir:      ${ANNOTATION_DIR}"
echo "Video JSON:          ${VIDEO_JSON}"
echo "Source videos:       ${SOURCE_VIDEOS}"
echo "Final-segment output: ${OUTPUT_DIR}"
echo "Full corruption out:  ${FULL_CORRUPTION_OUTPUT_DIR}"
echo "Original output:      ${ORIGINAL_OUTPUT_DIR}"
echo "Focus plot output:    $([[ "${WRITE_FOCUS_PLOTS}" == "1" ]] && echo "${FOCUS_PLOT_OUTPUT_DIR}" || echo "disabled")"
echo "Debug overlay output: $([[ "${WRITE_DEBUG_OVERLAY}" == "1" ]] && echo "${DEBUG_OVERLAY_OUTPUT_DIR}" || echo "disabled")"
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
echo "Overwrite:            ${OVERWRITE}"
echo "Skip final segment:   ${SKIP_FINAL_SEGMENT_OUTPUT}"

PYTHON_ARGS=(
    python /workspace/eyetrack/gaze_blocked_partition.py
    "${OUTPUT_DIR}"
    "${PUPIL_PARENT}"
    "${ANNOTATION_DIR}"
    "${SOURCE_VIDEOS}"
    --video-json "${VIDEO_JSON}"
    --media-url-prefix "${MEDIA_URL_PREFIX}"
    --original-output-parent "${ORIGINAL_OUTPUT_DIR}"
    --full-corruption-output-parent "${FULL_CORRUPTION_OUTPUT_DIR}"
    --effect "${EFFECT}"
    --clip-length "${CLIP_LENGTH}"
    --focus-region-ratio "${FOCUS_REGION_RATIO}"
    --focus-region-shape "${FOCUS_REGION_SHAPE}"
    --max-gaze-gap "${MAX_GAZE_GAP}"
    --gaze-mapping "${GAZE_MAPPING}"
    --response-selection "${RESPONSE_SELECTION}"
    --full-corruption-localization-source "${FULL_CORRUPTION_LOCALIZATION_SOURCE}"
)

if [[ "${WRITE_FOCUS_PLOTS}" == "1" ]]; then
    PYTHON_ARGS+=(--focus-plot-output-parent "${FOCUS_PLOT_OUTPUT_DIR}")
fi

if [[ "${WRITE_DEBUG_OVERLAY}" == "1" ]]; then
    PYTHON_ARGS+=(--debug-overlay-output-parent "${DEBUG_OVERLAY_OUTPUT_DIR}")
fi

if [[ -n "${UTT}" ]]; then
    PYTHON_ARGS+=(--utt "${UTT}")
fi

if [[ "${SKIP_FINAL_SEGMENT_OUTPUT}" == "1" ]]; then
    PYTHON_ARGS+=(--skip-final-segment-output)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    PYTHON_ARGS+=(--overwrite)
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
