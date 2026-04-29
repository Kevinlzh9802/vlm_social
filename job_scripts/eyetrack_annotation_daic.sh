#!/bin/bash
#SBATCH --job-name="eyetrack_focus"
#SBATCH --time=04:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=logs/eyetrack_annotation_daic_%j.out
#SBATCH --error=logs/eyetrack_annotation_daic_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/eyetrack.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_PUPIL_PARENT="${DEFAULT_DATA_ROOT}/human_eval/pupil"
DEFAULT_ANNOTATION_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/results"
DEFAULT_VIDEO_JSON="${DEFAULT_DATA_ROOT}/human_eval/task2/task2.json"
DEFAULT_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/extraction_focus_new"
DEFAULT_LOCAL_PATH_PREFIX="${DEFAULT_DATA_ROOT}/human_eval/videos"
DEFAULT_MEDIA_URL_PREFIX="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_FOCUS_MAPPING="measured-player"
DEFAULT_RESPONSE_SELECTION="latest-submitted"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--annotation-dir PATH] [--video-json PATH] [--output-dir PATH] [--local-path-prefix PATH] [--media-url-prefix URL] [--focus-mapping legacy-extraction|measured-player] [--response-selection latest-submitted|first-response]" >&2
    echo "  pupil-parent: parent folder with T{x}_{y}_annotator1/2 Pupil recordings (default: ${DEFAULT_PUPIL_PARENT})" >&2
    echo "  annotation-dir: folder with T{x}_{y}.json annotation files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  video-json: ordered video list JSON (default: ${DEFAULT_VIDEO_JSON})" >&2
    echo "  output-dir: focus plot output folder (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  local-path-prefix: local path replacing the media URL prefix (default: ${DEFAULT_LOCAL_PATH_PREFIX})" >&2
    echo "  media-url-prefix: media URL prefix to replace (default: ${DEFAULT_MEDIA_URL_PREFIX})" >&2
    echo "  focus-mapping: screen-to-video transform, default ${DEFAULT_FOCUS_MAPPING}" >&2
    echo "  response-selection: annotation response selector, default ${DEFAULT_RESPONSE_SELECTION}" >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
VIDEO_JSON="${DEFAULT_VIDEO_JSON}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
LOCAL_PATH_PREFIX="${DEFAULT_LOCAL_PATH_PREFIX}"
MEDIA_URL_PREFIX="${DEFAULT_MEDIA_URL_PREFIX}"
FOCUS_MAPPING="${DEFAULT_FOCUS_MAPPING}"
RESPONSE_SELECTION="${DEFAULT_RESPONSE_SELECTION}"

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
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --local-path-prefix)
            LOCAL_PATH_PREFIX="${2:?Missing value for --local-path-prefix}"
            shift 2
            ;;
        --media-url-prefix)
            MEDIA_URL_PREFIX="${2:?Missing value for --media-url-prefix}"
            shift 2
            ;;
        --focus-mapping)
            FOCUS_MAPPING="${2:?Missing value for --focus-mapping}"
            shift 2
            ;;
        --response-selection)
            RESPONSE_SELECTION="${2:?Missing value for --response-selection}"
            shift 2
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

TIMING_CSV="${OUTPUT_DIR}/timing_tables.csv"
SUMMARY_CSV="${OUTPUT_DIR}/extraction_summary.csv"

if [[ "${FOCUS_MAPPING}" != "legacy-extraction" && "${FOCUS_MAPPING}" != "measured-player" ]]; then
    echo "Invalid --focus-mapping: ${FOCUS_MAPPING}. Expected legacy-extraction or measured-player." >&2
    exit 1
fi

if [[ "${RESPONSE_SELECTION}" != "latest-submitted" && "${RESPONSE_SELECTION}" != "first-response" ]]; then
    echo "Invalid --response-selection: ${RESPONSE_SELECTION}. Expected latest-submitted or first-response." >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
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

if [[ ! -d "${LOCAL_PATH_PREFIX}" ]]; then
    echo "Local path prefix does not exist: ${LOCAL_PATH_PREFIX}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Project root:       ${PROJECT_ROOT}"
echo "Apptainer image:    ${SIF_PATH}"
echo "Data root:          ${DEFAULT_DATA_ROOT}"
echo "Pupil parent:       ${PUPIL_PARENT}"
echo "Annotation dir:     ${ANNOTATION_DIR}"
echo "Video JSON:         ${VIDEO_JSON}"
echo "Output dir:         ${OUTPUT_DIR}"
echo "Timing CSV:         ${TIMING_CSV}"
echo "Summary CSV:        ${SUMMARY_CSV}"
echo "Local path prefix:  ${LOCAL_PATH_PREFIX}"
echo "Media URL prefix:   ${MEDIA_URL_PREFIX}"
echo "Focus mapping:      ${FOCUS_MAPPING}"
echo "Response selection: ${RESPONSE_SELECTION}"

PYTHON_ARGS=(
    python /workspace/eyetrack/eyetrack_annotation.py
    "${PUPIL_PARENT}"
    "${ANNOTATION_DIR}"
    "${LOCAL_PATH_PREFIX}"
    --video-json "${VIDEO_JSON}"
    --output-dir "${OUTPUT_DIR}"
    --timing-csv "${TIMING_CSV}"
    --summary-csv "${SUMMARY_CSV}"
    --media-url-prefix "${MEDIA_URL_PREFIX}"
    --focus-mapping "${FOCUS_MAPPING}"
    --response-selection "${RESPONSE_SELECTION}"
)

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
