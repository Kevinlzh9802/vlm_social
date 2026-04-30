#!/bin/bash
#SBATCH --job-name="gaze_focus"
#SBATCH --time=04:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/gaze_blocked_focus_<cluster2>_%j.out
#SBATCH --error=logs/gaze_blocked_focus_<cluster2>_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/eyetrack.sif"
DEFAULT_PUPIL_PARENT="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/pupil"
DEFAULT_ANNOTATION_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/results"
DEFAULT_VIDEO_JSON="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/task2.json"
DEFAULT_SOURCE_VIDEOS="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/videos"
DEFAULT_OUTPUT_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/task2/extraction_focus_gaze_blocked"
DEFAULT_MEDIA_URL_PREFIX="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_GAZE_MAPPING="legacy-extraction"
DEFAULT_RESPONSE_SELECTION="latest-submitted"
DEFAULT_UTT=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--annotation-dir PATH] [--video-json PATH] [--source-videos PATH] [--output-dir PATH] [--media-url-prefix URL] [--gaze-mapping legacy-extraction|measured-player] [--response-selection latest-submitted|first-response] [--utt 1,2,3] [--no-overwrite]" >&2
    echo "  Writes only gaze_blocked_partition focus plots plus annotation_sources.csv/provenance.json. No videos are generated." >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
VIDEO_JSON="${DEFAULT_VIDEO_JSON}"
SOURCE_VIDEOS="${DEFAULT_SOURCE_VIDEOS}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
MEDIA_URL_PREFIX="${DEFAULT_MEDIA_URL_PREFIX}"
GAZE_MAPPING="${DEFAULT_GAZE_MAPPING}"
RESPONSE_SELECTION="${DEFAULT_RESPONSE_SELECTION}"
UTT="${DEFAULT_UTT}"
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
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --media-url-prefix)
            MEDIA_URL_PREFIX="${2:?Missing value for --media-url-prefix}"
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
        --utt)
            UTT="${2:?Missing value for --utt}"
            shift 2
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

if [[ "${GAZE_MAPPING}" != "legacy-extraction" && "${GAZE_MAPPING}" != "measured-player" ]]; then
    echo "Invalid --gaze-mapping: ${GAZE_MAPPING}. Expected legacy-extraction or measured-player." >&2
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

for required_dir in "${PUPIL_PARENT}" "${ANNOTATION_DIR}" "${SOURCE_VIDEOS}"; do
    if [[ ! -e "${required_dir}" ]]; then
        echo "Required path does not exist: ${required_dir}" >&2
        exit 1
    fi
done

if [[ ! -f "${VIDEO_JSON}" ]]; then
    echo "Video JSON does not exist: ${VIDEO_JSON}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Project root:       ${PROJECT_ROOT}"
echo "Apptainer image:    ${SIF_PATH}"
echo "Output dir:         ${OUTPUT_DIR}"
echo "Pupil parent:       ${PUPIL_PARENT}"
echo "Annotation dir:     ${ANNOTATION_DIR}"
echo "Video JSON:         ${VIDEO_JSON}"
echo "Source videos:      ${SOURCE_VIDEOS}"
echo "Gaze mapping:       ${GAZE_MAPPING}"
echo "Response selection: ${RESPONSE_SELECTION}"
echo "Utterance groups:   ${UTT:-all}"
echo "Overwrite:          ${OVERWRITE}"

PYTHON_ARGS=(
    python /workspace/eyetrack/gaze_blocked_partition.py
    "${OUTPUT_DIR}/unused_final_segment"
    "${PUPIL_PARENT}"
    "${ANNOTATION_DIR}"
    "${SOURCE_VIDEOS}"
    --video-json "${VIDEO_JSON}"
    --media-url-prefix "${MEDIA_URL_PREFIX}"
    --focus-plot-output-parent "${OUTPUT_DIR}"
    --skip-final-segment-output
    --gaze-mapping "${GAZE_MAPPING}"
    --response-selection "${RESPONSE_SELECTION}"
)

if [[ -n "${UTT}" ]]; then
    PYTHON_ARGS+=(--utt "${UTT}")
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
