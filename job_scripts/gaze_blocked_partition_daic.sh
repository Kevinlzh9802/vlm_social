#!/bin/bash
#SBATCH --job-name="gaze_block"
#SBATCH --time=08:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/eyetrack.sif"
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_PUPIL_PARENT="${DEFAULT_DATA_ROOT}/human_eval/pupil"
DEFAULT_ANNOTATION_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/results"
DEFAULT_VIDEO_JSON="${DEFAULT_DATA_ROOT}/human_eval/task2/task2.json"
DEFAULT_SOURCE_VIDEOS="${DEFAULT_DATA_ROOT}/human_eval/videos"
DEFAULT_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation"
DEFAULT_FULL_CORRUPTION_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full"
DEFAULT_ORIGINAL_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/original"
DEFAULT_MEDIA_URL_PREFIX="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_EFFECT="block"
DEFAULT_CLIP_LENGTH="0.5"
DEFAULT_FOCUS_BOX_RATIO="0.18"
DEFAULT_MAX_GAZE_GAP="0.5"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--annotation-dir PATH] [--video-json PATH] [--source-videos PATH] [--output-dir PATH] [--full-corruption-output-dir PATH] [--original-output-dir PATH] [--media-url-prefix URL] [--effect blur|block] [--clip-length SEC] [--focus-box-ratio RATIO] [--max-gaze-gap SEC] [--utt 1,2,3] [--with-final-segment-output] [--no-overwrite]" >&2
    echo "  pupil-parent: parent folder with T{x}_{y}_annotator1/2 Pupil recordings (default: ${DEFAULT_PUPIL_PARENT})" >&2
    echo "  annotation-dir: folder with T{x}_{y}.json annotation files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  video-json: fallback ordered video list JSON (default: ${DEFAULT_VIDEO_JSON})" >&2
    echo "  source-videos: local path replacing the media URL prefix (default: ${DEFAULT_SOURCE_VIDEOS})" >&2
    echo "  output-dir: final-segment manipulated output parent (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  full-corruption-output-dir: whole-clip gaze-corrupted output parent (default: ${DEFAULT_FULL_CORRUPTION_OUTPUT_DIR})" >&2
    echo "  original-output-dir: unmanipulated copied output parent (default: ${DEFAULT_ORIGINAL_OUTPUT_DIR})" >&2
    echo "  --with-final-segment-output: also generate final-0.5s corrupted cumulative clips (default: skipped)" >&2
    echo "  media-url-prefix: media URL prefix to replace (default: ${DEFAULT_MEDIA_URL_PREFIX})" >&2
    echo "  effect: manipulation effect, default ${DEFAULT_EFFECT}" >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
VIDEO_JSON="${DEFAULT_VIDEO_JSON}"
SOURCE_VIDEOS="${DEFAULT_SOURCE_VIDEOS}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
FULL_CORRUPTION_OUTPUT_DIR="${DEFAULT_FULL_CORRUPTION_OUTPUT_DIR}"
ORIGINAL_OUTPUT_DIR="${DEFAULT_ORIGINAL_OUTPUT_DIR}"
MEDIA_URL_PREFIX="${DEFAULT_MEDIA_URL_PREFIX}"
EFFECT="${DEFAULT_EFFECT}"
CLIP_LENGTH="${DEFAULT_CLIP_LENGTH}"
FOCUS_BOX_RATIO="${DEFAULT_FOCUS_BOX_RATIO}"
MAX_GAZE_GAP="${DEFAULT_MAX_GAZE_GAP}"
UTT=""
OVERWRITE=1
SKIP_FINAL_SEGMENT_OUTPUT=1

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
        --focus-box-ratio)
            FOCUS_BOX_RATIO="${2:?Missing value for --focus-box-ratio}"
            shift 2
            ;;
        --max-gaze-gap)
            MAX_GAZE_GAP="${2:?Missing value for --max-gaze-gap}"
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

mkdir -p "${OUTPUT_DIR}" "${FULL_CORRUPTION_OUTPUT_DIR}" "${ORIGINAL_OUTPUT_DIR}"

echo "Project root:         ${PROJECT_ROOT}"
echo "Apptainer image:      ${SIF_PATH}"
echo "Data root:            ${DEFAULT_DATA_ROOT}"
echo "Pupil parent:         ${PUPIL_PARENT}"
echo "Annotation dir:       ${ANNOTATION_DIR}"
echo "Video JSON:           ${VIDEO_JSON}"
echo "Source videos:        ${SOURCE_VIDEOS}"
echo "Final-segment output: ${OUTPUT_DIR}"
echo "Full corruption out:  ${FULL_CORRUPTION_OUTPUT_DIR}"
echo "Original output:      ${ORIGINAL_OUTPUT_DIR}"
echo "Media URL prefix:     ${MEDIA_URL_PREFIX}"
echo "Effect:               ${EFFECT}"
echo "Clip length:          ${CLIP_LENGTH}"
echo "Focus box ratio:      ${FOCUS_BOX_RATIO}"
echo "Max gaze gap:         ${MAX_GAZE_GAP}"
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
    --focus-box-ratio "${FOCUS_BOX_RATIO}"
    --max-gaze-gap "${MAX_GAZE_GAP}"
)

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
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
