#!/bin/bash
#SBATCH --job-name="eyetrack_focus"
#SBATCH --time=04:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/zli33/projects/vlm_social"
LEGACY_ROOT="${PROJECT_ROOT}/legacy_64b3e28"
SIF_PATH="/scratch/zli33/apptainers/eyetrack.sif"
DEFAULT_PUPIL_PARENT="/scratch/zli33/data/gestalt_bench/human_eval/pupil"
DEFAULT_ANNOTATION_DIR="/scratch/zli33/data/gestalt_bench/human_eval/task2/results"
DEFAULT_VIDEO_JSON="/scratch/zli33/data/gestalt_bench/human_eval/task2/task2.json"
DEFAULT_OUTPUT_DIR="/scratch/zli33/data/gestalt_bench/human_eval/task2/extraction_focus_old"
DEFAULT_LOCAL_PATH_PREFIX="/scratch/zli33/data/gestalt_bench/human_eval/videos"
DEFAULT_MEDIA_URL_PREFIX="http://localhost:5000/api/media/gestalt_bench/annotation1/"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--annotation-dir PATH] [--video-json PATH] [--output-dir PATH] [--local-path-prefix PATH] [--media-url-prefix URL]" >&2
    echo "  pupil-parent: parent folder with T{x}_{y}_annotator1/2 Pupil recordings (default: ${DEFAULT_PUPIL_PARENT})" >&2
    echo "  annotation-dir: folder with T{x}_{y}.json annotation files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  video-json: ordered video list JSON (default: ${DEFAULT_VIDEO_JSON})" >&2
    echo "  output-dir: focus plot output folder (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  local-path-prefix: local path replacing the media URL prefix (default: ${DEFAULT_LOCAL_PATH_PREFIX})" >&2
    echo "  media-url-prefix: media URL prefix to replace (default: ${DEFAULT_MEDIA_URL_PREFIX})" >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
VIDEO_JSON="${DEFAULT_VIDEO_JSON}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
LOCAL_PATH_PREFIX="${DEFAULT_LOCAL_PATH_PREFIX}"
MEDIA_URL_PREFIX="${DEFAULT_MEDIA_URL_PREFIX}"

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

if [[ ! -d "${LEGACY_ROOT}" ]]; then
    echo "Missing legacy source folder: ${LEGACY_ROOT}" >&2
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

if [[ ! -d "${LOCAL_PATH_PREFIX}" ]]; then
    echo "Local path prefix does not exist: ${LOCAL_PATH_PREFIX}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Project root:       ${PROJECT_ROOT}"
echo "Legacy root:        ${LEGACY_ROOT}"
echo "Apptainer image:    ${SIF_PATH}"
echo "Pupil parent:       ${PUPIL_PARENT}"
echo "Annotation dir:     ${ANNOTATION_DIR}"
echo "Video JSON:         ${VIDEO_JSON}"
echo "Output dir:         ${OUTPUT_DIR}"
echo "Timing CSV:         ${TIMING_CSV}"
echo "Summary CSV:        ${SUMMARY_CSV}"
echo "Local path prefix:  ${LOCAL_PATH_PREFIX}"
echo "Media URL prefix:   ${MEDIA_URL_PREFIX}"

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
)

srun apptainer exec \
    --bind "${LEGACY_ROOT}:/workspace" \
    --bind /home/zli33:/home/zli33 \
    --bind /scratch/zli33:/scratch/zli33 \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
