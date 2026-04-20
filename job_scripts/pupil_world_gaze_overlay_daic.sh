#!/bin/bash
#SBATCH --job-name="world_gaze"
#SBATCH --time=04:00:00
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
DEFAULT_OUTPUT_NAME="world_gaze.mp4"
DEFAULT_CSV_NAME="world_gaze_points.csv"
DEFAULT_SUMMARY_CSV="${DEFAULT_PUPIL_PARENT}/world_gaze_overlay_summary.csv"
DEFAULT_CONFIDENCE="0.6"
DEFAULT_MAX_GAZE_GAP="0.05"
DEFAULT_RADIUS="24"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--pupil-parent PATH] [--output-name NAME] [--csv-name NAME] [--summary-csv PATH] [--confidence FLOAT] [--max-gaze-gap SEC] [--radius PX] [--recursive] [--no-overwrite] [--no-label]" >&2
    echo "  pupil-parent: parent folder with Pupil recordings (default: ${DEFAULT_PUPIL_PARENT})" >&2
    echo "  output-name: overlay video written inside each recording folder (default: ${DEFAULT_OUTPUT_NAME})" >&2
    echo "  csv-name: per-frame gaze CSV written inside each recording folder (default: ${DEFAULT_CSV_NAME})" >&2
}

PUPIL_PARENT="${DEFAULT_PUPIL_PARENT}"
OUTPUT_NAME="${DEFAULT_OUTPUT_NAME}"
CSV_NAME="${DEFAULT_CSV_NAME}"
SUMMARY_CSV="${DEFAULT_SUMMARY_CSV}"
CONFIDENCE="${DEFAULT_CONFIDENCE}"
MAX_GAZE_GAP="${DEFAULT_MAX_GAZE_GAP}"
RADIUS="${DEFAULT_RADIUS}"
RECURSIVE=0
OVERWRITE=1
LABEL=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pupil-parent)
            PUPIL_PARENT="${2:?Missing value for --pupil-parent}"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="${2:?Missing value for --output-name}"
            shift 2
            ;;
        --csv-name)
            CSV_NAME="${2:?Missing value for --csv-name}"
            shift 2
            ;;
        --summary-csv)
            SUMMARY_CSV="${2:?Missing value for --summary-csv}"
            shift 2
            ;;
        --confidence)
            CONFIDENCE="${2:?Missing value for --confidence}"
            shift 2
            ;;
        --max-gaze-gap)
            MAX_GAZE_GAP="${2:?Missing value for --max-gaze-gap}"
            shift 2
            ;;
        --radius)
            RADIUS="${2:?Missing value for --radius}"
            shift 2
            ;;
        --recursive)
            RECURSIVE=1
            shift
            ;;
        --no-overwrite)
            OVERWRITE=0
            shift
            ;;
        --no-label)
            LABEL=0
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

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    exit 1
fi

if [[ ! -d "${PUPIL_PARENT}" ]]; then
    echo "Pupil parent folder does not exist: ${PUPIL_PARENT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${SUMMARY_CSV}")"

echo "Project root:      ${PROJECT_ROOT}"
echo "Apptainer image:   ${SIF_PATH}"
echo "Pupil parent:      ${PUPIL_PARENT}"
echo "Output name:       ${OUTPUT_NAME}"
echo "CSV name:          ${CSV_NAME}"
echo "Summary CSV:       ${SUMMARY_CSV}"
echo "Confidence:        ${CONFIDENCE}"
echo "Max gaze gap:      ${MAX_GAZE_GAP}"
echo "Marker radius:     ${RADIUS}"
echo "Recursive search:  ${RECURSIVE}"
echo "Overwrite:         ${OVERWRITE}"
echo "Draw label:        ${LABEL}"

PYTHON_ARGS=(
    python /workspace/eyetrack/pupil_world_gaze_overlay.py
    "${PUPIL_PARENT}"
    --output-name "${OUTPUT_NAME}"
    --csv-name "${CSV_NAME}"
    --summary-csv "${SUMMARY_CSV}"
    --confidence "${CONFIDENCE}"
    --max-gaze-gap "${MAX_GAZE_GAP}"
    --radius "${RADIUS}"
)

if [[ "${RECURSIVE}" == "1" ]]; then
    PYTHON_ARGS+=(--recursive)
fi

if [[ "${OVERWRITE}" == "1" ]]; then
    PYTHON_ARGS+=(--overwrite)
fi

if [[ "${LABEL}" == "0" ]]; then
    PYTHON_ARGS+=(--no-label)
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
