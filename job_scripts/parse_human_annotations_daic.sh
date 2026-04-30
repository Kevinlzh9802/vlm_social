#!/bin/bash
#SBATCH --job-name="human_annot"
#SBATCH --time=00:30:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=logs/parse_human_annotations_daic_%j.out
#SBATCH --error=logs/parse_human_annotations_daic_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/vlm_social DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/eyetrack.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_TASK_JSON="${DEFAULT_DATA_ROOT}/human_eval/task1/task1.json"
DEFAULT_ANNOTATION_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/annotations"
DEFAULT_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/annotation_extracted"
DEFAULT_TASK_NUMBER="1"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--task-json PATH] [--annotation-dir PATH] [--output-dir PATH] [--task-number N]" >&2
    echo "  task-json: task media JSON used for linking (default: ${DEFAULT_TASK_JSON})" >&2
    echo "  annotation-dir: folder containing T1_y.json files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  output-dir: folder for extracted CSV/JSON outputs (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  task-number: optional task number filter passed to the parser (default: ${DEFAULT_TASK_NUMBER})" >&2
}

TASK_JSON="${DEFAULT_TASK_JSON}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
TASK_NUMBER="${DEFAULT_TASK_NUMBER}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task-json)
            TASK_JSON="${2:?Missing value for --task-json}"
            shift 2
            ;;
        --annotation-dir)
            ANNOTATION_DIR="${2:?Missing value for --annotation-dir}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --task-number)
            TASK_NUMBER="${2:?Missing value for --task-number}"
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

case "${TASK_NUMBER}" in
    ''|*[!0-9]*)
        echo "Invalid --task-number: ${TASK_NUMBER}. Expected a positive integer." >&2
        exit 1
        ;;
esac

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    exit 1
fi

if [[ ! -f "${TASK_JSON}" ]]; then
    echo "Task JSON does not exist: ${TASK_JSON}" >&2
    exit 1
fi

if [[ ! -d "${ANNOTATION_DIR}" ]]; then
    echo "Annotation directory does not exist: ${ANNOTATION_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p logs/vlm_social

OUTPUT_CSV="${OUTPUT_DIR}/human_annotations.csv"
OUTPUT_JSON="${OUTPUT_DIR}/human_annotations_linked.json"

echo "Project root:    ${PROJECT_ROOT}"
echo "Apptainer image: ${SIF_PATH}"
echo "Task JSON:       ${TASK_JSON}"
echo "Annotation dir:  ${ANNOTATION_DIR}"
echo "Output dir:      ${OUTPUT_DIR}"
echo "Output CSV:      ${OUTPUT_CSV}"
echo "Output JSON:     ${OUTPUT_JSON}"
echo "Task number:     ${TASK_NUMBER}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/parse_human_annotations.py
    "${ANNOTATION_DIR}"
    --task-json "${TASK_JSON}"
    --output-csv "${OUTPUT_CSV}"
    --output-json "${OUTPUT_JSON}"
    --task-number "${TASK_NUMBER}"
)

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
