#!/bin/bash
#SBATCH --job-name="human_sim"
#SBATCH --time=00:10:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=logs/human_annotation_similarity_<cluster1>_%j.out
#SBATCH --error=logs/human_annotation_similarity_<cluster1>_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/analysis.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_TASK_JSON="${DEFAULT_DATA_ROOT}/human_eval/task1"
DEFAULT_ANNOTATION_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/annotations"
DEFAULT_EXTRACTION_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/annotation_extracted"
DEFAULT_PLOT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/plots"
DEFAULT_PLOT_DATA_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/plot_data"
DEFAULT_TASK_NUMBER="1"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--task-json PATH] [--annotation-dir PATH] [--extraction-output-dir PATH] [--plot-dir PATH] [--plot-data-dir PATH] [--task-number N] [--model MODEL] [--model-path PATH] [--progress-partitions N]" >&2
    echo "  task-json: task media JSON file or directory used for linking (default: ${DEFAULT_TASK_JSON})" >&2
    echo "  annotation-dir: folder containing T1_y.json or T1_bx_y.json files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  extraction-output-dir: extracted CSV/JSON path (default: ${DEFAULT_EXTRACTION_OUTPUT_DIR})" >&2
    echo "  plot-dir: output folder for plot images (default: ${DEFAULT_PLOT_DIR})" >&2
    echo "  plot-data-dir: reusable plot-data folder (default: ${DEFAULT_PLOT_DATA_DIR})" >&2
    echo "  task-number: optional task number filter passed to the parser (default: ${DEFAULT_TASK_NUMBER})" >&2
    echo "  model: SentenceTransformer model name (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: optional local SentenceTransformer directory. Recommended on compute nodes." >&2
}

TASK_JSON="${DEFAULT_TASK_JSON}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
EXTRACTION_OUTPUT_DIR="${DEFAULT_EXTRACTION_OUTPUT_DIR}"
PLOT_DIR="${DEFAULT_PLOT_DIR}"
PLOT_DATA_DIR="${DEFAULT_PLOT_DATA_DIR}"
TASK_NUMBER="${DEFAULT_TASK_NUMBER}"
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
PROGRESS_PARTITIONS="20"

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
        --extraction-output-dir)
            EXTRACTION_OUTPUT_DIR="${2:?Missing value for --extraction-output-dir}"
            shift 2
            ;;
        --plot-dir)
            PLOT_DIR="${2:?Missing value for --plot-dir}"
            shift 2
            ;;
        --plot-data-dir)
            PLOT_DATA_DIR="${2:?Missing value for --plot-data-dir}"
            shift 2
            ;;
        --task-number)
            TASK_NUMBER="${2:?Missing value for --task-number}"
            shift 2
            ;;
        --model)
            MODEL="${2:?Missing value for --model}"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="${2:?Missing value for --model-path}"
            shift 2
            ;;
        --progress-partitions)
            PROGRESS_PARTITIONS="${2:?Missing value for --progress-partitions}"
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

case "${PROGRESS_PARTITIONS}" in
    ''|*[!0-9]*)
        echo "Invalid --progress-partitions: ${PROGRESS_PARTITIONS}. Expected a positive integer." >&2
        exit 1
        ;;
esac

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build or copy it from ${PROJECT_ROOT}/apptainer/analysis.def first." >&2
    exit 1
fi

if [[ ! -e "${TASK_JSON}" ]]; then
    echo "Task JSON path does not exist: ${TASK_JSON}" >&2
    exit 1
fi

if [[ ! -d "${ANNOTATION_DIR}" ]]; then
    echo "Annotation directory does not exist: ${ANNOTATION_DIR}" >&2
    exit 1
fi

if [[ -n "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p logs/gesbench
mkdir -p "${EXTRACTION_OUTPUT_DIR}" "${PLOT_DIR}" "${PLOT_DATA_DIR}"

echo "Project root:           ${PROJECT_ROOT}"
echo "Apptainer image:        ${SIF_PATH}"
echo "Task JSON path:         ${TASK_JSON}"
echo "Annotation dir:         ${ANNOTATION_DIR}"
echo "Extraction output dir:  ${EXTRACTION_OUTPUT_DIR}"
echo "Plot dir:               ${PLOT_DIR}"
echo "Plot data dir:          ${PLOT_DATA_DIR}"
echo "Task number:            ${TASK_NUMBER}"
echo "Model:                  ${MODEL}"
echo "Model path:             ${MODEL_PATH:-<download by name>}"
echo "Progress partitions:    ${PROGRESS_PARTITIONS}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/human_annotation_similarity.py
    "${ANNOTATION_DIR}"
    --task-json "${TASK_JSON}"
    --extraction-output-dir "${EXTRACTION_OUTPUT_DIR}"
    --plot-dir "${PLOT_DIR}"
    --plot-data-dir "${PLOT_DATA_DIR}"
    --task-number "${TASK_NUMBER}"
    --model "${MODEL}"
    --progress-partitions "${PROGRESS_PARTITIONS}"
)

if [[ -n "${MODEL_PATH}" ]]; then
    PYTHON_ARGS+=(--model-path "${MODEL_PATH}")
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"

POINTS_CSV="${PLOT_DATA_DIR}/partial_to_full_points.csv"
if [[ ! -f "${POINTS_CSV}" ]]; then
    echo "Expected point CSV was not created: ${POINTS_CSV}" >&2
    exit 1
fi

IFS= read -r POINTS_HEADER < "${POINTS_CSV}" || true
POINTS_HEADER="${POINTS_HEADER%$'\r'}"
if [[ ",${POINTS_HEADER}," != *",neighboring_similarity_to_next,"* ]]; then
    echo "Point CSV is missing neighboring_similarity_to_next: ${POINTS_CSV}" >&2
    echo "Make sure the cluster is running the updated project code mounted at ${PROJECT_ROOT}." >&2
    exit 1
fi

echo "Verified point CSV includes neighboring_similarity_to_next: ${POINTS_CSV}"
