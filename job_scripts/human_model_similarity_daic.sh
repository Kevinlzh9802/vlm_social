#!/bin/bash
#SBATCH --job-name="human_model_sim"
#SBATCH --time=02:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/analysis.sif"
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/results"
DEFAULT_GEMINI_RESULTS_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench/human_eval/gemini"
DEFAULT_TASK_JSON="${DEFAULT_DATA_ROOT}/human_eval/task1/task1.json"
DEFAULT_ANNOTATION_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/annotations"
DEFAULT_EXTRACTION_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/annotation_extracted"
DEFAULT_PLOT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/human_model_plots"
DEFAULT_PLOT_DATA_DIR="${DEFAULT_DATA_ROOT}/human_eval/task1/human_model_plot_data"
DEFAULT_TASK_NUMBER="1"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--results-root PATH] [--gemini-results-root PATH] [--skip-gemini] [--task-json PATH] [--annotation-dir PATH] [--extraction-output-dir PATH] [--plot-dir PATH] [--plot-data-dir PATH] [--task-number N] [--model MODEL] [--model-path PATH] [--progress-partitions N]" >&2
    echo "  results-root: model results root (default: ${DEFAULT_RESULTS_ROOT})" >&2
    echo "  gemini-results-root: Gemini result tree from gemini_retrieve_daic.sh (default: ${DEFAULT_GEMINI_RESULTS_ROOT})" >&2
    echo "  --skip-gemini: do not include Gemini in human-model similarity analysis" >&2
    echo "  task-json: task media JSON used for linking (default: ${DEFAULT_TASK_JSON})" >&2
    echo "  annotation-dir: folder containing T1_y.json files (default: ${DEFAULT_ANNOTATION_DIR})" >&2
    echo "  extraction-output-dir: extracted CSV/JSON path (default: ${DEFAULT_EXTRACTION_OUTPUT_DIR})" >&2
    echo "  plot-dir: output folder for plot images (default: ${DEFAULT_PLOT_DIR})" >&2
    echo "  plot-data-dir: reusable plot-data folder (default: ${DEFAULT_PLOT_DATA_DIR})" >&2
    echo "  task-number: optional task number filter passed to the parser (default: ${DEFAULT_TASK_NUMBER})" >&2
    echo "  model: SentenceTransformer model name (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: optional local SentenceTransformer directory. Recommended on compute nodes." >&2
}

RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
GEMINI_RESULTS_ROOT="${DEFAULT_GEMINI_RESULTS_ROOT}"
TASK_JSON="${DEFAULT_TASK_JSON}"
ANNOTATION_DIR="${DEFAULT_ANNOTATION_DIR}"
EXTRACTION_OUTPUT_DIR="${DEFAULT_EXTRACTION_OUTPUT_DIR}"
PLOT_DIR="${DEFAULT_PLOT_DIR}"
PLOT_DATA_DIR="${DEFAULT_PLOT_DATA_DIR}"
TASK_NUMBER="${DEFAULT_TASK_NUMBER}"
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
PROGRESS_PARTITIONS="20"
SKIP_GEMINI=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT="${2:?Missing value for --results-root}"
            shift 2
            ;;
        --gemini-results-root)
            GEMINI_RESULTS_ROOT="${2:?Missing value for --gemini-results-root}"
            shift 2
            ;;
        --skip-gemini)
            SKIP_GEMINI=1
            shift
            ;;
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

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Results root does not exist: ${RESULTS_ROOT}" >&2
    exit 1
fi

if [[ "${SKIP_GEMINI}" == "0" && ! -d "${GEMINI_RESULTS_ROOT}" ]]; then
    echo "Gemini results root does not exist: ${GEMINI_RESULTS_ROOT}" >&2
    echo "Run gemini_retrieve_daic.sh after Gemini jobs complete, pass --gemini-results-root, or pass --skip-gemini." >&2
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

if [[ -n "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/vlm_social
mkdir -p "${EXTRACTION_OUTPUT_DIR}" "${PLOT_DIR}" "${PLOT_DATA_DIR}"

echo "Project root:           ${PROJECT_ROOT}"
echo "Apptainer image:        ${SIF_PATH}"
echo "Results root:           ${RESULTS_ROOT}"
echo "Gemini results root:    ${GEMINI_RESULTS_ROOT}"
echo "Skip Gemini:            ${SKIP_GEMINI}"
echo "Task JSON:              ${TASK_JSON}"
echo "Annotation dir:         ${ANNOTATION_DIR}"
echo "Extraction output dir:  ${EXTRACTION_OUTPUT_DIR}"
echo "Plot dir:               ${PLOT_DIR}"
echo "Plot data dir:          ${PLOT_DATA_DIR}"
echo "Task number:            ${TASK_NUMBER}"
echo "Model:                  ${MODEL}"
echo "Model path:             ${MODEL_PATH:-<download by name>}"
echo "Progress partitions:    ${PROGRESS_PARTITIONS}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/human_model_similarity.py
    "${ANNOTATION_DIR}"
    --task-json "${TASK_JSON}"
    --results-root "${RESULTS_ROOT}"
    --extraction-output-dir "${EXTRACTION_OUTPUT_DIR}"
    --plot-dir "${PLOT_DIR}"
    --plot-data-dir "${PLOT_DATA_DIR}"
    --task-number "${TASK_NUMBER}"
    --model "${MODEL}"
    --progress-partitions "${PROGRESS_PARTITIONS}"
)

if [[ "${SKIP_GEMINI}" == "0" ]]; then
    PYTHON_ARGS+=(--additional-results-root "${GEMINI_RESULTS_ROOT}")
fi

if [[ -n "${MODEL_PATH}" ]]; then
    PYTHON_ARGS+=(--model-path "${MODEL_PATH}")
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
