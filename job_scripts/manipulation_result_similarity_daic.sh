#!/bin/bash
#SBATCH --job-name="manip_result_sim"
#SBATCH --time=00:08:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/analysis.sif"
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_SOURCE_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/results"
DEFAULT_COMPARISON_SOURCE_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/results_comparison"
DEFAULT_REFERENCE_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/results"
DEFAULT_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/result_similarity"
DEFAULT_COMPARISON_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/results_similarity_comparison"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--comparison] [--source-results-root PATH] [--reference-results-root PATH] [--output-dir PATH] [--model MODEL] [--model-path PATH]" >&2
    echo "  source-results-root: manipulated task-2 result tree (default: ${DEFAULT_SOURCE_RESULTS_ROOT})" >&2
    echo "  --comparison: use comparison source/output defaults (${DEFAULT_COMPARISON_SOURCE_RESULTS_ROOT}, ${DEFAULT_COMPARISON_OUTPUT_DIR})" >&2
    echo "  reference-results-root: full benchmark result tree (default: ${DEFAULT_REFERENCE_RESULTS_ROOT})" >&2
    echo "  output-dir: table and point-data output directory (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  model: SentenceTransformer model name (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: optional local SentenceTransformer directory. Recommended on compute nodes." >&2
}

SOURCE_RESULTS_ROOT="${DEFAULT_SOURCE_RESULTS_ROOT}"
REFERENCE_RESULTS_ROOT="${DEFAULT_REFERENCE_RESULTS_ROOT}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
COMPARISON=0
SOURCE_RESULTS_ROOT_EXPLICIT=0
OUTPUT_DIR_EXPLICIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --comparison)
            COMPARISON=1
            shift
            ;;
        --source-results-root)
            SOURCE_RESULTS_ROOT="${2:?Missing value for --source-results-root}"
            SOURCE_RESULTS_ROOT_EXPLICIT=1
            shift 2
            ;;
        --reference-results-root)
            REFERENCE_RESULTS_ROOT="${2:?Missing value for --reference-results-root}"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
            OUTPUT_DIR_EXPLICIT=1
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

if [[ "${COMPARISON}" == "1" ]]; then
    if [[ "${SOURCE_RESULTS_ROOT_EXPLICIT}" == "0" ]]; then
        SOURCE_RESULTS_ROOT="${DEFAULT_COMPARISON_SOURCE_RESULTS_ROOT}"
    fi
    if [[ "${OUTPUT_DIR_EXPLICIT}" == "0" ]]; then
        OUTPUT_DIR="${DEFAULT_COMPARISON_OUTPUT_DIR}"
    fi
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build or copy it from ${PROJECT_ROOT}/apptainer/analysis.def first." >&2
    exit 1
fi

if [[ ! -d "${SOURCE_RESULTS_ROOT}" ]]; then
    echo "Source results root does not exist: ${SOURCE_RESULTS_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${REFERENCE_RESULTS_ROOT}" ]]; then
    echo "Reference results root does not exist: ${REFERENCE_RESULTS_ROOT}" >&2
    exit 1
fi

if [[ -n "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/vlm_social
mkdir -p "${OUTPUT_DIR}"

echo "Project root:            ${PROJECT_ROOT}"
echo "Apptainer image:         ${SIF_PATH}"
echo "Comparison mode:         ${COMPARISON}"
echo "Source results root:     ${SOURCE_RESULTS_ROOT}"
echo "Reference results root:  ${REFERENCE_RESULTS_ROOT}"
echo "Output dir:              ${OUTPUT_DIR}"
echo "Model:                   ${MODEL}"
echo "Model path:              ${MODEL_PATH:-<download by name>}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/manipulation_result_similarity.py
    --source-results-root "${SOURCE_RESULTS_ROOT}"
    --reference-results-root "${REFERENCE_RESULTS_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL}"
)

if [[ -n "${MODEL_PATH}" ]]; then
    PYTHON_ARGS+=(--model-path "${MODEL_PATH}")
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
