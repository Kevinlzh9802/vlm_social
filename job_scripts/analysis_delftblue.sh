#!/bin/bash
#SBATCH --job-name="analysis"
#SBATCH --time=02:00:00
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
SIF_PATH="/scratch/zli33/apptainers/analysis.sif"
DEFAULT_RESULTS_ROOT="/scratch/zli33/data/gestalt_bench/results/qwen2.5"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH="/scratch/zli33/models/all-MiniLM-L6-v2"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--results-root PATH] [--model MODEL_NAME] [--model-path PATH]" >&2
    echo "  results-root: path to results/qwen2.5 (default: ${DEFAULT_RESULTS_ROOT})" >&2
    echo "  model: SentenceTransformer model name, used only for download (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: local directory with pre-downloaded model (default: ${DEFAULT_MODEL_PATH})" >&2
    echo "" >&2
    echo "Pre-download the model on a login node before submitting:" >&2
    echo "  apptainer exec --bind /scratch/zli33:/scratch/zli33 ${SIF_PATH} \\" >&2
    echo "    python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('${DEFAULT_MODEL}').save('${DEFAULT_MODEL_PATH}')\"" >&2
}

RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-root)
            RESULTS_ROOT="${2:?Missing value for --results-root}"
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

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build analysis.sif analysis.def" >&2
    exit 1
fi

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Results root does not exist: ${RESULTS_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Pre-downloaded model not found: ${MODEL_PATH}" >&2
    echo "Download it on a login node first (network is unavailable on compute nodes):" >&2
    echo "  apptainer exec --bind /scratch/zli33:/scratch/zli33 ${SIF_PATH} \\" >&2
    echo "    python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL}').save('${MODEL_PATH}')\"" >&2
    exit 1
fi

mkdir -p "$(dirname "${RESULTS_ROOT}")"

echo "Project root:    ${PROJECT_ROOT}"
echo "Apptainer image: ${SIF_PATH}"
echo "Results root:    ${RESULTS_ROOT}"
echo "Model name:      ${MODEL}"
echo "Model path:      ${MODEL_PATH}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/main.py
    --results-root "${RESULTS_ROOT}"
    --model-path "${MODEL_PATH}"
)

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /home/zli33:/home/zli33 \
    --bind /scratch/zli33:/scratch/zli33 \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
