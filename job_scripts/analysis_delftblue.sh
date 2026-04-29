#!/bin/bash
#SBATCH --job-name="analysis"
#SBATCH --time=02:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/analysis_delftblue_%j.out
#SBATCH --error=logs/analysis_delftblue_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/analysis.sif"
DEFAULT_RESULTS_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}/results"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH="${MODEL_ROOT:-/path/to/models}/all-MiniLM-L6-v2"
DEFAULT_THRESHOLDS=("0.3" "0.5" "0.7" "0.9")

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--results-root PATH] [--model MODEL_NAME] [--model-path PATH] [--turnover-thresholds T1 T2 ...] [--with-scatter]" >&2
    echo "  results-root: path to the parent results folder (default: ${DEFAULT_RESULTS_ROOT})" >&2
    echo "  model: SentenceTransformer model name, used only for download (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: local directory with pre-downloaded model (default: ${DEFAULT_MODEL_PATH})" >&2
    echo "  turnover-thresholds: semantic-turnover thresholds (default: ${DEFAULT_THRESHOLDS[*]})" >&2
    echo "  --with-scatter: include scatter points in per-folder clip-to-final plots (default: disabled)" >&2
    echo "" >&2
    echo "Pre-download the model on a login node before submitting:" >&2
    echo "  apptainer exec --bind ${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench} ${SIF_PATH} \\" >&2
    echo "    python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('${DEFAULT_MODEL}').save('${DEFAULT_MODEL_PATH}')\"" >&2
}

RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
THRESHOLDS=("${DEFAULT_THRESHOLDS[@]}")
NO_SCATTER=1

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
        --turnover-thresholds)
            THRESHOLDS=()
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                THRESHOLDS+=("$1")
                shift
            done
            if [[ ${#THRESHOLDS[@]} -eq 0 ]]; then
                echo "Missing value(s) for --turnover-thresholds" >&2
                exit 1
            fi
            ;;
        --with-scatter)
            NO_SCATTER=0
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
    echo "  apptainer exec --bind ${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench} ${SIF_PATH} \\" >&2
    echo "    python -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL}').save('${MODEL_PATH}')\"" >&2
    exit 1
fi

mkdir -p "$(dirname "${RESULTS_ROOT}")"

echo "Project root:    ${PROJECT_ROOT}"
echo "Apptainer image: ${SIF_PATH}"
echo "Results root:    ${RESULTS_ROOT}"
echo "Model name:      ${MODEL}"
echo "Model path:      ${MODEL_PATH}"
echo "Thresholds:      ${THRESHOLDS[*]}"
echo "No scatter:      ${NO_SCATTER}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/main.py
    --results-root "${RESULTS_ROOT}"
    --model-path "${MODEL_PATH}"
    --turnover-thresholds "${THRESHOLDS[@]}"
)

if [[ "${NO_SCATTER}" == "1" ]]; then
    PYTHON_ARGS+=(--no-scatter)
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    --bind "${RESULTS_ROOT}:${RESULTS_ROOT}" \
    --bind "${MODEL_PATH}:${MODEL_PATH}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
