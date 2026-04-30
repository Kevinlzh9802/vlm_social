#!/bin/bash
#SBATCH --job-name="human_eval_sel"
#SBATCH --time=03:59:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=logs/select_human_eval_samples_daic_%j.out
#SBATCH --error=logs/select_human_eval_samples_daic_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/gesbench.sif"
DEFAULT_DATA_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}"
DEFAULT_OUTPUT_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}/human_eval/samples"
DEFAULT_SEED="42"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [output_root] [seed]" >&2
    echo "  sbatch $0 [--output-root PATH] [--seed N] [--data-root PATH] [--no-overwrite]" >&2
    echo "  data-root: dataset parent folder containing mintrec2, meld, and seamless_interaction" >&2
    echo "  output-root: destination folder for copied batches, zip files, and summaries" >&2
    echo "  seed: random seed for reproducible sampling" >&2
    echo "  default output-root: ${DEFAULT_OUTPUT_ROOT}" >&2
}

DATA_ROOT="${DEFAULT_DATA_ROOT}"
OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}"
SEED="${DEFAULT_SEED}"
OVERWRITE=1
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="${2:?Missing value for --data-root}"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="${2:?Missing value for --output-root}"
            shift 2
            ;;
        --seed)
            SEED="${2:?Missing value for --seed}"
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
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL_ARGS[@]} -gt 0 ]]; then
    OUTPUT_ROOT="${POSITIONAL_ARGS[0]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 1 ]]; then
    SEED="${POSITIONAL_ARGS[1]}"
fi

if [[ ${#POSITIONAL_ARGS[@]} -gt 2 ]]; then
    usage
    echo "Too many positional arguments." >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build gesbench.sif gesbench.def" >&2
    exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Data root does not exist: ${DATA_ROOT}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ROOT}")"

echo "Project root: ${PROJECT_ROOT}"
echo "Apptainer image: ${SIF_PATH}"
echo "Data root: ${DATA_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Seed: ${SEED}"
echo "Overwrite output root: ${OVERWRITE}"

PYTHON_ARGS=(
    python /workspace/dataset/select_human_eval_samples.py
    --data-root "${DATA_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --seed "${SEED}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
    PYTHON_ARGS+=(--overwrite)
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
