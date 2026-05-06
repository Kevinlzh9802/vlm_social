#!/bin/bash
#SBATCH --job-name="manip_result_sim"
#SBATCH --time=00:15:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
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
DEFAULT_NO_AUDIO_SOURCE_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio"
DEFAULT_NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio_comparison"
DEFAULT_REFERENCE_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/results"
DEFAULT_GEMINI_REFERENCE_RESULTS_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench/human_eval/gemini"
DEFAULT_OUTPUT_DIR="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/result_similarity_combined"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--source-results-root PATH] [--comparison-source-results-root PATH] [--no-audio-source-results-root PATH] [--no-audio-comparison-source-results-root PATH] [--reference-results-root PATH] [--gemini-reference-results-root PATH] [--skip-gemini-reference] [--output-dir PATH] [--model MODEL] [--model-path PATH]" >&2
    echo "  The script always processes all four corruption conditions together:" >&2
    echo "    with audio + normal corruption      (default: ${DEFAULT_SOURCE_RESULTS_ROOT})" >&2
    echo "    with audio + comparison corruption  (default: ${DEFAULT_COMPARISON_SOURCE_RESULTS_ROOT})" >&2
    echo "    no audio + normal corruption        (default: ${DEFAULT_NO_AUDIO_SOURCE_RESULTS_ROOT})" >&2
    echo "    no audio + comparison corruption    (default: ${DEFAULT_NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT})" >&2
    echo "  reference-results-root: full benchmark result tree (default: ${DEFAULT_REFERENCE_RESULTS_ROOT})" >&2
    echo "  gemini-reference-results-root: full benchmark Gemini result tree from gemini_retrieve_daic.sh (default: ${DEFAULT_GEMINI_REFERENCE_RESULTS_ROOT})" >&2
    echo "  --skip-gemini-reference: do not include Gemini full benchmark reference answers." >&2
    echo "  output-dir: combined HCS/NHCS output directory (default: ${DEFAULT_OUTPUT_DIR})" >&2
    echo "  expected model folders under source/reference roots include qwen2.5, gemma-4-e4b, and gemini when available." >&2
    echo "  model: SentenceTransformer model name (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: optional local SentenceTransformer directory. Recommended on compute nodes." >&2
}

SOURCE_RESULTS_ROOT="${DEFAULT_SOURCE_RESULTS_ROOT}"
COMPARISON_SOURCE_RESULTS_ROOT="${DEFAULT_COMPARISON_SOURCE_RESULTS_ROOT}"
NO_AUDIO_SOURCE_RESULTS_ROOT="${DEFAULT_NO_AUDIO_SOURCE_RESULTS_ROOT}"
NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT="${DEFAULT_NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT}"
REFERENCE_RESULTS_ROOT="${DEFAULT_REFERENCE_RESULTS_ROOT}"
GEMINI_REFERENCE_RESULTS_ROOT="${DEFAULT_GEMINI_REFERENCE_RESULTS_ROOT}"
OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
SKIP_GEMINI_REFERENCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-results-root)
            SOURCE_RESULTS_ROOT="${2:?Missing value for --source-results-root}"
            shift 2
            ;;
        --comparison-source-results-root)
            COMPARISON_SOURCE_RESULTS_ROOT="${2:?Missing value for --comparison-source-results-root}"
            shift 2
            ;;
        --no-audio-source-results-root)
            NO_AUDIO_SOURCE_RESULTS_ROOT="${2:?Missing value for --no-audio-source-results-root}"
            shift 2
            ;;
        --no-audio-comparison-source-results-root)
            NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT="${2:?Missing value for --no-audio-comparison-source-results-root}"
            shift 2
            ;;
        --reference-results-root)
            REFERENCE_RESULTS_ROOT="${2:?Missing value for --reference-results-root}"
            shift 2
            ;;
        --gemini-reference-results-root)
            GEMINI_REFERENCE_RESULTS_ROOT="${2:?Missing value for --gemini-reference-results-root}"
            shift 2
            ;;
        --skip-gemini-reference)
            SKIP_GEMINI_REFERENCE=1
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="${2:?Missing value for --output-dir}"
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
        --comparison|--no-audio)
            echo "[WARN] $1 is ignored. This script now processes all four source variants in one run." >&2
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
    echo "Build or copy it from ${PROJECT_ROOT}/apptainer/analysis.def first." >&2
    exit 1
fi

for SOURCE_ROOT in \
    "${SOURCE_RESULTS_ROOT}" \
    "${COMPARISON_SOURCE_RESULTS_ROOT}" \
    "${NO_AUDIO_SOURCE_RESULTS_ROOT}" \
    "${NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT}"; do
    if [[ ! -d "${SOURCE_ROOT}" ]]; then
        echo "Source results root does not exist: ${SOURCE_ROOT}" >&2
        exit 1
    fi
done

if [[ ! -d "${REFERENCE_RESULTS_ROOT}" ]]; then
    echo "Reference results root does not exist: ${REFERENCE_RESULTS_ROOT}" >&2
    exit 1
fi

if [[ "${SKIP_GEMINI_REFERENCE}" == "0" && ! -d "${GEMINI_REFERENCE_RESULTS_ROOT}" ]]; then
    echo "Gemini reference results root does not exist: ${GEMINI_REFERENCE_RESULTS_ROOT}" >&2
    echo "Run gemini_retrieve_daic.sh for the full benchmark, pass --gemini-reference-results-root, or pass --skip-gemini-reference." >&2
    exit 1
fi

warn_missing_model_roots() {
    local root="$1"
    local label="$2"
    for required_model_root in qwen2.5 gemma-4-e4b gemini; do
        if [[ ! -d "${root}/${required_model_root}" ]]; then
            echo "[WARN] ${label} is missing ${required_model_root}: ${root}/${required_model_root}" >&2
        fi
    done
}

warn_missing_model_roots "${SOURCE_RESULTS_ROOT}" "With-audio source root"
warn_missing_model_roots "${COMPARISON_SOURCE_RESULTS_ROOT}" "With-audio comparison source root"
warn_missing_model_roots "${NO_AUDIO_SOURCE_RESULTS_ROOT}" "No-audio source root"
warn_missing_model_roots "${NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT}" "No-audio comparison source root"
warn_missing_model_roots "${REFERENCE_RESULTS_ROOT}" "Reference root"

if [[ "${SKIP_GEMINI_REFERENCE}" == "0" && ! -d "${GEMINI_REFERENCE_RESULTS_ROOT}/gemini" ]]; then
    echo "[WARN] Gemini reference root is missing gemini/: ${GEMINI_REFERENCE_RESULTS_ROOT}/gemini" >&2
fi

if [[ -n "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/vlm_social
mkdir -p "${OUTPUT_DIR}"

echo "Project root:                        ${PROJECT_ROOT}"
echo "Apptainer image:                     ${SIF_PATH}"
echo "With-audio source root:              ${SOURCE_RESULTS_ROOT}"
echo "With-audio comparison root:          ${COMPARISON_SOURCE_RESULTS_ROOT}"
echo "No-audio source root:                ${NO_AUDIO_SOURCE_RESULTS_ROOT}"
echo "No-audio comparison root:            ${NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT}"
echo "Reference results root:              ${REFERENCE_RESULTS_ROOT}"
echo "Gemini reference root:               ${GEMINI_REFERENCE_RESULTS_ROOT}"
echo "Skip Gemini reference:               ${SKIP_GEMINI_REFERENCE}"
echo "Output dir:                          ${OUTPUT_DIR}"
echo "Model:                               ${MODEL}"
echo "Model path:                          ${MODEL_PATH:-<download by name>}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/manipulation_result_similarity.py
    --source-results-root "${SOURCE_RESULTS_ROOT}"
    --comparison-source-results-root "${COMPARISON_SOURCE_RESULTS_ROOT}"
    --no-audio-source-results-root "${NO_AUDIO_SOURCE_RESULTS_ROOT}"
    --no-audio-comparison-source-results-root "${NO_AUDIO_COMPARISON_SOURCE_RESULTS_ROOT}"
    --reference-results-root "${REFERENCE_RESULTS_ROOT}"
    --output-dir "${OUTPUT_DIR}"
    --model "${MODEL}"
)

if [[ "${SKIP_GEMINI_REFERENCE}" == "0" ]]; then
    PYTHON_ARGS+=(--additional-reference-results-root "${GEMINI_REFERENCE_RESULTS_ROOT}")
fi

if [[ -n "${MODEL_PATH}" ]]; then
    PYTHON_ARGS+=(--model-path "${MODEL_PATH}")
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
