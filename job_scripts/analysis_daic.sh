#!/bin/bash
#SBATCH --job-name="analysis"
#SBATCH --time=00:04:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/analysis.sif"
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_RESULTS_ROOT="${DEFAULT_DATA_ROOT}/results"
DEFAULT_GEMINI_RESULTS_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench/human_eval/gemini"
DEFAULT_GEMMA_RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}/gemma-4-e4b"
DEFAULT_HUMAN_ANNOTATION_SUMMARY_CSV="${DEFAULT_DATA_ROOT}/human_eval/task1/plot_data/partial_to_full_percentiles.csv"
DEFAULT_HUMAN_ANNOTATION_POINTS_CSV="${DEFAULT_DATA_ROOT}/human_eval/task1/plot_data/partial_to_full_points.csv"
DEFAULT_PLOT_DATA_DIR="${DEFAULT_DATA_ROOT}/plots/plot_data"
DEFAULT_PLOT_DATA_JSON="${DEFAULT_PLOT_DATA_DIR}/analysis_plot_data.json"
DEFAULT_MODEL="all-MiniLM-L6-v2"
DEFAULT_MODEL_PATH=""
DEFAULT_THRESHOLDS=("0.3" "0.5" "0.7" "0.9")

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--results-root PATH] [--gemini-results-root PATH] [--gemma-results-root PATH] [--skip-gemini] [--skip-gemma] [--human-annotation-summary-csv PATH] [--skip-human-overlay] [--save-plot-data] [--plot-data-dir PATH] [--from-plot-data [PATH]] [--model MODEL_NAME] [--model-path PATH] [--turnover-thresholds T1 T2 ...] [--progress-partitions N] [--with-scatter]" >&2
    echo "  results-root: path to the parent results folder (default: ${DEFAULT_RESULTS_ROOT})" >&2
    echo "  gemini-results-root: Gemini result tree from gemini_retrieve_daic.sh (default: ${DEFAULT_GEMINI_RESULTS_ROOT})" >&2
    echo "  gemma-results-root: Gemma 4 result tree from gemma_daic.sh (default: ${DEFAULT_GEMMA_RESULTS_ROOT})" >&2
    echo "  --skip-gemini: do not include Gemini partial-to-full similarity lines" >&2
    echo "  --skip-gemma: do not include Gemma 4 results when --results-root does not already include them" >&2
    echo "  human-annotation-summary-csv: partial_to_full_percentiles.csv from human_annotation_similarity.py (default: ${DEFAULT_HUMAN_ANNOTATION_SUMMARY_CSV}); the sibling partial_to_full_points.csv is also required for human ST overlay" >&2
    echo "  --skip-human-overlay: generate model-only aggregate plots without human annotation overlays" >&2
    echo "  --save-plot-data: save numeric plot data after embedding so plots can be regenerated without re-embedding" >&2
    echo "  plot-data-dir: output folder for --save-plot-data (default: ${DEFAULT_PLOT_DATA_DIR})" >&2
    echo "  from-plot-data: regenerate aggregate plots from a saved analysis_plot_data.json and skip embedding (default cache path: ${DEFAULT_PLOT_DATA_JSON})" >&2
    echo "  model: SentenceTransformer model name, used when --model-path is not set (default: ${DEFAULT_MODEL})" >&2
    echo "  model-path: optional local directory with pre-downloaded SentenceTransformer model" >&2
    echo "  turnover-thresholds: semantic-turnover thresholds (default: ${DEFAULT_THRESHOLDS[*]})" >&2
    echo "  progress-partitions: number of clip-progress bins (default: 20)" >&2
    echo "  --with-scatter: include scatter points in per-folder clip-to-final plots (default: disabled)" >&2
    echo "" >&2
    echo "If the human summary CSV is missing, run:" >&2
    echo "  sbatch job_scripts/human_annotation_similarity_daic.sh" >&2
}

RESULTS_ROOT="${DEFAULT_RESULTS_ROOT}"
GEMINI_RESULTS_ROOT="${DEFAULT_GEMINI_RESULTS_ROOT}"
GEMMA_RESULTS_ROOT="${DEFAULT_GEMMA_RESULTS_ROOT}"
HUMAN_ANNOTATION_SUMMARY_CSV="${DEFAULT_HUMAN_ANNOTATION_SUMMARY_CSV}"
PLOT_DATA_DIR="${DEFAULT_PLOT_DATA_DIR}"
FROM_PLOT_DATA=""
MODEL="${DEFAULT_MODEL}"
MODEL_PATH="${DEFAULT_MODEL_PATH}"
THRESHOLDS=("${DEFAULT_THRESHOLDS[@]}")
PROGRESS_PARTITIONS="20"
NO_SCATTER=1
SKIP_HUMAN_OVERLAY=0
SKIP_GEMINI=0
SKIP_GEMMA=0
SAVE_PLOT_DATA=0

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
        --gemma-results-root)
            GEMMA_RESULTS_ROOT="${2:?Missing value for --gemma-results-root}"
            shift 2
            ;;
        --skip-gemini)
            SKIP_GEMINI=1
            shift
            ;;
        --skip-gemma)
            SKIP_GEMMA=1
            shift
            ;;
        --human-annotation-summary-csv)
            HUMAN_ANNOTATION_SUMMARY_CSV="${2:?Missing value for --human-annotation-summary-csv}"
            shift 2
            ;;
        --skip-human-overlay)
            SKIP_HUMAN_OVERLAY=1
            shift
            ;;
        --save-plot-data)
            SAVE_PLOT_DATA=1
            shift
            ;;
        --plot-data-dir)
            PLOT_DATA_DIR="${2:?Missing value for --plot-data-dir}"
            shift 2
            ;;
        --from-plot-data)
            if [[ $# -ge 2 && "$2" != --* ]]; then
                FROM_PLOT_DATA="$2"
                shift 2
            else
                FROM_PLOT_DATA="${DEFAULT_PLOT_DATA_JSON}"
                shift
            fi
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
        --progress-partitions)
            PROGRESS_PARTITIONS="${2:?Missing value for --progress-partitions}"
            shift 2
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

if [[ -n "${FROM_PLOT_DATA}" && ! -f "${FROM_PLOT_DATA}" ]]; then
    echo "Plot data JSON does not exist: ${FROM_PLOT_DATA}" >&2
    exit 1
fi

if [[ -z "${FROM_PLOT_DATA}" && ! -d "${RESULTS_ROOT}" ]]; then
    echo "Results root does not exist: ${RESULTS_ROOT}" >&2
    exit 1
fi

if [[ -z "${FROM_PLOT_DATA}" && "${SKIP_GEMINI}" == "0" && ! -d "${GEMINI_RESULTS_ROOT}" ]]; then
    echo "Gemini results root does not exist: ${GEMINI_RESULTS_ROOT}" >&2
    echo "Run gemini_retrieve_daic.sh after Gemini jobs complete, pass --gemini-results-root, or pass --skip-gemini." >&2
    exit 1
fi

if [[ -z "${FROM_PLOT_DATA}" && "${SKIP_HUMAN_OVERLAY}" == "0" && ! -f "${HUMAN_ANNOTATION_SUMMARY_CSV}" ]]; then
    echo "Human annotation summary CSV does not exist: ${HUMAN_ANNOTATION_SUMMARY_CSV}" >&2
    echo "Run human_annotation_similarity_daic.sh first or pass --skip-human-overlay." >&2
    exit 1
fi

if [[ -z "${FROM_PLOT_DATA}" && "${SKIP_GEMMA}" == "0" && ! -d "${GEMMA_RESULTS_ROOT}" ]]; then
    echo "Gemma results root does not exist: ${GEMMA_RESULTS_ROOT}" >&2
    echo "Run gemma_daic.sh first, pass --gemma-results-root, or pass --skip-gemma." >&2
    exit 1
fi

HUMAN_ANNOTATION_POINTS_CSV="$(dirname "${HUMAN_ANNOTATION_SUMMARY_CSV}")/partial_to_full_points.csv"
if [[ -z "${FROM_PLOT_DATA}" && "${SKIP_HUMAN_OVERLAY}" == "0" && ! -f "${HUMAN_ANNOTATION_POINTS_CSV}" ]]; then
    echo "Human annotation point CSV does not exist: ${HUMAN_ANNOTATION_POINTS_CSV}" >&2
    echo "Run human_annotation_similarity_daic.sh first or pass --skip-human-overlay." >&2
    exit 1
fi

if [[ -z "${FROM_PLOT_DATA}" && "${SKIP_HUMAN_OVERLAY}" == "0" ]]; then
    IFS= read -r HUMAN_POINTS_HEADER < "${HUMAN_ANNOTATION_POINTS_CSV}" || true
    HUMAN_POINTS_HEADER="${HUMAN_POINTS_HEADER%$'\r'}"
    if [[ ",${HUMAN_POINTS_HEADER}," != *",neighboring_similarity_to_next,"* ]]; then
        echo "Human annotation point CSV is missing neighboring_similarity_to_next: ${HUMAN_ANNOTATION_POINTS_CSV}" >&2
        echo "Rerun human_annotation_similarity_daic.sh so analysis can overlay human ST curves, or pass --skip-human-overlay." >&2
        exit 1
    fi
fi

if [[ -z "${FROM_PLOT_DATA}" && -n "${MODEL_PATH}" && ! -d "${MODEL_PATH}" ]]; then
    echo "Model path does not exist: ${MODEL_PATH}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/vlm_social
if [[ "${SAVE_PLOT_DATA}" == "1" ]]; then
    mkdir -p "${PLOT_DATA_DIR}"
fi

echo "Project root:                  ${PROJECT_ROOT}"
echo "Apptainer image:               ${SIF_PATH}"
echo "Results root:                  ${RESULTS_ROOT}"
echo "Gemini results root:           ${GEMINI_RESULTS_ROOT}"
echo "Gemma results root:            ${GEMMA_RESULTS_ROOT}"
echo "Skip Gemini:                   ${SKIP_GEMINI}"
echo "Skip Gemma:                    ${SKIP_GEMMA}"
echo "Human annotation summary CSV:  ${HUMAN_ANNOTATION_SUMMARY_CSV}"
echo "Human annotation points CSV:   ${HUMAN_ANNOTATION_POINTS_CSV:-${DEFAULT_HUMAN_ANNOTATION_POINTS_CSV}}"
echo "Skip human overlay:            ${SKIP_HUMAN_OVERLAY}"
echo "Save plot data:                ${SAVE_PLOT_DATA}"
echo "Plot data dir:                 ${PLOT_DATA_DIR}"
echo "From plot data:                ${FROM_PLOT_DATA:-<disabled>}"
echo "Model name:                    ${MODEL}"
echo "Model path:                    ${MODEL_PATH:-<download by name>}"
echo "Thresholds:                    ${THRESHOLDS[*]}"
echo "Progress partitions:           ${PROGRESS_PARTITIONS}"
echo "No scatter:                    ${NO_SCATTER}"

if [[ -n "${FROM_PLOT_DATA}" ]]; then
    PYTHON_ARGS=(
        python /workspace/experiments/analysis/main.py
        --from-plot-data "${FROM_PLOT_DATA}"
    )
else
    PYTHON_ARGS=(
        python /workspace/experiments/analysis/main.py
        --results-root "${RESULTS_ROOT}"
        --model "${MODEL}"
        --turnover-thresholds "${THRESHOLDS[@]}"
        --progress-partitions "${PROGRESS_PARTITIONS}"
    )

    if [[ "${SKIP_GEMINI}" == "0" ]]; then
        PYTHON_ARGS+=(--additional-results-root "${GEMINI_RESULTS_ROOT}")
    fi

    if [[ "${SKIP_GEMMA}" == "0" ]]; then
        RESULTS_ROOT_REAL="$(readlink -f "${RESULTS_ROOT}")"
        GEMMA_RESULTS_ROOT_REAL="$(readlink -f "${GEMMA_RESULTS_ROOT}")"
        RESULTS_ROOT_GEMMA_CHILD_REAL="$(readlink -f "${RESULTS_ROOT}/gemma-4-e4b" 2>/dev/null || true)"
        if [[ "${RESULTS_ROOT_REAL}" != "${GEMMA_RESULTS_ROOT_REAL}" && "${RESULTS_ROOT_GEMMA_CHILD_REAL}" != "${GEMMA_RESULTS_ROOT_REAL}" ]]; then
            PYTHON_ARGS+=(--additional-results-root "${GEMMA_RESULTS_ROOT}")
        fi
    fi

    if [[ -n "${MODEL_PATH}" ]]; then
        PYTHON_ARGS+=(--model-path "${MODEL_PATH}")
    fi

    if [[ "${SKIP_HUMAN_OVERLAY}" == "0" ]]; then
        PYTHON_ARGS+=(--human-annotation-summary-csv "${HUMAN_ANNOTATION_SUMMARY_CSV}")
    fi

    if [[ "${SAVE_PLOT_DATA}" == "1" ]]; then
        PYTHON_ARGS+=(--save-plot-data --plot-data-dir "${PLOT_DATA_DIR}")
    fi

    if [[ "${NO_SCATTER}" == "1" ]]; then
        PYTHON_ARGS+=(--no-scatter)
    fi
fi

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /home/nfs/zli33:/home/nfs/zli33 \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
