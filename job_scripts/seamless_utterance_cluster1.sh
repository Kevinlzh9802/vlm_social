#!/bin/bash
#SBATCH --job-name="seamless_utt"
#SBATCH --time=18:00:00
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --partition=<partition>  # Request partition. Default is 'general'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=logs/seamless_utterance_<cluster1>_%j.out
#SBATCH --error=logs/seamless_utterance_<cluster1>_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/gesbench DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/gesbench.sif"
GROUPED_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}/seamless_interaction/grouped_interaction"
OUT_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}/seamless_interaction/utterance_level"

MODE="transcript"
PADDING="0.0"
MIN_DURATION="0.05"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build gesbench.sif gesbench.def" >&2
    exit 1
fi

if [[ ! -d "${GROUPED_ROOT}" ]]; then
    echo "Grouped interaction directory does not exist: ${GROUPED_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}"

echo "Project root:  ${PROJECT_ROOT}"
echo "Apptainer image: ${SIF_PATH}"
echo "Grouped root:  ${GROUPED_ROOT}"
echo "Output root:   ${OUT_ROOT}"
echo "Mode:          ${MODE}"
echo "Padding:       ${PADDING}"
echo "Min duration:  ${MIN_DURATION}"

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_ROOT:-/path/to/data/gestalt_bench}:${DATA_ROOT:-/path/to/data/gestalt_bench}" \
    "${SIF_PATH}" \
    python /workspace/dataset/seamless_construct_utterance.py \
    --grouped-root "${GROUPED_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --mode "${MODE}" \
    --padding "${PADDING}" \
    --min-duration "${MIN_DURATION}"