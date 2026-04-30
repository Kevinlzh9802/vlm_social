#!/bin/bash
#SBATCH --job-name="seamless_utt"
#SBATCH --time=12:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/seamless_utterance_%j.out
#SBATCH --error=logs/seamless_utterance_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export PROJECT_ROOT=/path/to/vlm_social DATA_ROOT=/path/to/data/gestalt_bench APPTAINER_ROOT=/path/to/apptainers

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
SIF_PATH="${APPTAINER_ROOT:-/path/to/apptainers}/vlm_social.sif"
GROUPED_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}/seamless_interaction/grouped_interaction"
OUT_ROOT="${DATA_ROOT:-/path/to/data/gestalt_bench}/seamless_interaction/utterance_level"

MODE="transcript"
PADDING="0.0"
MIN_DURATION="0.05"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build it first with: cd ${PROJECT_ROOT}/apptainer && apptainer build vlm_social.sif vlm_social.def" >&2
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
