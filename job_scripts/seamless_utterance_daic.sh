#!/bin/bash
#SBATCH --job-name="seamless_utt"
#SBATCH --time=18:00:00
#SBATCH --qos=medium         # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --partition=insy,general  # Request partition. Default is 'general'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/vlm_social.sif"
GROUPED_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/seamless_interaction/grouped_interaction"
OUT_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/seamless_interaction/utterance_level"

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
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    python /workspace/dataset/seamless_construct_utterance.py \
    --grouped-root "${GROUPED_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --mode "${MODE}" \
    --padding "${PADDING}" \
    --min-duration "${MIN_DURATION}"