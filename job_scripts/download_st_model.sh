#!/bin/bash
# Download a SentenceTransformer model to a local directory.
# Run this on a login node (which has internet) BEFORE submitting the SLURM job.
#
# Usage:
#   bash job_scripts/download_st_model.sh [model_name] [output_dir]
#
# Defaults:
#   model_name = all-MiniLM-L6-v2
#   output_dir = /scratch/zli33/models/<model_name>

set -euo pipefail

SIF_PATH="/scratch/zli33/apptainers/analysis.sif"

MODEL_NAME="${1:-all-MiniLM-L6-v2}"
OUTPUT_DIR="${2:-/scratch/zli33/models/${MODEL_NAME}}"

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Apptainer image not found: ${SIF_PATH}" >&2
    echo "Build it first: cd apptainer && apptainer build analysis.sif analysis.def" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Downloading model '${MODEL_NAME}' → ${OUTPUT_DIR}"

apptainer exec \
    --bind /scratch/zli33:/scratch/zli33 \
    "${SIF_PATH}" \
    python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('${MODEL_NAME}')
model.save('${OUTPUT_DIR}')
print('Saved to ${OUTPUT_DIR}')
"

echo "Done. You can now submit the job:"
echo "  sbatch job_scripts/analysis_delftblue.sh --model-path ${OUTPUT_DIR}"
