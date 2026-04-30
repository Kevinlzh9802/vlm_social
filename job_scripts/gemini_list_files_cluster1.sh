#!/bin/bash
#SBATCH --job-name="gemini-list-files"
#SBATCH --partition=<partition>
#SBATCH --time=18:00:00
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END
#SBATCH --output=logs/gemini_list_files_<cluster1>_%j.out
#SBATCH --error=logs/gemini_list_files_<cluster1>_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export APPTAINER_ROOT=/path/to/apptainers API_KEY_FILE=/path/to/api_key.txt

# List and delete all files uploaded to the Gemini Files API.
#
# Usage:
#   sbatch job_scripts/gemini_list_files_<cluster1>.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="${SLURM_SUBMIT_DIR:-.}"
sif_file=${APPTAINER_ROOT:-/path/to/apptainers}/gemini.sif
api_key_file=${API_KEY_FILE:-/path/to/api_key.txt}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2; exit 1
fi

if [ ! -f "$api_key_file" ]; then
    echo "[ERROR] API key file not found: $api_key_file" >&2; exit 1
fi

mkdir -p logs/gemini-batch

# ---------------------------------------------------------------------------
echo "[INFO] project_dir  = $project_dir"
echo "[INFO] api_key_file = $api_key_file"
echo ""

# Pipe "y" to auto-confirm the --delete-all prompt
echo "y" | apptainer exec \
    --bind "$project_dir":/workspace \
    --bind "$api_key_file":"$api_key_file":ro \
    --pwd /workspace \
    "$sif_file" \
    python "api_models/gemini_list_files.py" \
        --api-key-path "$api_key_file" \
        --count-only \
        --delete-all

echo ""
echo "[INFO] Done."
