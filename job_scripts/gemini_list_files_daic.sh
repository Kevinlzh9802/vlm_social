#!/bin/bash
#SBATCH --job-name="gemini-list-files"
#SBATCH --partition=insy,general
#SBATCH --time=18:00:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gemini-batch/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gemini-batch/slurm_%j.err

# List and delete all files uploaded to the Gemini Files API.
#
# Usage:
#   sbatch job_scripts/gemini_list_files_daic.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="${SLURM_SUBMIT_DIR:-.}"
sif_file=/tudelft.net/staff-umbrella/neon/apptainer/gemini.sif
api_key_file=/home/nfs/zli33/keys/gemini_api.txt

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2; exit 1
fi

if [ ! -f "$api_key_file" ]; then
    echo "[ERROR] API key file not found: $api_key_file" >&2; exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/gemini-batch

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
        --delete-all

echo ""
echo "[INFO] Done."
