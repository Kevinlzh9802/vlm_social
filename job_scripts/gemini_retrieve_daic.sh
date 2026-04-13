#!/bin/bash
#SBATCH --job-name="gemini-retrieve"
#SBATCH --partition=insy,general
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gemini-batch/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gemini-batch/slurm_%j.err

# Scan the shared Gemini batch registry, download results for all completed
# jobs, and mark them as retrieved.  No dataset / prompt / utt parameters
# needed — everything is stored in the registry.
#
# Usage:
#   sbatch job_scripts/gemini_retrieve_daic.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="${SLURM_SUBMIT_DIR:-.}"
sif_file=/tudelft.net/staff-umbrella/neon/apptainer/gemini.sif
# gestalt_root=/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench
# results_root=/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench

gestalt_root=/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/human_eval/samples
results_root=/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench/human_eval

api_key_file=/home/nfs/zli33/keys/gemini_api.txt
registry_file="${results_root}/gemini/gemini_registry.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2; exit 1
fi

if [ ! -f "$api_key_file" ]; then
    echo "[ERROR] API key file not found: $api_key_file" >&2; exit 1
fi

if [ ! -f "$registry_file" ]; then
    echo "[ERROR] Registry file not found: $registry_file" >&2
    echo "  Submit a job first with gemini_batch_daic.sh." >&2
    exit 1
fi

# Ensure slurm log directory exists
mkdir -p /home/nfs/zli33/slurm_outputs/gemini-batch

# ---------------------------------------------------------------------------
echo "[INFO] project_dir  = $project_dir"
echo "[INFO] registry     = $registry_file"
echo ""

apptainer exec \
    --bind "$project_dir":/workspace \
    --bind "${results_root}":"${results_root}" \
    --bind "${gestalt_root}":"${gestalt_root}" \
    --bind "$api_key_file":"$api_key_file":ro \
    --pwd /workspace \
    "$sif_file" \
    python "api_models/gemini_retrieve.py" \
        --registry "$registry_file" \
        --api-key-path "$api_key_file"

echo ""
echo "[INFO] Retrieval pass complete."
