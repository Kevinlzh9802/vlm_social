#!/bin/bash
#SBATCH --job-name="gemini-retrieve"
#SBATCH --partition=insy,general
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=logs/gemini_retrieve_daic_%j.out
#SBATCH --error=logs/gemini_retrieve_daic_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.
# User paths to set: export DATA_ROOT=/path/to/data/gestalt_bench RESULTS_ROOT=/path/to/results/gestalt_bench APPTAINER_ROOT=/path/to/apptainers API_KEY_FILE=/path/to/api_key.txt

# Scan the shared Gemini batch registry, download results for all completed
# jobs, and mark them as retrieved.  No dataset / prompt / utt parameters
# needed — everything is stored in the registry.
#
# Usage:
#   sbatch job_scripts/gemini_retrieve_daic.sh
#   sbatch job_scripts/gemini_retrieve_daic.sh --annotated
#   sbatch job_scripts/gemini_retrieve_daic.sh --annotated --comparison
#   sbatch job_scripts/gemini_retrieve_daic.sh --annotated --no-audio

set -euo pipefail

annotated=0
comparison=0
no_audio=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --annotated)
            annotated=1
            shift
            ;;
        --comparison)
            comparison=1
            shift
            ;;
        --no-audio)
            no_audio=1
            shift
            ;;
        -h|--help)
            echo "Usage: sbatch job_scripts/gemini_retrieve_daic.sh [--annotated] [--comparison] [--no-audio]" >&2
            echo "  --comparison is only valid together with --annotated." >&2
            echo "  --no-audio is only valid together with --annotated and selects no-audio result roots." >&2
            exit 0
            ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
        *)
            echo "[ERROR] Unexpected positional argument: $1" >&2
            exit 1
            ;;
    esac
done

if [ "$comparison" = "1" ] && [ "$annotated" = "0" ]; then
    echo "[ERROR] --comparison is only supported together with --annotated" >&2
    exit 1
fi
if [ "$no_audio" = "1" ] && [ "$annotated" = "0" ]; then
    echo "[ERROR] --no-audio is only supported together with --annotated" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="${SLURM_SUBMIT_DIR:-.}"
sif_file=${APPTAINER_ROOT:-/path/to/apptainers}/gemini.sif

gestalt_data_root=${DATA_ROOT:-/path/to/data/gestalt_bench}
default_results_root=${RESULTS_ROOT:-/path/to/results/gestalt_bench}/human_eval

if [ "$annotated" = "1" ]; then
    if [ "$comparison" = "1" ]; then
        gestalt_root="${gestalt_data_root}/human_eval/task2/manipulation_full/data_comparison"
        if [ "$no_audio" = "1" ]; then
            output_root="${gestalt_data_root}/human_eval/task2/manipulation_full/results_noaudio_comparison/gemini"
        else
            output_root="${gestalt_data_root}/human_eval/task2/manipulation_full/results_comparison/gemini"
        fi
    else
        gestalt_root="${gestalt_data_root}/human_eval/task2/manipulation_full/data"
        if [ "$no_audio" = "1" ]; then
            output_root="${gestalt_data_root}/human_eval/task2/manipulation_full/results_noaudio/gemini"
        else
            output_root="${gestalt_data_root}/human_eval/task2/manipulation_full/results/gemini"
        fi
    fi
else
    gestalt_root="${gestalt_data_root}/human_eval/samples"
    output_root="${default_results_root}/gemini"
fi

api_key_file=${API_KEY_FILE:-/path/to/api_key.txt}
registry_file="${output_root}/gemini_registry.json"

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
mkdir -p logs/gemini-batch

# ---------------------------------------------------------------------------
echo "[INFO] project_dir  = $project_dir"
echo "[INFO] annotated    = $annotated"
echo "[INFO] comparison   = $comparison"
echo "[INFO] no_audio     = $no_audio"
echo "[INFO] registry     = $registry_file"
echo ""

apptainer exec \
    --bind "$project_dir":/workspace \
    --bind "${output_root}":"${output_root}" \
    --bind "${gestalt_root}":"${gestalt_root}" \
    --bind "$api_key_file":"$api_key_file":ro \
    --pwd /workspace \
    "$sif_file" \
    python "api_models/gemini_retrieve.py" \
        --registry "$registry_file" \
        --api-key-path "$api_key_file"

echo ""
echo "[INFO] Retrieval pass complete."
