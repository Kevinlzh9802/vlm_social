#!/bin/bash
#SBATCH --job-name="gemini-submit"
#SBATCH --partition=insy,general
#SBATCH --time=06:00:00
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gemini-batch/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gemini-batch/slurm_%j.err

# Submit a Gemini Batch API job (upload-only, exits immediately).
# Use gemini_retrieve_daic.sh to collect results afterwards.
#
# Examples:
#   sbatch job_scripts/gemini_batch_daic.sh --dataset mintrec2 --utt 1 --batch 1 --prompt intention
#   sbatch job_scripts/gemini_batch_daic.sh --dataset mintrec2 --utt 2 --batch 3 --prompt affordance --gemini-mode 2.5-flash

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
dataset_name=""
prompt_choice=""
utt_count=""
batch_number=""
gemini_mode="2.5-flash"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
    case "$1" in
        -set|--set|--dataset)
            if [ "$#" -lt 2 ]; then echo "[ERROR] Missing value for $1" >&2; exit 1; fi
            dataset_name="$2"; shift 2 ;;
        -prompt|--prompt)
            if [ "$#" -lt 2 ]; then echo "[ERROR] Missing value for $1" >&2; exit 1; fi
            prompt_choice="$2"; shift 2 ;;
        --utt)
            if [ "$#" -lt 2 ]; then echo "[ERROR] Missing value for $1" >&2; exit 1; fi
            utt_count="$2"; shift 2 ;;
        --batch)
            if [ "$#" -lt 2 ]; then echo "[ERROR] Missing value for $1" >&2; exit 1; fi
            batch_number="$2"; shift 2 ;;
        --gemini-mode)
            if [ "$#" -lt 2 ]; then echo "[ERROR] Missing value for $1" >&2; exit 1; fi
            gemini_mode="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: sbatch job_scripts/gemini_batch_daic.sh --dataset <dataset> --utt <1|2|3> --batch <number> --prompt <prompt_choice> [--gemini-mode <mode>] [--poll-interval <seconds>]" >&2
            exit 0 ;;
        -*)
            echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
        *)
            echo "[ERROR] Unexpected positional argument: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Required-argument checks
# ---------------------------------------------------------------------------
usage_msg="Usage: sbatch job_scripts/gemini_batch_daic.sh --dataset <dataset> --utt <1|2|3> --batch <number> --prompt <prompt_choice> [--gemini-mode <mode>] [--poll-interval <seconds>]"

if [ -z "$dataset_name" ]; then echo "[ERROR] --dataset is required" >&2; echo "$usage_msg" >&2; exit 1; fi
if [ -z "$prompt_choice" ]; then echo "[ERROR] --prompt is required" >&2; echo "$usage_msg" >&2; exit 1; fi
if [ -z "$utt_count" ];     then echo "[ERROR] --utt is required"    >&2; echo "$usage_msg" >&2; exit 1; fi
if [ -z "$batch_number" ];  then echo "[ERROR] --batch is required"  >&2; echo "$usage_msg" >&2; exit 1; fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
case "$prompt_choice" in
    intention|affordance) ;;
    *)
        echo "[WARN] Unexpected prompt choice: $prompt_choice (expected 'intention' or 'affordance'). Continuing; prompts.json validation will decide." >&2 ;;
esac

case "$utt_count" in
    1|2|3) ;;
    *)
        echo "[ERROR] Invalid utt count: $utt_count (expected 1, 2, or 3)" >&2; exit 1 ;;
esac

case "$batch_number" in
    ''|*[!0-9]*)
        echo "[ERROR] Invalid batch number: $batch_number (expected a positive integer)" >&2; exit 1 ;;
esac

batch_id=$(printf "%02d" "$batch_number")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
project_dir="${SLURM_SUBMIT_DIR:-.}"
sif_file=/tudelft.net/staff-umbrella/neon/apptainer/gemini.sif
# gestalt_root=/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench
# results_root=/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench

gestalt_root=/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench/human_eval/samples
results_root=/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench/human_eval

data_parent="${gestalt_root}/${dataset_name}/context/${utt_count}-utt_group/batch${batch_id}"
output_dir="${results_root}/gemini/${dataset_name}/context/${utt_count}-utt_group/${gemini_mode}_${prompt_choice}_single-turn"
output_json="${output_dir}/batch${batch_id}.json"

inference_script="api_models/gemini_batch.py"
prompt_config="${project_dir}/api_models/configs/prompts.json"
api_key_file=/home/nfs/zli33/keys/gemini_api.txt
registry_file="${results_root}/gemini/gemini_registry.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build/copy the SIF first." >&2
    exit 1
fi

if [ ! -f "$project_dir/$inference_script" ]; then
    echo "[ERROR] $inference_script not found in: $project_dir" >&2
    exit 1
fi

if [ ! -d "$data_parent" ]; then
    echo "[ERROR] Batch data folder not found: $data_parent" >&2
    exit 1
fi

if [ ! -f "$prompt_config" ]; then
    echo "[ERROR] Prompt config not found: $prompt_config" >&2
    exit 1
fi

if [ ! -f "$api_key_file" ]; then
    echo "[ERROR] Gemini API key file not found: $api_key_file" >&2
    exit 1
fi

# Ensure output and slurm log directories exist
mkdir -p /home/nfs/zli33/slurm_outputs/gemini-batch
mkdir -p "$output_dir"

# ---------------------------------------------------------------------------
# Run batch inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file       = $sif_file"
echo "[INFO] project_dir    = $project_dir"
echo "[INFO] dataset        = $dataset_name"
echo "[INFO] utt            = $utt_count"
echo "[INFO] batch          = $batch_id"
echo "[INFO] data_parent    = $data_parent"
echo "[INFO] gemini_mode    = $gemini_mode"
echo "[INFO] prompt_config  = $prompt_config"
echo "[INFO] prompt_choice  = $prompt_choice"
echo "[INFO] inference_script = $inference_script"
echo "[INFO] registry       = $registry_file"
echo "[INFO] output_json    = $output_json"
echo ""

apptainer exec \
    --bind "$project_dir":/workspace \
    --bind "${gestalt_root}":"${gestalt_root}" \
    --bind "${results_root}":"${results_root}" \
    --bind "$api_key_file":"$api_key_file":ro \
    --pwd /workspace \
    "$sif_file" \
    python "api_models/gemini_batch.py" \
        --data-root "$data_parent" \
        --output "$output_json" \
        --prompt-choice "$prompt_choice" \
        --prompts-config "api_models/configs/prompts.json" \
        --utt-count "$utt_count" \
        --mode "$gemini_mode" \
        --api-key-path "$api_key_file" \
        --dataset "$dataset_name" \
        --batch-id "batch${batch_id}" \
        --registry "$registry_file"

echo ""
echo "[INFO] Batch job submitted. Run gemini_retrieve_daic.sh to download results later."
