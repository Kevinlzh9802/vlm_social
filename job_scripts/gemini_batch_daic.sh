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
#   sbatch job_scripts/gemini_batch_daic.sh --dataset mintrec2 --batch 1 --prompt intention --annotated
#   sbatch job_scripts/gemini_batch_daic.sh --dataset mintrec2 --batch 1 --prompt intention --annotated --comparison
#   sbatch job_scripts/gemini_batch_daic.sh --dataset mintrec2 --batch 1 --prompt intention --annotated --no-audio

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
dataset_name=""
prompt_choice=""
utt_count=""
batch_number=""
gemini_mode="2.5-flash"
annotated=0
comparison=0
no_audio=0

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
        --annotated)
            annotated=1; shift ;;
        --comparison)
            comparison=1; shift ;;
        --no-audio)
            no_audio=1; shift ;;
        -h|--help)
            echo "Usage: sbatch job_scripts/gemini_batch_daic.sh --dataset <dataset> --batch <number> --prompt <prompt_choice> [--utt <1|2|3>] [--gemini-mode <mode>] [--annotated] [--comparison] [--no-audio]" >&2
            echo "  --utt is required unless --annotated is set. With --annotated, all 1/2/3-utt groups are submitted." >&2
            echo "  --no-audio is only valid together with --annotated, omits .wav inputs, and selects no-audio result roots." >&2
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
usage_msg="Usage: sbatch job_scripts/gemini_batch_daic.sh --dataset <dataset> --batch <number> --prompt <prompt_choice> [--utt <1|2|3>] [--gemini-mode <mode>] [--annotated] [--comparison] [--no-audio]"

if [ -z "$dataset_name" ]; then echo "[ERROR] --dataset is required" >&2; echo "$usage_msg" >&2; exit 1; fi
if [ -z "$prompt_choice" ]; then echo "[ERROR] --prompt is required" >&2; echo "$usage_msg" >&2; exit 1; fi
if [ -z "$batch_number" ];  then echo "[ERROR] --batch is required"  >&2; echo "$usage_msg" >&2; exit 1; fi
if [ "$annotated" = "1" ] && [ -n "$utt_count" ]; then
    echo "[ERROR] --utt is disabled when --annotated is set; annotated mode submits all 1/2/3-utt groups." >&2
    echo "$usage_msg" >&2
    exit 1
fi
if [ "$annotated" = "0" ] && [ -z "$utt_count" ]; then
    echo "[ERROR] --utt is required unless --annotated is set" >&2
    echo "$usage_msg" >&2
    exit 1
fi
if [ "$comparison" = "1" ] && [ "$annotated" = "0" ]; then
    echo "[ERROR] --comparison is only supported together with --annotated" >&2
    echo "$usage_msg" >&2
    exit 1
fi
if [ "$no_audio" = "1" ] && [ "$annotated" = "0" ]; then
    echo "[ERROR] --no-audio is only supported together with --annotated" >&2
    echo "$usage_msg" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
case "$prompt_choice" in
    intention|affordance) ;;
    *)
        echo "[WARN] Unexpected prompt choice: $prompt_choice (expected 'intention' or 'affordance'). Continuing; prompts.json validation will decide." >&2 ;;
esac

if [ "$annotated" = "0" ]; then
    case "$utt_count" in
        1|2|3) ;;
        *)
            echo "[ERROR] Invalid utt count: $utt_count (expected 1, 2, or 3)" >&2; exit 1 ;;
    esac
fi

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
gestalt_data_root=/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench
default_gestalt_root="${gestalt_data_root}/human_eval/samples"
default_output_root=/tudelft.net/staff-umbrella/neon/zonghuan/results/gestalt_bench/human_eval/gemini

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
    gestalt_root="${default_gestalt_root}"
    output_root="${default_output_root}"
fi

if [ "$annotated" = "1" ]; then
    candidate_utt_counts=(1 2 3)
else
    candidate_utt_counts=("$utt_count")
fi

inference_script="api_models/gemini_batch.py"
prompt_config="${project_dir}/api_models/configs/prompts.json"
api_key_file=/home/nfs/zli33/keys/gemini_api.txt
registry_file="${output_root}/gemini_registry.json"

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

utt_counts=()
for current_utt in "${candidate_utt_counts[@]}"; do
    data_parent="${gestalt_root}/${dataset_name}/context/${current_utt}-utt_group/batch${batch_id}"
    if [ ! -d "$data_parent" ]; then
        if [ "$annotated" = "1" ]; then
            echo "[WARN] Batch data folder not found, skipping ${current_utt}-utt group: $data_parent" >&2
            continue
        else
            echo "[ERROR] Batch data folder not found: $data_parent" >&2
            exit 1
        fi
    fi
    utt_counts+=("$current_utt")
done

if [ "${#utt_counts[@]}" -eq 0 ]; then
    echo "[ERROR] No batch data folders found for dataset=${dataset_name}, batch=${batch_id}" >&2
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
for current_utt in "${utt_counts[@]}"; do
    output_dir="${output_root}/${dataset_name}/context/${current_utt}-utt_group/${gemini_mode}_${prompt_choice}_single-turn"
    mkdir -p "$output_dir"
done

# ---------------------------------------------------------------------------
# Run batch inference
# ---------------------------------------------------------------------------
echo "[INFO] sif_file       = $sif_file"
echo "[INFO] project_dir    = $project_dir"
echo "[INFO] dataset        = $dataset_name"
echo "[INFO] utt            = ${utt_count:-all}"
echo "[INFO] batch          = $batch_id"
echo "[INFO] annotated      = $annotated"
echo "[INFO] comparison     = $comparison"
echo "[INFO] no_audio       = $no_audio"
echo "[INFO] gestalt_root   = $gestalt_root"
echo "[INFO] output_root    = $output_root"
echo "[INFO] gemini_mode    = $gemini_mode"
echo "[INFO] prompt_config  = $prompt_config"
echo "[INFO] prompt_choice  = $prompt_choice"
echo "[INFO] inference_script = $inference_script"
echo "[INFO] registry       = $registry_file"
echo ""

python_extra_args=()
if [ "$no_audio" = "1" ]; then
    python_extra_args+=(--no-audio)
fi

for current_utt in "${utt_counts[@]}"; do
    data_parent="${gestalt_root}/${dataset_name}/context/${current_utt}-utt_group/batch${batch_id}"
    output_dir="${output_root}/${dataset_name}/context/${current_utt}-utt_group/${gemini_mode}_${prompt_choice}_single-turn"
    output_json="${output_dir}/batch${batch_id}.json"

    echo "[INFO] submitting utt       = $current_utt"
    echo "[INFO] data_parent          = $data_parent"
    echo "[INFO] output_json          = $output_json"
    echo ""

    apptainer exec \
        --bind "$project_dir":/workspace \
        --bind "${gestalt_root}":"${gestalt_root}" \
        --bind "${output_root}":"${output_root}" \
        --bind "$api_key_file":"$api_key_file":ro \
        --pwd /workspace \
        "$sif_file" \
        python "api_models/gemini_batch.py" \
            --data-root "$data_parent" \
            --output "$output_json" \
            --prompt-choice "$prompt_choice" \
            --prompts-config "api_models/configs/prompts.json" \
            --utt-count "$current_utt" \
            --mode "$gemini_mode" \
            --api-key-path "$api_key_file" \
            --dataset "$dataset_name" \
            --batch-id "batch${batch_id}" \
            --registry "$registry_file" \
            "${python_extra_args[@]}"

    echo ""
done

echo ""
echo "[INFO] Batch job(s) submitted. Run gemini_retrieve_daic.sh to download results later."
