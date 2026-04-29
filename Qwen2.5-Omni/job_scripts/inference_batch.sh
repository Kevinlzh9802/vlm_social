#!/bin/bash
#SBATCH --job-name="qwen2.5-omni_batch_infer"
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/qwen2.5-omni/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/qwen2.5-omni/slurm_%j.err

# Batch Qwen2.5-Omni inference via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --utt 2 --batch 2 -model 7B -prompt affordance
#   sbatch job_scripts/inference_batch.sh -set mintrec2 --mode nested --utt 3 --batch 12 -model 3B -prompt intention
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --utt 1 --batch 1 --model 7B --prompt intention --conversation-mode single-turn
#   sbatch job_scripts/inference_batch.sh --dataset mintrec6 --mode nested --utt 2 --batch 5 --model 7B --prompt intention
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --utt 3 --batch 9 --model 3B --prompt affordance
#   sbatch job_scripts/inference_batch.sh -set mintrec6 --mode nested --utt 1 --batch 10 -model 3B -prompt intention
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --batch 2 -model 7B -prompt intention -annotator 1

set -euo pipefail

model_size=7B
dataset_name=""
prompt_choice=""
mode_name="context"
utt_count=""
batch_number=""
conversation_mode="single-turn"
conversation_mode_set=false
annotator_id=""

usage() {
    echo "Usage: sbatch job_scripts/inference_batch.sh --dataset <dataset> [--mode <context|nested>] --utt <1|2|3> --batch <number> [--conversation-mode <single-turn|multi-turn>] [-model 7B|3B] -prompt <prompt_choice>" >&2
    echo "       sbatch job_scripts/inference_batch.sh --dataset <dataset> [--mode <context|nested>] --batch <number> [--conversation-mode <single-turn|multi-turn>] [-model 7B|3B] -prompt <prompt_choice> -annotator <n>" >&2
}

validate_prompt_variant() {
    local target_utt_count="$1"

    if ! python3 -c 'import json, sys; data = json.load(open(sys.argv[1], encoding="utf-8")); conversation_mode = sys.argv[3]; utt_count = int(sys.argv[4]); section_name = "single_turn_prompts" if conversation_mode == "single-turn" else "multi_turn_prompts"; variant_suffix = "single_utt" if utt_count == 1 else "multi_utt"; variant_key = f"{sys.argv[2]}_{variant_suffix}"; section = data.get(section_name, {}); prompt = section.get(variant_key); required_fields = ("text",) if conversation_mode == "single-turn" else ("first", "after"); ok = isinstance(section, dict) and isinstance(prompt, dict) and all(isinstance(prompt.get(field), str) and prompt.get(field, "").strip() for field in required_fields); sys.exit(0 if ok else 1)' "$prompt_config" "$prompt_choice" "$conversation_mode" "$target_utt_count"; then
        echo "[ERROR] Prompt choice '$prompt_choice' does not provide the required variant for conversation_mode=$conversation_mode and utt_count=$target_utt_count in: $prompt_config" >&2
        exit 1
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -set|--set|--dataset)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            dataset_name="$2"
            shift 2
            ;;
        -model|--model)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            model_size="$2"
            shift 2
            ;;
        -prompt|--prompt)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            prompt_choice="$2"
            shift 2
            ;;
        --mode)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            mode_name="$2"
            shift 2
            ;;
        --utt)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            utt_count="$2"
            shift 2
            ;;
        batch|--batch)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            batch_number="$2"
            shift 2
            ;;
        --conversation-mode)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            conversation_mode="$2"
            conversation_mode_set=true
            shift 2
            ;;
        -annotator|--annotator)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            annotator_id="$2"
            shift 2
            ;;
        -h|--help)
            usage
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

if [ -z "$prompt_choice" ]; then
    usage
    exit 1
fi

if [ -z "$dataset_name" ] || [ -z "$batch_number" ]; then
    usage
    exit 1
fi

if [ -n "$annotator_id" ]; then
    case "$annotator_id" in
        ''|*[!0-9]*)
            echo "[ERROR] Invalid annotator id: $annotator_id (expected a positive integer)" >&2
            exit 1
            ;;
    esac
elif [ -z "$utt_count" ]; then
    usage
    exit 1
fi

case "$prompt_choice" in
    intention|affordance)
        ;;
    *)
        echo "[WARN] Unexpected prompt choice: $prompt_choice (expected 'intention' or 'affordance'). Continuing; prompts.json validation will decide whether it is usable." >&2
        ;;
esac

case "$model_size" in
    7B|3B)
        ;;
    *)
        echo "[ERROR] Invalid model size: $model_size (expected 7B or 3B)" >&2
        exit 1
        ;;
esac

if [ -n "$utt_count" ]; then
    case "$utt_count" in
        1|2|3)
            ;;
        *)
            echo "[ERROR] Invalid utt count: $utt_count (expected 1, 2, or 3)" >&2
            exit 1
            ;;
    esac
fi

case "$mode_name" in
    context|nested)
        ;;
    *)
        echo "[ERROR] Invalid mode: $mode_name (expected context or nested)" >&2
        exit 1
        ;;
esac

case "$batch_number" in
    ''|*[!0-9]*)
        echo "[ERROR] Invalid batch number: $batch_number (expected a positive integer)" >&2
        exit 1
        ;;
esac

case "$conversation_mode" in
    single-turn|multi-turn)
        ;;
    *)
        echo "[ERROR] Invalid conversation mode: $conversation_mode (expected single-turn or multi-turn)" >&2
        exit 1
        ;;
esac

if [ "$mode_name" = "nested" ]; then
    if [ "$conversation_mode_set" = false ]; then
        conversation_mode="multi-turn"
    elif [ "$conversation_mode" != "multi-turn" ]; then
        echo "[ERROR] --mode nested requires --conversation-mode multi-turn" >&2
        exit 1
    fi
fi

batch_id=$(printf "%02d" "$batch_number")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir=/home/zli33/projects/Qwen2.5-Omni
sif_file=/scratch/zli33/apptainers/qwen2.5-omni-inference.sif
hf_cache_host=/scratch/zli33/.cache/huggingface
data_root_host=/scratch/zli33/data
model_root_host=/scratch/zli33/models
gestalt_root=/scratch/zli33/data/gestalt_bench

data_base_root="$gestalt_root"
output_base_root="${gestalt_root}/results/qwen2.5"
if [ -n "$annotator_id" ]; then
    data_base_root="${gestalt_root}/human_eval/task2/manipulation_full/annotator${annotator_id}"
    output_base_root="${gestalt_root}/human_eval/task2/manipulation_full/results"
fi

model_path="/scratch/zli33/models/Qwen2.5-Omni-${model_size}"
prompt_config="$project_dir/prompts/prompts.json"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build/copy the SIF first." >&2
    exit 1
fi

if [ ! -d "$model_path" ]; then
    echo "[ERROR] Model path not found: $model_path" >&2
    exit 1
fi

if [ ! -f "$prompt_config" ]; then
    echo "[ERROR] Prompt config not found: $prompt_config" >&2
    exit 1
fi

# Ensure output and cache directories exist
mkdir -p /scratch/zli33/slurm_outputs/qwen2.5-omni
mkdir -p "$hf_cache_host"

# ---------------------------------------------------------------------------
# Run batch inference
# ---------------------------------------------------------------------------
if [ -n "$annotator_id" ]; then
    if [ -n "$utt_count" ]; then
        echo "[INFO] --utt is ignored when -annotator is set; running utt groups 1, 2, and 3 sequentially."
    fi
    utt_counts=(1 2 3)
else
    utt_counts=("$utt_count")
fi

for current_utt_count in "${utt_counts[@]}"; do
    if [ "$mode_name" = "nested" ]; then
        data_parent="${data_base_root}/${dataset_name}/nested/data/batch${batch_id}"
        inference_script="batch_infer_nested.py"
    else
        data_parent="${data_base_root}/${dataset_name}/${mode_name}/${current_utt_count}-utt_group/batch${batch_id}"
        inference_script="batch_infer_context.py"
    fi
    output_dir="${output_base_root}/${dataset_name}/${mode_name}/${current_utt_count}-utt_group/${model_size}_${prompt_choice}_${conversation_mode}"
    output_json="$output_dir/batch${batch_id}.json"

    if [ ! -f "$project_dir/$inference_script" ]; then
        echo "[ERROR] $inference_script not found in: $project_dir" >&2
        exit 1
    fi

    if [ ! -d "$data_parent" ]; then
        echo "[ERROR] Batch data folder not found: $data_parent" >&2
        exit 1
    fi

    validate_prompt_variant "$current_utt_count"
    mkdir -p "$output_dir"

    echo "[INFO] sif_file    = $sif_file"
    echo "[INFO] project_dir = $project_dir"
    echo "[INFO] hf_cache    = $hf_cache_host"
    echo "[INFO] dataset     = $dataset_name"
    echo "[INFO] mode        = $mode_name"
    echo "[INFO] utt         = $current_utt_count"
    echo "[INFO] batch       = $batch_id"
    echo "[INFO] data_parent = $data_parent"
    echo "[INFO] model_size  = $model_size"
    echo "[INFO] model_path  = $model_path"
    echo "[INFO] prompt_config = $prompt_config"
    echo "[INFO] prompt_choice = $prompt_choice"
    echo "[INFO] conversation_mode = $conversation_mode"
    echo "[INFO] inference_script = $inference_script"
    echo "[INFO] output_json = $output_json"
    echo ""

    python_args=(
        python "$inference_script"
        --model "$model_path"
        --data-root "$data_parent"
        --output "$output_json"
        --mode "$mode_name"
        --prompt-choice "$prompt_choice"
        --utt-count "$current_utt_count"
        --conversation-mode "$conversation_mode"
    )

    if [ "$mode_name" = "nested" ]; then
        python_args+=(--group "$current_utt_count")
    fi

    apptainer exec --nv \
        --bind "$project_dir":/workspace \
        --bind "$hf_cache_host":/opt/huggingface \
        --bind "$data_root_host":"$data_root_host" \
        --bind "$model_root_host":"$model_root_host" \
        --env HF_HOME=/opt/huggingface \
        --env TRANSFORMERS_CACHE=/opt/huggingface \
        --pwd /workspace \
        "$sif_file" \
        "${python_args[@]}"

    echo ""
    echo "[INFO] Batch inference completed. Results saved to $output_json"
done
