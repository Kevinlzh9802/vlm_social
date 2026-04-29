#!/bin/bash
#SBATCH --job-name="ming-lite-omni_batch_infer"
#SBATCH --partition=gpu-a100
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/scratch/zli33/slurm_outputs/ming-lite-omni/slurm_%j.out
#SBATCH --error=/scratch/zli33/slurm_outputs/ming-lite-omni/slurm_%j.err

# Batch Ming-Lite-Omni inference via Apptainer.
#
# Submit from the project folder:
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --mode context --utt 2 --batch 2 --prompt affordance
#   sbatch job_scripts/inference_batch.sh -set mintrec2 --mode nested --utt 3 --batch 12 -prompt intention
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --utt 1 --batch 1 --prompt intention --conversation-mode single-turn
#   sbatch job_scripts/inference_batch.sh --dataset mintrec6 --mode nested --utt 2 --batch 5 --prompt intention
#   sbatch job_scripts/inference_batch.sh --dataset mintrec2 --utt 3 --batch 9 --prompt affordance

set -euo pipefail

usage_message="Usage: sbatch job_scripts/inference_batch.sh --dataset <dataset> [--mode <context|nested>] [--utt <1|2|3>] --batch <number> [--conversation-mode <single-turn|multi-turn>] [--attn-implementation <auto|eager|sdpa|flash_attention_2>] [--annotator <n>] --prompt <prompt_choice>"

dataset_name=""
prompt_choice=""
mode_name="context"
utt_count=""
batch_number=""
conversation_mode="single-turn"
conversation_mode_set=false
attn_implementation="flash_attention_2"
annotator_number=""

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
        --batch)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            batch_number="$2"
            shift 2
            ;;
        -annotator|--annotator)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            annotator_number="$2"
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
        --attn-implementation)
            if [ "$#" -lt 2 ]; then
                echo "[ERROR] Missing value for $1" >&2
                exit 1
            fi
            attn_implementation="$2"
            shift 2
            ;;
        -h|--help)
            echo "$usage_message" >&2
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

if [ -z "$dataset_name" ]; then
    echo "$usage_message" >&2
    exit 1
fi

if [ -z "$prompt_choice" ]; then
    echo "$usage_message" >&2
    exit 1
fi

if [ -z "$batch_number" ]; then
    echo "$usage_message" >&2
    exit 1
fi

case "$prompt_choice" in
    intention|affordance)
        ;;
    *)
        echo "[WARN] Unexpected prompt choice: $prompt_choice (expected 'intention' or 'affordance'). Continuing; prompts.json validation will decide whether it is usable." >&2
        ;;
esac

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

case "$annotator_number" in
    '')
        ;;
    *[!0-9]*)
        echo "[ERROR] Invalid annotator number: $annotator_number (expected a positive integer)" >&2
        exit 1
        ;;
esac

if [ -n "$annotator_number" ]; then
    if [ -n "$utt_count" ]; then
        echo "[ERROR] --annotator cannot be combined with --utt; annotator mode runs all utt groups sequentially" >&2
        exit 1
    fi
else
    case "$utt_count" in
        1|2|3)
            ;;
        *)
            echo "[ERROR] Invalid utt count: $utt_count (expected 1, 2, or 3)" >&2
            exit 1
            ;;
    esac
fi

case "$conversation_mode" in
    single-turn|multi-turn)
        ;;
    *)
        echo "[ERROR] Invalid conversation mode: $conversation_mode (expected single-turn or multi-turn)" >&2
        exit 1
        ;;
esac

case "$attn_implementation" in
    auto|eager|sdpa|flash_attention_2)
        ;;
    *)
        echo "[ERROR] Invalid attn implementation: $attn_implementation (expected auto, eager, sdpa, or flash_attention_2)" >&2
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
project_dir="/scratch/zli33/models/Ming-lite-omni"
sif_file="/scratch/zli33/apptainers/ming-lite-omni.sif"
gestalt_root="/scratch/zli33/data/gestalt_bench"
data_root_host="/scratch/zli33/data"
prompt_config="$project_dir/prompts/prompts.json"

if [ -n "$annotator_number" ]; then
    data_base_root="${gestalt_root}/human_eval/task2/manipulation_full/annotator${annotator_number}"
    results_base_root="${gestalt_root}/human_eval/task2/manipulation_full/results"
    utt_counts_to_run=(1 2 3)
else
    data_base_root="$gestalt_root"
    results_base_root="${gestalt_root}/results/ming-lite-omni"
    utt_counts_to_run=("$utt_count")
fi

resolve_data_root() {
    local utt_value="$1"
    if [ "$mode_name" = "nested" ]; then
        printf '%s\n' "${data_base_root}/${dataset_name}/nested/data/batch${batch_id}"
    else
        printf '%s\n' "${data_base_root}/${dataset_name}/${mode_name}/${utt_value}-utt_group/batch${batch_id}"
    fi
}

resolve_output_dir() {
    local utt_value="$1"
    printf '%s\n' "${results_base_root}/${dataset_name}/${mode_name}/${utt_value}-utt_group/Ming-lite-omni_${prompt_choice}_${conversation_mode}"
}

validate_prompt_variant() {
    local utt_value="$1"
    if ! python3 -c 'import json, sys; data = json.load(open(sys.argv[1], encoding="utf-8")); conversation_mode = sys.argv[3]; utt_count = int(sys.argv[4]); section_name = "single_turn_prompts" if conversation_mode == "single-turn" else "multi_turn_prompts"; variant_suffix = "single_utt" if utt_count == 1 else "multi_utt"; variant_key = f"{sys.argv[2]}_{variant_suffix}"; section = data.get(section_name, {}); prompt = section.get(variant_key); required_fields = ("text",) if conversation_mode == "single-turn" else ("first", "after"); ok = isinstance(section, dict) and isinstance(prompt, dict) and all(isinstance(prompt.get(field), str) and prompt.get(field, "").strip() for field in required_fields); sys.exit(0 if ok else 1)' "$prompt_config" "$prompt_choice" "$conversation_mode" "$utt_value"; then
        echo "[ERROR] Prompt choice '$prompt_choice' does not provide the required variant for conversation_mode=$conversation_mode and utt_count=$utt_value in: $prompt_config" >&2
        return 1
    fi
}

run_single_inference() {
    local utt_value="$1"
    local data_root="$2"
    local output_dir="$3"
    local output_json="$4"

    mkdir -p /scratch/zli33/slurm_outputs/ming-lite-omni
    mkdir -p "$output_dir"

    echo "[INFO] sif_file          = $sif_file"
    echo "[INFO] project_dir       = $project_dir"
    echo "[INFO] dataset           = $dataset_name"
    echo "[INFO] mode              = $mode_name"
    echo "[INFO] utt               = $utt_value"
    echo "[INFO] batch             = $batch_id"
    echo "[INFO] annotator         = ${annotator_number:-<none>}"
    echo "[INFO] data_root         = $data_root"
    echo "[INFO] prompt_config     = $prompt_config"
    echo "[INFO] prompt_choice     = $prompt_choice"
    echo "[INFO] conversation_mode = $conversation_mode"
    echo "[INFO] attn_impl         = $attn_implementation"
    echo "[INFO] output_json       = $output_json"
    echo ""

    echo "[HOST] nvidia-smi:"
    nvidia-smi || true

    # Triton's JIT compiler needs the unversioned "libcuda.so" to link GPU
    # kernels. Apptainer --nv injects the host driver into /.singularity.d/libs,
    # but Triton's libcuda_dirs() does not look there by default.
    apptainer exec --nv --writable-tmpfs \
      --bind "$project_dir":/workspace \
      --bind "$data_root_host":"$data_root_host" \
      --pwd /workspace \
      "$sif_file" \
      bash -lc '
        set -euo pipefail

        real=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
        test -n "$real"

        mkdir -p /tmp/triton-libcuda
        ln -sf "$real" /tmp/triton-libcuda/libcuda.so
        ln -sf "$real" /tmp/triton-libcuda/libcuda.so.1

        export TRITON_LIBCUDA_PATH=/tmp/triton-libcuda
        export LD_LIBRARY_PATH=/tmp/triton-libcuda:${LD_LIBRARY_PATH:-}

        python - <<PY
from pathlib import Path
p = Path("/opt/conda/lib/python3.10/site-packages/triton/common/build.py")
txt = p.read_text()
needle = "@functools.lru_cache()\ndef libcuda_dirs():\n"
patch = """@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv(\"TRITON_LIBCUDA_PATH\")
    if env_libcuda_path:
        return [env_libcuda_path]
"""
if "TRITON_LIBCUDA_PATH" not in txt:
    txt = txt.replace(needle, patch)
    p.write_text(txt)
    print("Patched Triton build.py")
else:
    print("Triton build.py already patched")
PY

        python /workspace/batch_infer.py \
          --model /workspace \
          --data-root "'"$data_root"'" \
          --output "'"$output_json"'" \
          --mode "'"$mode_name"'" \
          --prompt-choice "'"$prompt_choice"'" \
          --utt-count "'"$utt_value"'" \
          --conversation-mode "'"$conversation_mode"'" \
          --attn-implementation "'"$attn_implementation"'"
      '

    echo ""
    echo "[INFO] Batch inference completed. Results saved to $output_json"
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
if [ ! -f "$sif_file" ]; then
    echo "[ERROR] SIF not found: $sif_file" >&2
    echo "  Build it first: bash apptainer/build.sh" >&2
    exit 1
fi

if [ ! -f "$project_dir/batch_infer.py" ]; then
    echo "[ERROR] Script not found: $project_dir/batch_infer.py" >&2
    exit 1
fi

if [ ! -f "$prompt_config" ]; then
    echo "[ERROR] Prompt config not found: $prompt_config" >&2
    exit 1
fi

ran_any=false
for current_utt in "${utt_counts_to_run[@]}"; do
    validate_prompt_variant "$current_utt"

    data_root="$(resolve_data_root "$current_utt")"
    output_dir="$(resolve_output_dir "$current_utt")"
    output_json="${output_dir}/batch${batch_id}.json"

    if [ ! -d "$data_root" ]; then
        if [ -n "$annotator_number" ]; then
            echo "[WARN] Skipping utt=$current_utt because batch data folder was not found: $data_root" >&2
            continue
        fi
        echo "[ERROR] Batch data folder not found: $data_root" >&2
        exit 1
    fi

    ran_any=true
    run_single_inference "$current_utt" "$data_root" "$output_dir" "$output_json"
done

if [ "$ran_any" = false ]; then
    echo "[ERROR] No batch data folders were found for dataset=$dataset_name mode=$mode_name under $data_base_root" >&2
    exit 1
fi
