#!/bin/bash
#SBATCH --job-name="gemma_e4b_infer"
#SBATCH --partition=insy,general
#SBATCH --qos=short 
#SBATCH --time=3:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/gemma/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/gemma/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif"
DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
MODEL_PATH="/tudelft.net/staff-umbrella/neon/zonghuan/models/GemmaE4B"
HF_CACHE="/tudelft.net/staff-umbrella/neon/zonghuan/.cache/huggingface"

dataset_name=""
prompt_choice=""
utt_count=""
batch_number=""
conversation_mode="single-turn"
max_new_tokens="512"
max_video_frames="32"
enable_thinking=0
do_sample=0
no_audio=0
annotated=0
comparison=0

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 --dataset DATASET --utt 1|2|3 --batch N --prompt PROMPT [options]" >&2
    echo "  sbatch $0 --dataset DATASET --batch N --prompt PROMPT --annotated [--comparison] [--no-audio] [options]" >&2
    echo "Options:" >&2
    echo "  --conversation-mode single-turn|multi-turn  Default: single-turn" >&2
    echo "  --model-path PATH                         Default: ${MODEL_PATH}" >&2
    echo "  --sif-path PATH                           Default: ${SIF_PATH}" >&2
    echo "  --data-root PATH                          Default: ${DATA_ROOT}" >&2
    echo "  --max-new-tokens N                        Default: ${max_new_tokens}" >&2
    echo "  --max-video-frames N                      Default: ${max_video_frames}" >&2
    echo "  --enable-thinking                         Enable Gemma thinking mode" >&2
    echo "  --do-sample                               Use Gemma sampling parameters" >&2
    echo "  --annotated                               Run manipulation_full annotated data for utt 1, 2, and 3" >&2
    echo "  --comparison                              Use manipulation_full comparison data/results" >&2
    echo "  --no-audio                                Omit separate .wav inputs for annotated runs" >&2
}

validate_prompt_variant() {
    local target_utt_count="$1"

    if ! python3 -c 'import json, sys; data=json.load(open(sys.argv[1], encoding="utf-8")); mode=sys.argv[3]; utt=int(sys.argv[4]); section_name="single_turn_prompts" if mode=="single-turn" else "multi_turn_prompts"; suffix="single_utt" if utt == 1 else "multi_utt"; variant=sys.argv[2]+"_"+suffix; section=data.get(section_name, {}); prompt=section.get(variant); required=("text",) if mode=="single-turn" else ("first","after"); ok=isinstance(prompt, dict) and all(isinstance(prompt.get(field), str) and prompt[field].strip() for field in required); sys.exit(0 if ok else 1)' "${prompt_config}" "${prompt_choice}" "${conversation_mode}" "${target_utt_count}"; then
        echo "[ERROR] Prompt choice '${prompt_choice}' is not valid for conversation_mode=${conversation_mode}, utt=${target_utt_count}" >&2
        exit 1
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -set|--set|--dataset)
            dataset_name="${2:?Missing value for $1}"
            shift 2
            ;;
        -prompt|--prompt)
            prompt_choice="${2:?Missing value for $1}"
            shift 2
            ;;
        --utt)
            utt_count="${2:?Missing value for --utt}"
            shift 2
            ;;
        --batch)
            batch_number="${2:?Missing value for --batch}"
            shift 2
            ;;
        --conversation-mode)
            conversation_mode="${2:?Missing value for --conversation-mode}"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="${2:?Missing value for --model-path}"
            shift 2
            ;;
        --sif-path)
            SIF_PATH="${2:?Missing value for --sif-path}"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="${2:?Missing value for --data-root}"
            shift 2
            ;;
        --max-new-tokens)
            max_new_tokens="${2:?Missing value for --max-new-tokens}"
            shift 2
            ;;
        --max-video-frames)
            max_video_frames="${2:?Missing value for --max-video-frames}"
            shift 2
            ;;
        --enable-thinking)
            enable_thinking=1
            shift
            ;;
        --do-sample)
            do_sample=1
            shift
            ;;
        --no-audio)
            no_audio=1
            shift
            ;;
        -annotated|--annotated)
            annotated=1
            shift
            ;;
        --comparison)
            comparison=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            usage
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            usage
            echo "Unexpected positional argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${dataset_name}" || -z "${prompt_choice}" || -z "${batch_number}" ]]; then
    usage
    exit 1
fi

if [[ "${annotated}" == "0" && -z "${utt_count}" ]]; then
    usage
    exit 1
fi

if [[ "${no_audio}" == "1" && "${annotated}" == "0" ]]; then
    echo "[ERROR] --no-audio can only be used with --annotated" >&2
    exit 1
fi

if [[ -n "${utt_count}" ]]; then
    case "${utt_count}" in
        1|2|3) ;;
        *)
            echo "[ERROR] Invalid utt count: ${utt_count} (expected 1, 2, or 3)" >&2
            exit 1
            ;;
    esac
fi

case "${batch_number}" in
    ''|*[!0-9]*)
        echo "[ERROR] Invalid batch number: ${batch_number}" >&2
        exit 1
        ;;
esac

case "${max_video_frames}" in
    ''|*[!0-9]*)
        echo "[ERROR] Invalid max video frames: ${max_video_frames}" >&2
        exit 1
        ;;
esac

case "${conversation_mode}" in
    single-turn|multi-turn) ;;
    *)
        echo "[ERROR] Invalid conversation mode: ${conversation_mode}" >&2
        exit 1
        ;;
esac

batch_id=$(printf "%02d" "${batch_number}")
prompt_config="${PROJECT_ROOT}/api_models/configs/prompts.json"
data_base_root="${DATA_ROOT}"
output_base_root="${DATA_ROOT}/results/gemma-4-e4b"

if [[ "${annotated}" == "1" ]]; then
    if [[ "${comparison}" == "1" ]]; then
        data_base_root="${DATA_ROOT}/human_eval/task2/manipulation_full/data_comparison"
        if [[ "${no_audio}" == "1" ]]; then
            output_base_root="${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio_comparison/gemma-4-e4b"
        else
            output_base_root="${DATA_ROOT}/human_eval/task2/manipulation_full/results_comparison/gemma-4-e4b"
        fi
    else
        data_base_root="${DATA_ROOT}/human_eval/task2/manipulation_full/data"
        if [[ "${no_audio}" == "1" ]]; then
            output_base_root="${DATA_ROOT}/human_eval/task2/manipulation_full/results_noaudio/gemma-4-e4b"
        else
            output_base_root="${DATA_ROOT}/human_eval/task2/manipulation_full/results/gemma-4-e4b"
        fi
    fi
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "[ERROR] SIF not found: ${SIF_PATH}" >&2
    echo "Build or copy a Gemma inference image with transformers, torch, torchvision, librosa, and accelerate." >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/gemma/batch_infer_context.py" ]]; then
    echo "[ERROR] Gemma inference script not found under ${PROJECT_ROOT}/gemma" >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "[ERROR] Model path not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${prompt_config}" ]]; then
    echo "[ERROR] Prompt config not found: ${prompt_config}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/gemma
mkdir -p "${HF_CACHE}"

if [[ "${annotated}" == "1" ]]; then
    if [[ -n "${utt_count}" ]]; then
        echo "[INFO] --utt is ignored when --annotated is set; running utt groups 1, 2, and 3 sequentially."
    fi
    utt_counts=(1 2 3)
else
    utt_counts=("${utt_count}")
fi

for current_utt_count in "${utt_counts[@]}"; do
    data_parent="${data_base_root}/${dataset_name}/context/${current_utt_count}-utt_group/batch${batch_id}"
    output_dir="${output_base_root}/${dataset_name}/context/${current_utt_count}-utt_group/E4B_${prompt_choice}_${conversation_mode}"
    output_json="${output_dir}/batch${batch_id}.json"

    if [[ ! -d "${data_parent}" ]]; then
        echo "[ERROR] Batch data folder not found: ${data_parent}" >&2
        exit 1
    fi

    validate_prompt_variant "${current_utt_count}"
    mkdir -p "${output_dir}"

    echo "[INFO] project_root      = ${PROJECT_ROOT}"
    echo "[INFO] sif_path          = ${SIF_PATH}"
    echo "[INFO] data_root         = ${DATA_ROOT}"
    echo "[INFO] data_base_root    = ${data_base_root}"
    echo "[INFO] data_parent       = ${data_parent}"
    echo "[INFO] output_base_root  = ${output_base_root}"
    echo "[INFO] model_path        = ${MODEL_PATH}"
    echo "[INFO] prompt_config     = ${prompt_config}"
    echo "[INFO] prompt_choice     = ${prompt_choice}"
    echo "[INFO] utt_count         = ${current_utt_count}"
    echo "[INFO] batch_id          = ${batch_id}"
    echo "[INFO] conversation_mode = ${conversation_mode}"
    echo "[INFO] max_video_frames  = ${max_video_frames}"
    echo "[INFO] annotated         = ${annotated}"
    echo "[INFO] comparison        = ${comparison}"
    echo "[INFO] no_audio          = ${no_audio}"
    echo "[INFO] output_json       = ${output_json}"

    python_args=(
        python /workspace/gemma/batch_infer_context.py
        --model "${MODEL_PATH}"
        --data-root "${data_parent}"
        --output "${output_json}"
        --prompt-config /workspace/api_models/configs/prompts.json
        --prompt-choice "${prompt_choice}"
        --mode context
        --utt-count "${current_utt_count}"
        --conversation-mode "${conversation_mode}"
        --max-new-tokens "${max_new_tokens}"
        --max-video-frames "${max_video_frames}"
    )

    if [[ "${enable_thinking}" == "1" ]]; then
        python_args+=(--enable-thinking)
    fi
    if [[ "${do_sample}" == "1" ]]; then
        python_args+=(--do-sample)
    fi
    if [[ "${no_audio}" == "1" ]]; then
        python_args+=(--no-audio)
    fi

    srun apptainer exec --nv \
        --bind "${PROJECT_ROOT}:/workspace" \
        --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
        --bind /home/nfs/zli33:/home/nfs/zli33 \
        --env HF_HOME="${HF_CACHE}" \
        --env TRANSFORMERS_CACHE="${HF_CACHE}" \
        "${SIF_PATH}" \
        "${python_args[@]}"

    echo "[INFO] Gemma batch inference completed: ${output_json}"
done
