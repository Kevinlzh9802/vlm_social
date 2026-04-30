#!/bin/bash
#SBATCH --job-name="gemma_ingroup"
#SBATCH --time=10:00:00
#SBATCH --partition=ewi-insy,all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --mail-type=END
#SBATCH --output=/home/zli33/linuxhome/slurm_outputs/gemma/slurm_%j.out
#SBATCH --error=/home/zli33/linuxhome/slurm_outputs/gemma/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/zli33/linuxhome/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif"
DATA_ROOT="/tudelft.net/staff-umbrella/neon/ingroup_dataset"
MODEL_PATH="/tudelft.net/staff-umbrella/neon/zonghuan/models/GemmaE4B"
HF_CACHE="/tudelft.net/staff-umbrella/neon/zonghuan/.cache/huggingface"

VIDEO_MEDIA_PATH_PREFIX="https://covfee.ewi.tudelft.nl/P8wPkLamHiAMOvb29g9h3AFy8tXACT1e/video_segs"
AUDIO_MEDIA_PATH_PREFIX="https://covfee.ewi.tudelft.nl/P8wPkLamHiAMOvb29g9h3AFy8tXACT1e/audio_segs_normalized"

input_json="/tudelft.net/staff-umbrella/neon/B1_pipeline/annotation_clips.json"
output_dir="/tudelft.net/staff-umbrella/neon/B1_pipeline/model_responses"
output_json=""
prompt_config="${PROJECT_ROOT}/gemma/prompt_ingroup.json"
max_new_tokens="512"
max_video_frames="32"
enable_thinking=0
do_sample=0
no_audio=0
limit=""
index_range=""
start_index=""
end_index=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [options]" >&2
    echo "Options:" >&2
    echo "  --input-json PATH                          Default: ${input_json}" >&2
    echo "  --output PATH                              Exact output JSON file path" >&2
    echo "  --output-dir PATH                          Default: ${output_dir}" >&2
    echo "  --prompt-config PATH                       Default: ${prompt_config}" >&2
    echo "  --model-path PATH                          Default: ${MODEL_PATH}" >&2
    echo "  --sif-path PATH                            Default: ${SIF_PATH}" >&2
    echo "  --data-root PATH                           Default: ${DATA_ROOT}" >&2
    echo "  --max-new-tokens N                         Default: ${max_new_tokens}" >&2
    echo "  --max-video-frames N                       Default: ${max_video_frames}" >&2
    echo "  --enable-thinking                          Enable Gemma thinking mode" >&2
    echo "  --do-sample                                Use Gemma sampling parameters" >&2
    echo "  --no-audio                                 Omit audio inputs" >&2
    echo "  --limit N                                  Cap retained records" >&2
    echo "  --index-range X-Y                          Process zero-based JSON indices X through Y" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-json)
            input_json="${2:?Missing value for --input-json}"
            shift 2
            ;;
        --output)
            output_json="${2:?Missing value for --output}"
            shift 2
            ;;
        --output-dir)
            output_dir="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --prompt-config)
            prompt_config="${2:?Missing value for --prompt-config}"
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
        --limit)
            limit="${2:?Missing value for --limit}"
            shift 2
            ;;
        --index-range)
            index_range="${2:?Missing value for --index-range}"
            shift 2
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

case "${max_video_frames}" in
    ''|*[!0-9]*)
        echo "[ERROR] Invalid max video frames: ${max_video_frames}" >&2
        exit 1
        ;;
esac

case "${max_new_tokens}" in
    ''|*[!0-9]*)
        echo "[ERROR] Invalid max new tokens: ${max_new_tokens}" >&2
        exit 1
        ;;
esac

if [[ -n "${limit}" ]]; then
    case "${limit}" in
        ''|*[!0-9]*)
            echo "[ERROR] Invalid limit: ${limit}" >&2
            exit 1
            ;;
    esac
fi

if [[ -n "${index_range}" ]]; then
    if [[ "${index_range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        start_index="${BASH_REMATCH[1]}"
        end_index="${BASH_REMATCH[2]}"
        if (( end_index < start_index )); then
            echo "[ERROR] Invalid index range: ${index_range} (end is before start)" >&2
            exit 1
        fi
    else
        echo "[ERROR] Invalid index range: ${index_range}; expected X-Y, for example 0-99" >&2
        exit 1
    fi
fi

if [[ "${input_json}" != /* ]]; then
    input_json="${DATA_ROOT}/${input_json}"
fi

VIDEO_LOCAL_PATH_PREFIX="${DATA_ROOT}/for_annotation/annotation_video"
AUDIO_LOCAL_PATH_PREFIX="${DATA_ROOT}/for_annotation/annotation_audio"

if [[ -z "${output_json}" ]]; then
    input_stem="$(basename "${input_json}")"
    input_stem="${input_stem%.json}"
    if [[ -n "${index_range}" ]]; then
        input_stem="${input_stem}_${start_index}-${end_index}"
    fi
    output_json="${output_dir}/${input_stem}.json"
fi

if [[ "${output_json}" != /* ]]; then
    output_json="${DATA_ROOT}/${output_json}"
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "[ERROR] SIF not found: ${SIF_PATH}" >&2
    echo "Build or copy a Gemma inference image with transformers, torch, torchvision, librosa, and accelerate." >&2
    exit 1
fi

if [[ ! -f "${PROJECT_ROOT}/gemma/batch_infer_ingroup.py" ]]; then
    echo "[ERROR] Gemma ingroup inference script not found under ${PROJECT_ROOT}/gemma" >&2
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

if [[ ! -f "${input_json}" ]]; then
    echo "[ERROR] Input JSON not found: ${input_json}" >&2
    exit 1
fi

mkdir -p /home/zli33/linuxhome/slurm_outputs/gemma
mkdir -p "${HF_CACHE}"
mkdir -p "$(dirname "${output_json}")"

echo "[INFO] project_root              = ${PROJECT_ROOT}"
echo "[INFO] sif_path                  = ${SIF_PATH}"
echo "[INFO] data_root                 = ${DATA_ROOT}"
echo "[INFO] input_json                = ${input_json}"
echo "[INFO] output_dir                = ${output_dir}"
echo "[INFO] output_json               = ${output_json}"
echo "[INFO] model_path                = ${MODEL_PATH}"
echo "[INFO] prompt_config             = ${prompt_config}"
echo "[INFO] max_new_tokens            = ${max_new_tokens}"
echo "[INFO] max_video_frames          = ${max_video_frames}"
echo "[INFO] no_audio                  = ${no_audio}"
echo "[INFO] enable_thinking           = ${enable_thinking}"
echo "[INFO] do_sample                 = ${do_sample}"
echo "[INFO] limit                     = ${limit:-<none>}"
echo "[INFO] index_range               = ${index_range:-<none>}"
echo "[INFO] video_media_path_prefix   = ${VIDEO_MEDIA_PATH_PREFIX}"
echo "[INFO] video_local_path_prefix   = ${VIDEO_LOCAL_PATH_PREFIX}"
echo "[INFO] audio_media_path_prefix   = ${AUDIO_MEDIA_PATH_PREFIX}"
echo "[INFO] audio_local_path_prefix   = ${AUDIO_LOCAL_PATH_PREFIX}"

python_args=(
    python /workspace/gemma/batch_infer_ingroup.py
    --model "${MODEL_PATH}"
    --input-json "${input_json}"
    --output "${output_json}"
    --prompt-config "${prompt_config}"
    --max-new-tokens "${max_new_tokens}"
    --max-video-frames "${max_video_frames}"
    --video-media-path-prefix "${VIDEO_MEDIA_PATH_PREFIX}"
    --video-local-path-prefix "${VIDEO_LOCAL_PATH_PREFIX}"
    --audio-media-path-prefix "${AUDIO_MEDIA_PATH_PREFIX}"
    --audio-local-path-prefix "${AUDIO_LOCAL_PATH_PREFIX}"
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
if [[ -n "${limit}" ]]; then
    python_args+=(--limit "${limit}")
fi
if [[ -n "${index_range}" ]]; then
    python_args+=(--start-index "${start_index}" --end-index "${end_index}")
fi

srun apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    --bind /home/zli33/linuxhome:/home/zli33/linuxhome \
    --env HF_HOME="${HF_CACHE}" \
    --env TRANSFORMERS_CACHE="${HF_CACHE}" \
    "${SIF_PATH}" \
    "${python_args[@]}"

echo "[INFO] Gemma ingroup inference completed: ${output_json}"
