#!/bin/bash
#SBATCH --job-name="clip_stats"
#SBATCH --time=00:30:00
#SBATCH --partition=insy,general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/analysis.sif"
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_FULL_ROOT="${DEFAULT_DATA_ROOT}"
DEFAULT_TASK2_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/data"
DEFAULT_TASK1_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task1"
DEFAULT_TASK1_JSON="${DEFAULT_TASK1_ROOT}"
DEFAULT_TASK1_LOCAL_PREFIX="${DEFAULT_DATA_ROOT}/human_eval/videos"
DEFAULT_MEDIA_URL_PREFIX_1="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_MEDIA_URL_PREFIX_2="http://localhost:5000/api/media/gestalt_bench/annotation2/"
DEFAULT_OUTPUT_NAME="clip_statistics"
DEFAULT_FFPROBE="ffprobe"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--full-root PATH] [--task2-root PATH] [--task1-root PATH] [--task1-json PATH] [--task1-local-prefix PATH] [--media-url-prefix URL] [--output-name NAME] [--ffprobe PATH]" >&2
    echo "  full-root: GestaltBench root with <dataset>/context/<n>-utt_group (default: ${DEFAULT_FULL_ROOT})" >&2
    echo "  task2-root: task2 manipulation_full/data root (default: ${DEFAULT_TASK2_ROOT})" >&2
    echo "  task1-root: task1 output root (default: ${DEFAULT_TASK1_ROOT})" >&2
    echo "  task1-json: task1 JSON file or directory containing task1_b*.json (default: ${DEFAULT_TASK1_JSON})" >&2
    echo "  task1-local-prefix: local prefix replacing task1 media URLs (default: ${DEFAULT_TASK1_LOCAL_PREFIX})" >&2
    echo "  media-url-prefix: URL prefix to replace; may be passed multiple times. Defaults include annotation1 and annotation2." >&2
    echo "  output-name: basename for .txt/.json outputs in each output root (default: ${DEFAULT_OUTPUT_NAME})" >&2
    echo "  ffprobe: ffprobe executable inside the container (default: ${DEFAULT_FFPROBE})" >&2
}

FULL_ROOT="${DEFAULT_FULL_ROOT}"
TASK2_ROOT="${DEFAULT_TASK2_ROOT}"
TASK1_ROOT="${DEFAULT_TASK1_ROOT}"
TASK1_JSON="${DEFAULT_TASK1_JSON}"
TASK1_LOCAL_PREFIX="${DEFAULT_TASK1_LOCAL_PREFIX}"
MEDIA_URL_PREFIXES=("${DEFAULT_MEDIA_URL_PREFIX_1}" "${DEFAULT_MEDIA_URL_PREFIX_2}")
OUTPUT_NAME="${DEFAULT_OUTPUT_NAME}"
FFPROBE="${DEFAULT_FFPROBE}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full-root)
            FULL_ROOT="${2:?Missing value for --full-root}"
            shift 2
            ;;
        --task2-root)
            TASK2_ROOT="${2:?Missing value for --task2-root}"
            shift 2
            ;;
        --task1-root)
            TASK1_ROOT="${2:?Missing value for --task1-root}"
            shift 2
            ;;
        --task1-json)
            TASK1_JSON="${2:?Missing value for --task1-json}"
            shift 2
            ;;
        --task1-local-prefix)
            TASK1_LOCAL_PREFIX="${2:?Missing value for --task1-local-prefix}"
            shift 2
            ;;
        --media-url-prefix)
            if [[ "${#MEDIA_URL_PREFIXES[@]}" -eq 2 && "${MEDIA_URL_PREFIXES[0]}" == "${DEFAULT_MEDIA_URL_PREFIX_1}" && "${MEDIA_URL_PREFIXES[1]}" == "${DEFAULT_MEDIA_URL_PREFIX_2}" ]]; then
                MEDIA_URL_PREFIXES=()
            fi
            MEDIA_URL_PREFIXES+=("${2:?Missing value for --media-url-prefix}")
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="${2:?Missing value for --output-name}"
            shift 2
            ;;
        --ffprobe)
            FFPROBE="${2:?Missing value for --ffprobe}"
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

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build or copy it from ${PROJECT_ROOT}/apptainer/analysis.def first." >&2
    exit 1
fi

if [[ ! -d "${FULL_ROOT}" ]]; then
    echo "Full root does not exist: ${FULL_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${TASK2_ROOT}" ]]; then
    echo "Task2 root does not exist: ${TASK2_ROOT}" >&2
    exit 1
fi

if [[ ! -e "${TASK1_JSON}" ]]; then
    echo "Task1 JSON path does not exist: ${TASK1_JSON}" >&2
    exit 1
fi

if [[ ! -d "${TASK1_LOCAL_PREFIX}" ]]; then
    echo "Task1 local prefix does not exist: ${TASK1_LOCAL_PREFIX}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/vlm_social
mkdir -p "${TASK1_ROOT}"

echo "Project root:          ${PROJECT_ROOT}"
echo "Apptainer image:       ${SIF_PATH}"
echo "Full root:             ${FULL_ROOT}"
echo "Task2 root:            ${TASK2_ROOT}"
echo "Task1 root:            ${TASK1_ROOT}"
echo "Task1 JSON:            ${TASK1_JSON}"
echo "Task1 local prefix:    ${TASK1_LOCAL_PREFIX}"
echo "Output name:           ${OUTPUT_NAME}"
echo "ffprobe:               ${FFPROBE}"
echo "Media URL prefixes:    ${MEDIA_URL_PREFIXES[*]}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/clip_statistics.py
    --full-root "${FULL_ROOT}"
    --task2-root "${TASK2_ROOT}"
    --task1-root "${TASK1_ROOT}"
    --task1-json "${TASK1_JSON}"
    --task1-local-prefix "${TASK1_LOCAL_PREFIX}"
    --output-name "${OUTPUT_NAME}"
    --ffprobe "${FFPROBE}"
)

for media_url_prefix in "${MEDIA_URL_PREFIXES[@]}"; do
    PYTHON_ARGS+=(--media-url-prefix "${media_url_prefix}")
done

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /home/nfs/zli33:/home/nfs/zli33 \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
