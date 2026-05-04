#!/bin/bash
#SBATCH --job-name="clip_stats"
#SBATCH --time=03:00:00
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vlm_social/slurm_%j.err

set -euo pipefail

PROJECT_ROOT="/home/nfs/zli33/projects/vlm_social"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/eyetrack.sif"
DEFAULT_DATA_ROOT="/tudelft.net/staff-umbrella/neon/zonghuan/data/gestalt_bench"
DEFAULT_FULL_ROOT="${DEFAULT_DATA_ROOT}"
DEFAULT_TASK2_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task2/manipulation_full/data"
DEFAULT_TASK1_ROOT="${DEFAULT_DATA_ROOT}/human_eval/task1"
DEFAULT_TASK1_ANNOTATION_DIR="${DEFAULT_TASK1_ROOT}/annotations"
DEFAULT_TASK1_LOCAL_PREFIX="${DEFAULT_DATA_ROOT}/human_eval/videos"
DEFAULT_MEDIA_URL_PREFIX_1="http://localhost:5000/api/media/gestalt_bench/annotation1/"
DEFAULT_MEDIA_URL_PREFIX_2="http://localhost:5000/api/media/gestalt_bench/annotation2/"
DEFAULT_OUTPUT_NAME="clip_statistics"
DEFAULT_FFPROBE="ffprobe"
DEFAULT_STAT_SOURCE="all"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--stat-source full|task1|task2|all] [--full-root PATH] [--task2-root PATH] [--task1-root PATH] [--task1-annotation-dir PATH] [--task1-local-prefix PATH] [--media-url-prefix URL] [--output-name NAME] [--ffprobe PATH]" >&2
    echo "  stat-source: source to summarize; may be passed multiple times (default: ${DEFAULT_STAT_SOURCE})" >&2
    echo "  full-root: GestaltBench root with <dataset>/context/<n>-utt_group (default: ${DEFAULT_FULL_ROOT})" >&2
    echo "  task2-root: task2 manipulation_full/data root (default: ${DEFAULT_TASK2_ROOT})" >&2
    echo "  task1-root: task1 output root (default: ${DEFAULT_TASK1_ROOT})" >&2
    echo "  task1-annotation-dir: folder containing T1*.json annotation files (default: ${DEFAULT_TASK1_ANNOTATION_DIR})" >&2
    echo "  task1-local-prefix: local prefix replacing task1 media URLs (default: ${DEFAULT_TASK1_LOCAL_PREFIX})" >&2
    echo "  media-url-prefix: URL prefix to replace; may be passed multiple times. Defaults include annotation1 and annotation2." >&2
    echo "  output-name: basename for .txt/.json outputs in each output root (default: ${DEFAULT_OUTPUT_NAME})" >&2
    echo "  ffprobe: ffprobe executable inside the container (default: ${DEFAULT_FFPROBE})" >&2
}

FULL_ROOT="${DEFAULT_FULL_ROOT}"
TASK2_ROOT="${DEFAULT_TASK2_ROOT}"
TASK1_ROOT="${DEFAULT_TASK1_ROOT}"
TASK1_ANNOTATION_DIR="${DEFAULT_TASK1_ANNOTATION_DIR}"
TASK1_LOCAL_PREFIX="${DEFAULT_TASK1_LOCAL_PREFIX}"
MEDIA_URL_PREFIXES=("${DEFAULT_MEDIA_URL_PREFIX_1}" "${DEFAULT_MEDIA_URL_PREFIX_2}")
OUTPUT_NAME="${DEFAULT_OUTPUT_NAME}"
FFPROBE="${DEFAULT_FFPROBE}"
STAT_SOURCES=("${DEFAULT_STAT_SOURCE}")

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stat-source)
            if [[ "${#STAT_SOURCES[@]}" -eq 1 && "${STAT_SOURCES[0]}" == "${DEFAULT_STAT_SOURCE}" ]]; then
                STAT_SOURCES=()
            fi
            case "${2:?Missing value for --stat-source}" in
                full|task1|task2|all)
                    STAT_SOURCES+=("$2")
                    ;;
                *)
                    usage
                    echo "Invalid --stat-source: $2. Expected full, task1, task2, or all." >&2
                    exit 1
                    ;;
            esac
            shift 2
            ;;
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
        --task1-annotation-dir)
            TASK1_ANNOTATION_DIR="${2:?Missing value for --task1-annotation-dir}"
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

RUN_FULL=0
RUN_TASK1=0
RUN_TASK2=0
for stat_source in "${STAT_SOURCES[@]}"; do
    case "${stat_source}" in
        all)
            RUN_FULL=1
            RUN_TASK1=1
            RUN_TASK2=1
            ;;
        full)
            RUN_FULL=1
            ;;
        task1)
            RUN_TASK1=1
            ;;
        task2)
            RUN_TASK2=1
            ;;
    esac
done

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "Missing Apptainer image: ${SIF_PATH}" >&2
    echo "Build or copy it from ${PROJECT_ROOT}/apptainer/eyetrack.def first." >&2
    exit 1
fi

if [[ "${RUN_FULL}" == "1" && ! -d "${FULL_ROOT}" ]]; then
    echo "Full root does not exist: ${FULL_ROOT}" >&2
    exit 1
fi

if [[ "${RUN_TASK2}" == "1" && ! -d "${TASK2_ROOT}" ]]; then
    echo "Task2 root does not exist: ${TASK2_ROOT}" >&2
    exit 1
fi

if [[ "${RUN_TASK1}" == "1" && ! -d "${TASK1_ANNOTATION_DIR}" ]]; then
    echo "Task1 annotation directory does not exist: ${TASK1_ANNOTATION_DIR}" >&2
    exit 1
fi

if [[ "${RUN_TASK1}" == "1" && ! -d "${TASK1_LOCAL_PREFIX}" ]]; then
    echo "Task1 local prefix does not exist: ${TASK1_LOCAL_PREFIX}" >&2
    exit 1
fi

mkdir -p /home/nfs/zli33/slurm_outputs/vlm_social
if [[ "${RUN_TASK1}" == "1" ]]; then
    mkdir -p "${TASK1_ROOT}"
fi

echo "Project root:          ${PROJECT_ROOT}"
echo "Apptainer image:       ${SIF_PATH}"
echo "Stat sources:          ${STAT_SOURCES[*]}"
echo "Full root:             ${FULL_ROOT}"
echo "Task2 root:            ${TASK2_ROOT}"
echo "Task1 root:            ${TASK1_ROOT}"
echo "Task1 annotation dir:  ${TASK1_ANNOTATION_DIR}"
echo "Task1 local prefix:    ${TASK1_LOCAL_PREFIX}"
echo "Output name:           ${OUTPUT_NAME}"
echo "ffprobe:               ${FFPROBE}"
echo "Media URL prefixes:    ${MEDIA_URL_PREFIXES[*]}"

PYTHON_ARGS=(
    python /workspace/experiments/analysis/clip_statistics.py
    --full-root "${FULL_ROOT}"
    --task2-root "${TASK2_ROOT}"
    --task1-root "${TASK1_ROOT}"
    --task1-annotation-dir "${TASK1_ANNOTATION_DIR}"
    --task1-local-prefix "${TASK1_LOCAL_PREFIX}"
    --output-name "${OUTPUT_NAME}"
    --ffprobe "${FFPROBE}"
)

for stat_source in "${STAT_SOURCES[@]}"; do
    PYTHON_ARGS+=(--stat-source "${stat_source}")
done

for media_url_prefix in "${MEDIA_URL_PREFIXES[@]}"; do
    PYTHON_ARGS+=(--media-url-prefix "${media_url_prefix}")
done

srun apptainer exec \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /home/nfs/zli33:/home/nfs/zli33 \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    "${SIF_PATH}" \
    "${PYTHON_ARGS[@]}"
