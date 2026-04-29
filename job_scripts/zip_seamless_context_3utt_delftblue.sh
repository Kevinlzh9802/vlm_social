#!/bin/bash
#SBATCH --job-name="zip_3utt_ctx"
#SBATCH --time=10:00:00
#SBATCH --partition=compute-p1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000M
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=logs/zip_seamless_context_3utt_delftblue_%j.out
#SBATCH --error=logs/zip_seamless_context_3utt_delftblue_%j.err
# Submit from the repository root; ensure logs/ exists before sbatch.

set -euo pipefail

DEFAULT_SOURCE_DIR="${DATA_ROOT:-/path/to/data/gestalt_bench}/seamless_interaction/context/3-utt_group"
DEFAULT_OUTPUT_ZIP="${DATA_ROOT:-/path/to/data/gestalt_bench}/seamless_interaction/context/3-utt_group.zip"

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 [--source-dir PATH] [--output-zip PATH] [--overwrite]" >&2
    echo "" >&2
    echo "Defaults:" >&2
    echo "  source-dir: ${DEFAULT_SOURCE_DIR}" >&2
    echo "  output-zip: ${DEFAULT_OUTPUT_ZIP}" >&2
}

SOURCE_DIR="${DEFAULT_SOURCE_DIR}"
OUTPUT_ZIP="${DEFAULT_OUTPUT_ZIP}"
OVERWRITE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)
            SOURCE_DIR="${2:?Missing value for --source-dir}"
            shift 2
            ;;
        --output-zip)
            OUTPUT_ZIP="${2:?Missing value for --output-zip}"
            shift 2
            ;;
        --overwrite)
            OVERWRITE=1
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

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "Source directory does not exist: ${SOURCE_DIR}" >&2
    exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
    echo "'zip' command not found on the compute node." >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ZIP}")"

if [[ -e "${OUTPUT_ZIP}" ]]; then
    if [[ "${OVERWRITE}" -eq 1 ]]; then
        rm -f "${OUTPUT_ZIP}"
    else
        echo "Output zip already exists: ${OUTPUT_ZIP}" >&2
        echo "Pass --overwrite to replace it." >&2
        exit 1
    fi
fi

SOURCE_PARENT="$(dirname "${SOURCE_DIR}")"
SOURCE_NAME="$(basename "${SOURCE_DIR}")"

echo "Source dir: ${SOURCE_DIR}"
echo "Output zip: ${OUTPUT_ZIP}"
echo "Working dir: ${SOURCE_PARENT}"

cd "${SOURCE_PARENT}"
zip -r "${OUTPUT_ZIP}" "${SOURCE_NAME}"

echo "Created zip archive: ${OUTPUT_ZIP}"
