#!/usr/bin/env bash
set -euo pipefail

# Usage examples:
#   bash apptainer/run_infer.sh python test_infer.py
#   bash apptainer/run_infer.sh python test_infer_gen_image.py
#   bash apptainer/run_infer.sh python test_audio_tasks.py

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_FILE="$ROOT_DIR/apptainer/ming-lite-omni.sif"

if [ ! -f "$SIF_FILE" ]; then
    echo "SIF not found: $SIF_FILE"
    echo "Build it first: bash apptainer/build.sh"
    exit 1
fi

if [ "$#" -eq 0 ]; then
    echo "No command provided. Example:"
    echo "  bash apptainer/run_infer.sh python test_infer.py"
    exit 1
fi

apptainer exec --nv \
    --bind "$ROOT_DIR":/workspace \
    --pwd /workspace \
    "$SIF_FILE" \
    bash -c '
        cuda_stub_dir=$(mktemp -d)
        real_libcuda=$(ldconfig -p 2>/dev/null | grep "libcuda.so.1 " | head -1 | sed "s/.*=> //")
        if [ -z "$real_libcuda" ]; then
            real_libcuda=$(find /.singularity.d/libs -name "libcuda.so.1" 2>/dev/null | head -1)
        fi
        if [ -n "$real_libcuda" ]; then
            ln -s "$real_libcuda" "$cuda_stub_dir/libcuda.so"
        fi
        export LD_LIBRARY_PATH="$cuda_stub_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        exec "$@"
        rm -rf "$cuda_stub_dir"
    ' _ "$@"
