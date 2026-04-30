#!/usr/bin/env bash
set -euo pipefail

# Run from repo root:
#   bash apptainer/build.sh
# Or from the apptainer/ directory:
#   bash build.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="$ROOT_DIR/apptainer"
DEF_FILE="$APP_DIR/Ming-lite-omni.def"
SIF_FILE="$APP_DIR/ming-lite-omni.sif"

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not installed or not in PATH"
    exit 1
fi

echo "Building $SIF_FILE from $DEF_FILE"
BUILD_FLAGS=()
if [ "$(id -u)" -ne 0 ]; then
    BUILD_FLAGS+=(--fakeroot)
fi

apptainer build "${BUILD_FLAGS[@]}" "$SIF_FILE" "$DEF_FILE"

echo "Build completed: $SIF_FILE"
