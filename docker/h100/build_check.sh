#!/usr/bin/env bash
# Compile-only check of the DeepGEMM extension inside the H100 image (no GPU needed).
#   bash docker/h100/build_check.sh
set -uo pipefail
IMAGE="${IMAGE:-fastafd-h100}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/fastafd-h100}"
mkdir -p "$CACHE_DIR"
podman run --rm -e MAX_JOBS="${MAX_JOBS:-8}" -v "$CACHE_DIR:/cache" "$IMAGE" \
  python -c 'from minisgl.kernel.deepgemm import _build_extension_path as b; print("BUILT", b())'
