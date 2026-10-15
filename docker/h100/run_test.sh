#!/usr/bin/env bash
# Build the H100 test image with podman and run the SM90 expert-path test.
#   GPU=0 bash docker/h100/run_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-fastafd-h100}"
GPU="${GPU:-0}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/fastafd-h100}"

if ! podman info --format '{{.Host.CDIDevices}}' 2>/dev/null | grep -q nvidia.com/gpu \
    && [[ ! -e /etc/cdi/nvidia.yaml && ! -e /var/run/cdi/nvidia.yaml ]]; then
  echo "NVIDIA CDI spec not found. Run: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml" >&2
  exit 1
fi

mkdir -p "$CACHE_DIR"
podman build -t "$IMAGE" -f "$REPO_ROOT/docker/h100/Containerfile" "$REPO_ROOT"
exec podman run --rm \
  --device "nvidia.com/gpu=$GPU" \
  --security-opt=label=disable \
  -v "$CACHE_DIR:/cache" \
  "$IMAGE" "$@"
