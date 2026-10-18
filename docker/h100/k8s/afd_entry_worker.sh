#!/usr/bin/env bash
# Runs on the WORKER pod of a 2-pod multi-node AFD run.
# Role: MLP-only node (MLP_DP=1 TP=4 EP=4 → 4 GPUs).
#
# Steps:
#  1. Install Python 3.12 + deps (identical to head)
#  2. Wait until the head's Ray GCS port is reachable
#  3. Join the head's Ray cluster as a worker
#  4. Sleep until Ray exits (head pod finished / deleted)
#
# The head pod drives the actual AFD run; this pod just contributes GPUs to
# the shared Ray cluster. It must NOT start its own ray head or run minisgl.
#
# Environment variables injected by afd-multinode.yaml:
#   GPU_COUNT, RAY_HEAD_PORT, HEAD_POD_DNS, KEEP_ALIVE_S
set -uo pipefail

export HOME=/work/home PATH=/work/bin:$PATH
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDA_13_HOME=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS="${MAX_JOBS:-16}" RAY_TMPDIR=/tmp/ray RAY_DEDUP_LOGS=0
export CONDA_DEFAULT_ENV=minisgl-cuda130
export PYTHONUNBUFFERED=1
# Cross-pod run: GIN (RDMA) must be enabled so nccl.cu takes the full LSA window path.
# EP_DISABLE_GIN=0 lets gin_enabled=true when NCCL sees ginType != NONE on the RoCE NIC.
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-0}"
export EP_BUFFER_DEBUG="${EP_BUFFER_DEBUG:-1}"
# NCCL TCP bootstrap must use the routable pod network (eth0), not a RoCE rail.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
# Default: pin every rank to mlx5_0 (the shared rail NIC on the validation cluster). Override NCCL_IB_HCA per cluster.
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0}"
mkdir -p "$HOME" /work/bin /tmp/ray

fail() { echo "ERROR: $*"; echo "=== WORKER FAIL ==="; exit 1; }

echo "=== GPUs (worker/mlp node) ==="
nvidia-smi -L || fail "no GPU visible"

echo "=== env setup ==="
uv python install 3.12 || fail "uv python install"
uv venv /work/venv --python 3.12 || fail "uv venv"
PY=/work/venv/bin/python
uv pip install --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.10.0+cu130" "triton==3.6.0" "nvidia-nccl-cu13==2.28.9" \
  "flashinfer-python==0.6.6" "ray==2.54.0" "apache-tvm-ffi==0.1.9" \
  "transformers==4.57.3" huggingface_hub accelerate modelscope "pybind11>=2.13" \
  "setuptools>=75" wheel ninja fastapi uvicorn pyzmq msgpack "pydantic>=2" \
  prompt_toolkit openai \
  || fail "pip install"
uv pip install --python "$PY" --reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" "setuptools==80.10.2" "fsspec==2026.2.0" \
  || fail "pip overrides"
export PATH=/work/venv/bin:/work/bin:$PATH

cd /work/FastAFD || fail "missing checkout"
uv pip install --python "$PY" --no-deps -e . || fail "pip install -e"
export PYTHONPATH=/work/FastAFD/python
echo "commit: $(cat .commit 2>/dev/null || echo unknown)"

# --- userspace RDMA (libibverbs + mlx5 provider) -------------------------------
# The CUDA base image has no rdma-core, so NCCL cannot load its IB/RoCE plugin,
# falls back to TCP sockets, finds no GIN, and DeepEP fails cross-node with
# "Communicator does not support symmetric memory". The pod runs as non-root, so
# unpack the Ubuntu 24.04 packages under /work/rdma instead of apt-get install.
install_userspace_rdma() {
  local root=/work/rdma lib=/work/rdma/usr/lib/x86_64-linux-gnu
  local ubuntu="${RDMA_DEB_MIRROR:-http://archive.ubuntu.com/ubuntu/pool/main}"
  local rdma_ver="${RDMA_CORE_DEB_VERSION:-50.0-2ubuntu0.2}" nl_ver="${LIBNL_DEB_VERSION:-3.7.0-0.3build1.1}"
  mkdir -p "$root/debs"
  local url
  for url in \
      "$ubuntu/r/rdma-core/libibverbs1_${rdma_ver}_amd64.deb" \
      "$ubuntu/r/rdma-core/ibverbs-providers_${rdma_ver}_amd64.deb" \
      "$ubuntu/libn/libnl3/libnl-3-200_${nl_ver}_amd64.deb" \
      "$ubuntu/libn/libnl3/libnl-route-3-200_${nl_ver}_amd64.deb"; do
    "$PY" -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
      "$url" "$root/debs/$(basename "$url")" || return 1
    dpkg-deb -x "$root/debs/$(basename "$url")" "$root" || return 1
  done
  if LD_LIBRARY_PATH="$lib" ldd "$lib/libibverbs.so.1" "$lib/libibverbs/libmlx5-rdmav34.so" | grep -q "not found"; then
    return 1
  fi
  # libibverbs falls back to dlopen("libmlx5-rdmav34.so") via LD_LIBRARY_PATH.
  export LD_LIBRARY_PATH="$lib:$lib/libibverbs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export RDMAV_DRIVERS="${RDMAV_DRIVERS:-mlx5}"
}
echo "=== installing userspace RDMA libraries (/work/rdma) ==="
install_userspace_rdma || fail "userspace RDMA install"

# --- wait for head Ray GCS to be reachable ------------------------------------
HEAD_DNS="${HEAD_POD_DNS:-}"
[ -n "$HEAD_DNS" ] || fail "HEAD_POD_DNS is not set"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"

echo "=== waiting for Ray head at $HEAD_DNS:$RAY_HEAD_PORT ==="
WAIT_HEAD_S="${WAIT_HEAD_S:-600}"
deadline=$(( SECONDS + WAIT_HEAD_S ))
while ! "$PY" -c "
import socket, sys
try:
    s = socket.create_connection(('${HEAD_DNS}', ${RAY_HEAD_PORT}), timeout=3)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    fail "head $HEAD_DNS:$RAY_HEAD_PORT not reachable within ${WAIT_HEAD_S}s"
  fi
  echo "  head not yet reachable — retrying in 5s"
  sleep 5
done
echo "  head is reachable."

# The head downloads the model to the shared PVC before it starts Ray, so the
# cache is complete here; keep the MLP ranks off the Hub.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# --- join Ray cluster as worker -----------------------------------------------
RAY_CPUS="${RAY_CPUS:-32}"
GPU_COUNT="${GPU_COUNT:-4}"
HEAD_IP="$("$PY" -c "import socket; print(socket.gethostbyname('${HEAD_DNS}'))" 2>/dev/null)" \
  || fail "could not resolve HEAD_POD_DNS=$HEAD_DNS"
echo "=== ray start --worker joining $HEAD_IP:$RAY_HEAD_PORT (gpus=$GPU_COUNT cpus=$RAY_CPUS) ==="
ray start \
  --address="$HEAD_IP:$RAY_HEAD_PORT" \
  --num-gpus="$GPU_COUNT" \
  --num-cpus="$RAY_CPUS" \
  --object-store-memory=2000000000 \
  --disable-usage-stats \
  || fail "ray start worker"

echo "=== worker joined Ray cluster — waiting for head to finish ==="
# Poll until the Ray head GCS port goes away (head pod exited) or KEEP_ALIVE_S expires.
KEEP="${KEEP_ALIVE_S:-900}"
deadline=$(( SECONDS + KEEP ))
while (( SECONDS < deadline )); do
  if ! "$PY" -c "
import socket, sys
try:
    s = socket.create_connection(('${HEAD_DNS}', ${RAY_HEAD_PORT}), timeout=3)
    s.close()
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    echo "=== head GCS gone — worker exiting ==="
    break
  fi
  sleep 15
done

echo "=== worker pod done ==="
ray stop --force 2>/dev/null || true
exit 0
