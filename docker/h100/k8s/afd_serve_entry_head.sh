#!/usr/bin/env bash
# Persistent serve entrypoint for the HEAD pod of a 2-pod multi-node AFD deployment.
# Role: attention node (ATTN_DP=1 TP=4 -> 4 GPUs) + Ray head + OpenAI API server.
# The MLP ranks run on the worker pod (afd_entry_worker.sh, unchanged).
#
# Same setup as afd_entry_head.sh (deps, model, userspace RDMA, Ray head, wait for
# the worker, placement exports), but instead of a one-shot alignment run it starts
# `minisgl.server.launch --mode afd-serve` on 0.0.0.0:$SERVE_PORT, waits for
# /v1/models, prints "=== AFD SERVE READY ===", and blocks until the server dies
# (non-zero exit so the pod shows Failed).
#
# Environment variables injected by afd-multinode-serve.yaml:
#   GPU_COUNT, ATTN_DP_SIZE, ATTN_TP_SIZE, MLP_DP_SIZE, MLP_TP_SIZE, AFD_MLP_EP_SIZE,
#   SERVE_PORT, RAY_HEAD_PORT, WORKER_POD_DNS, MODEL_PATH, HF_HOME,
#   MINISGL_MAX_RUNNING_REQUESTS, AFD_DECODE_GRAPH_BS, MINISGL_CUDA_GRAPH_MAX_BS
set -uo pipefail

export HOME=/work/home PATH=/work/bin:$PATH
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDA_13_HOME=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS="${MAX_JOBS:-16}" RAY_TMPDIR=/tmp/ray RAY_DEDUP_LOGS=0
export CONDA_DEFAULT_ENV=minisgl-cuda130
export PYTHONUNBUFFERED=1
# Cross-pod run: GIN (RDMA) must be enabled so nccl.cu takes the full LSA window path.
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-0}"
export EP_BUFFER_DEBUG="${EP_BUFFER_DEBUG:-1}"
# NCCL TCP bootstrap must use the routable pod network (eth0), not a RoCE rail.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
# Default: pin every rank to mlx5_0 (the shared rail NIC on the validation cluster). Override NCCL_IB_HCA per cluster.
export NCCL_IB_HCA="${NCCL_IB_HCA:-=mlx5_0}"
mkdir -p "$HOME" /work/bin /tmp/ray

# Topology / serving knobs
ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-4}"
MLP_DP_SIZE="${MLP_DP_SIZE:-1}"
MLP_TP_SIZE="${MLP_TP_SIZE:-4}"
AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
GPU_COUNT="${GPU_COUNT:-4}"
SERVE_PORT="${SERVE_PORT:-8000}"
MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"

# H100 (SM90) DeepEP + FlashInfer serving defaults (same as afd_serve_entry.sh)
AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-deepep}"
AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-deep_gemm}"
MINISGL_AFD_MOE_BACKEND="${MINISGL_AFD_MOE_BACKEND:-deepep}"
MINISGL_PAGE_SIZE="${MINISGL_PAGE_SIZE:-1}"
AFD_CACHE_TYPE="${AFD_CACHE_TYPE:-naive}"
AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend fi --memory-ratio 0.65}"
MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-64}"
MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-64}"
AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,32,64}"
AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-16384}"
MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-4096}"
AFD_NUM_MB="${AFD_NUM_MB:-1}"
export MINISGL_AFD_MOE_BACKEND
export MINISGL_PAGE_SIZE

fail() { echo "ERROR: $*"; echo "=== AFD SERVE FAIL ==="; exit 1; }

echo "=== GPUs (head/attn node) ==="
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

# --- model download -----------------------------------------------------------
echo "=== downloading $MODEL ==="
"$PY" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))" "$MODEL" || fail "model download"
# Cached now: keep every rank (both pods) off the Hub so none start late.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

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

# --- Ray HEAD -----------------------------------------------------------------
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_CPUS="${RAY_CPUS:-32}"
MY_IP="$(hostname -I | awk '{print $1}')"
echo "=== ray start --head on $MY_IP:$RAY_HEAD_PORT (gpus=$GPU_COUNT cpus=$RAY_CPUS) ==="
ray start --head \
  --num-gpus="$GPU_COUNT" \
  --num-cpus="$RAY_CPUS" \
  --port="$RAY_HEAD_PORT" \
  --object-store-memory=2000000000 \
  --include-dashboard=false \
  --disable-usage-stats \
  || fail "ray start head"

# --- Wait for worker to join --------------------------------------------------
WORKER_DNS="${WORKER_POD_DNS:-}"
[ -n "$WORKER_DNS" ] || fail "WORKER_POD_DNS is not set"
echo "=== waiting for worker ($WORKER_DNS) to join Ray cluster ==="
WAIT_WORKER_S="${WAIT_WORKER_S:-600}"
deadline=$(( SECONDS + WAIT_WORKER_S ))
while true; do
  node_count="$("$PY" -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(sum(1 for n in ray.nodes() if n.get('Alive')))" 2>/dev/null || echo 0)"
  echo "  live Ray nodes: $node_count (need 2)"
  if [ "$node_count" -ge 2 ] 2>/dev/null; then
    echo "  worker joined."
    break
  fi
  if (( SECONDS >= deadline )); then
    fail "worker did not join within ${WAIT_WORKER_S}s"
  fi
  sleep 5
done

# Resolve worker IP from DNS (the headless Service gives pod IP directly)
WORKER_IP="$("$PY" -c "import socket; print(socket.gethostbyname('${WORKER_DNS}'))" 2>/dev/null)" \
  || fail "could not resolve WORKER_POD_DNS=$WORKER_DNS"

# One entry per DP rank; every attention rank is on this pod, every MLP rank on the worker.
nodes_csv() { local ip="$1" n="$2" out="" i; for ((i = 0; i < n; i++)); do out="${out:+$out,}$ip"; done; echo "$out"; }
export MINISGL_AFD_ATTN_NODES="$(nodes_csv "$MY_IP" "$ATTN_DP_SIZE")"
export MINISGL_AFD_MLP_NODES="$(nodes_csv "$WORKER_IP" "$MLP_DP_SIZE")"
echo "=== placement: attn=$MY_IP  mlp=$WORKER_IP ==="

# --- Launch miniSGL AFD API server --------------------------------------------
export MINISGL_LOCAL_CACHE_BASE="${MINISGL_LOCAL_CACHE_BASE:-/work/FastAFD/cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$MINISGL_LOCAL_CACHE_BASE/flashinfer}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/tvm_ffi}"
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_jit}"
export N2M_M2N_GIN_BUILD_DIR="${N2M_M2N_GIN_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/gin_comm}"
export MINISGL_DEEPEP_BUILD_DIR="${MINISGL_DEEPEP_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_moe}"
export MINISGL_DEEPGEMM_BUILD_DIR="${MINISGL_DEEPGEMM_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepgemm}"
export CUDA_HOME CUDA_PATH CUDA_NVCC_EXECUTABLE

SERVE_LOG=/work/afd_serve.log
echo "=== starting AFD serve (attn dp${ATTN_DP_SIZE}/tp${ATTN_TP_SIZE} on $MY_IP, mlp dp${MLP_DP_SIZE}/tp${MLP_TP_SIZE}/ep${AFD_MLP_EP_SIZE} on $WORKER_IP, port ${SERVE_PORT}) ==="
RAY_DEDUP_LOGS=0 python -m minisgl.server.launch \
  --mode afd-serve \
  --model "$MODEL" \
  --host 0.0.0.0 \
  --port "$SERVE_PORT" \
  --launch-backend ray \
  --ray-address auto \
  --afd-attn-dp-size "$ATTN_DP_SIZE" \
  --afd-mlp-dp-size  "$MLP_DP_SIZE" \
  --afd-attn-tp-size "$ATTN_TP_SIZE" \
  --afd-mlp-tp-size  "$MLP_TP_SIZE" \
  --afd-mlp-ep-size  "$AFD_MLP_EP_SIZE" \
  --afd-moe-a2a-backend "$AFD_MOE_A2A_BACKEND" \
  --afd-moe-runner-backend "$AFD_MOE_RUNNER_BACKEND" \
  --afd-max-running-requests "$MINISGL_MAX_RUNNING_REQUESTS" \
  --afd-max-batched-tokens   "$AFD_MAX_BATCHED_TOKENS" \
  --afd-decode-graph-bs      "$AFD_DECODE_GRAPH_BS" \
  --afd-num-mb               "$AFD_NUM_MB" \
  --cache-type               "$AFD_CACHE_TYPE" \
  --page-size                "$MINISGL_PAGE_SIZE" \
  --cuda-graph-max-bs        "$MINISGL_CUDA_GRAPH_MAX_BS" \
  --max-seq-len-override     "$MINISGL_MAX_SEQ_LEN" \
  $AFD_SERVER_EXTRA_ARGS \
  >"$SERVE_LOG" 2>&1 &
AFD_PID=$!
echo "  afd_pid: $AFD_PID"
echo "  afd_log: $SERVE_LOG"

# --- Wait for readiness, then block -------------------------------------------
# Both pods JIT-build kernels independently, so allow a long cold start.
WAIT_READY_TIMEOUT="${WAIT_READY_TIMEOUT:-3600}"
deadline=$((SECONDS + WAIT_READY_TIMEOUT))
until "$PY" - "http://127.0.0.1:${SERVE_PORT}/v1/models" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(sys.argv[1], timeout=5).read()
PY
do
  if ! kill -0 "$AFD_PID" 2>/dev/null; then
    echo "AFD server exited before becoming ready." >&2
    tail -n 200 "$SERVE_LOG" >&2 || true
    fail "server exited during startup"
  fi
  if (( SECONDS >= deadline )); then
    echo "AFD server failed to become ready within ${WAIT_READY_TIMEOUT}s." >&2
    tail -n 200 "$SERVE_LOG" >&2 || true
    fail "server startup timeout"
  fi
  sleep 3
done

echo "=== AFD SERVE READY === http://0.0.0.0:${SERVE_PORT}/v1"

# Block until the server dies; exit non-zero so the pod enters Failed.
while kill -0 "$AFD_PID" 2>/dev/null; do
  sleep 10
done
rc=0
wait "$AFD_PID" 2>/dev/null || rc=$?
echo "AFD server exited (rc=$rc)." >&2
tail -n 100 "$SERVE_LOG" >&2 || true
# A serve process should never exit on its own; report failure even for rc=0.
exit "$(( rc == 0 ? 1 : rc ))"
