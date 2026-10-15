#!/usr/bin/env bash
# Persistent AFD serve entrypoint for the fastafd-qwen30b pod.
# Runs inside docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 as an arbitrary
# non-root UID (OpenShift restricted SCC). Init containers already placed the
# FastAFD checkout in /work/FastAFD and a static uv binary in /work/bin.
#
# 1. Installs Python 3.12 + the environment.cuda130.yml pip set with uv
# 2. Pre-downloads the model into the PVC-backed HF cache
# 3. Starts a single-node Ray head
# 4. Launches the miniSGL AFD API server (--mode afd-serve, 0.0.0.0:$SERVE_PORT)
# 5. Waits until /v1/models returns 200, prints "=== AFD SERVE READY ===",
#    then blocks; exits non-zero if the server dies unexpectedly.
#
# Topology is controlled via environment variables (set in the Pod manifest):
#   ATTN_DP_SIZE  (default 1)   ATTN_TP_SIZE  (default 4)
#   MLP_DP_SIZE   (default 4)   AFD_MLP_EP_SIZE (default 4)
#   GPU_COUNT     (default 8)
#   SERVE_PORT    (default 8000)
#   RAY_CPUS      (default 32)
#   EP_DISABLE_GIN (default 1)  -- no NCCL GIN needed on single-node NVLink
set -uo pipefail

export HOME=/work/home PATH=/work/bin:$PATH
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDA_13_HOME=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS="${MAX_JOBS:-16}" RAY_TMPDIR=/tmp/ray RAY_DEDUP_LOGS=0
export CONDA_DEFAULT_ENV=minisgl-cuda130   # satisfies the driver's env-name check
export PYTHONUNBUFFERED=1
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-1}"
mkdir -p "$HOME" /work/bin /tmp/ray

# Topology / serving knobs
ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
ATTN_TP_SIZE="${ATTN_TP_SIZE:-4}"
MLP_DP_SIZE="${MLP_DP_SIZE:-4}"
MLP_TP_SIZE="${MLP_TP_SIZE:-1}"
AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
GPU_COUNT="${GPU_COUNT:-8}"
SERVE_PORT="${SERVE_PORT:-8000}"
RAY_CPUS="${RAY_CPUS:-32}"
MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"

# H100 (SM90) DeepEP + FlashInfer serving defaults
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

echo "=== GPUs ==="
nvidia-smi -L || fail "no GPU visible"

# ---------------------------------------------------------------------------
# 1. Install Python + deps
# ---------------------------------------------------------------------------
echo "=== env setup ==="
uv python install 3.12 || fail "uv python install"
# Skip venv creation if it already exists (pod restart with warm emptyDir)
if [[ ! -x /work/venv/bin/python ]]; then
  uv venv /work/venv --python 3.12 || fail "uv venv"
fi
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

# --- shims for tools missing from the CUDA image ----------------------------
cat > /work/bin/_procshim.py <<'EOF'
import os, sys, urllib.request

def state_and_ppid(pid):
    try:
        data = open(f"/proc/{pid}/stat").read()
    except OSError:
        return None
    rest = data.rpartition(")")[2].split()
    return rest[0], int(rest[1])

def ps(argv):
    if "-p" in argv:
        pid = argv[argv.index("-p") + 1]
        info = state_and_ppid(pid)
        if info is None:
            return 1
        print(info[0])
        return 0
    return 0

def pgrep(argv):
    if "-P" not in argv:
        return 1
    parent = int(argv[argv.index("-P") + 1])
    found = False
    for name in os.listdir("/proc"):
        if name.isdigit():
            info = state_and_ppid(name)
            if info and info[1] == parent:
                print(name)
                found = True
    return 0 if found else 1

def curl(argv):
    urls = [a for a in argv if a.startswith("http")]
    if not urls:
        return 2
    try:
        urllib.request.urlopen(urls[0], timeout=10).read()
        return 0
    except Exception:
        return 22

if __name__ == "__main__":
    tool = os.path.basename(sys.argv[1])
    sys.exit({"ps": ps, "pgrep": pgrep, "curl": curl}[tool](sys.argv[2:]))
EOF
for t in ps pgrep curl; do
  printf '#!/bin/sh\nexec /work/venv/bin/python /work/bin/_procshim.py %s "$@"\n' "$t" > "/work/bin/$t"
  chmod +x "/work/bin/$t"
done

# ---------------------------------------------------------------------------
# 2. Download model (cached on PVC)
# ---------------------------------------------------------------------------
echo "=== downloading $MODEL to HF_HOME=${HF_HOME:-<default>} ==="
"$PY" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))" "$MODEL" || fail "model download"

# ---------------------------------------------------------------------------
# 3. Ray head
# ---------------------------------------------------------------------------
echo "=== ray start (gpus=$GPU_COUNT cpus=$RAY_CPUS) ==="
ray start --head --num-gpus="$GPU_COUNT" --num-cpus="$RAY_CPUS" \
  --object-store-memory=2000000000 --include-dashboard=false --disable-usage-stats \
  || fail "ray start"

# ---------------------------------------------------------------------------
# 4. Launch miniSGL AFD API server (persistent serve, no benchmark loop)
# ---------------------------------------------------------------------------
export MINISGL_LOCAL_CACHE_BASE="${MINISGL_LOCAL_CACHE_BASE:-/work/FastAFD/cache}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$MINISGL_LOCAL_CACHE_BASE/flashinfer}"
export TVM_FFI_CACHE_DIR="${TVM_FFI_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/tvm_ffi}"
export EP_JIT_CACHE_DIR="${EP_JIT_CACHE_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_jit}"
export N2M_M2N_GIN_BUILD_DIR="${N2M_M2N_GIN_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/gin_comm}"
export MINISGL_DEEPEP_BUILD_DIR="${MINISGL_DEEPEP_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepep_moe}"
export MINISGL_DEEPGEMM_BUILD_DIR="${MINISGL_DEEPGEMM_BUILD_DIR:-$MINISGL_LOCAL_CACHE_BASE/deepgemm}"
export CUDA_HOME CUDA_PATH CUDA_NVCC_EXECUTABLE

SERVE_LOG=/work/afd_serve.log
mkdir -p /work

echo "=== starting AFD serve (attn dp${ATTN_DP_SIZE}/tp${ATTN_TP_SIZE}, mlp dp${MLP_DP_SIZE}/tp${MLP_TP_SIZE}/ep${AFD_MLP_EP_SIZE}, port ${SERVE_PORT}) ==="
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

# ---------------------------------------------------------------------------
# 5. Wait for readiness, then block
# ---------------------------------------------------------------------------
WAIT_READY_TIMEOUT="${WAIT_READY_TIMEOUT:-1800}"
deadline=$((SECONDS + WAIT_READY_TIMEOUT))
until python - "http://127.0.0.1:${SERVE_PORT}/v1/models" <<'PY' 2>/dev/null
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

# Block until the server dies; exit non-zero so the pod enters Failed/restarts.
while kill -0 "$AFD_PID" 2>/dev/null; do
  sleep 10
done
wait "$AFD_PID" 2>/dev/null
rc=$?
echo "AFD server exited (rc=$rc)." >&2
tail -n 100 "$SERVE_LOG" >&2 || true
exit "$rc"
