#!/usr/bin/env bash
# Runs inside docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 as an arbitrary
# non-root UID (OpenShift restricted SCC). Init containers already placed the
# FastAFD checkout in /work/FastAFD and a static uv binary in /work/bin.
#
# 1. installs Python 3.12 + the environment.cuda130.yml pip set with uv
# 2. adds ps / pgrep / curl shims (the image has none; the launch driver needs them)
# 3. pre-downloads the model into the PVC-backed HF cache
# 4. starts a single-node Ray head and runs the H100 AFD launch script
#
# Topology comes from the pod env (ATTN_DP_SIZE, ATTN_TP_SIZE, MLP_DP_SIZE,
# MLP_TP_SIZE, AFD_MLP_EP_SIZE). Prints "=== AFD RUN PASS ===" / "=== AFD RUN FAIL ===".
set -uo pipefail

export HOME=/work/home PATH=/work/bin:$PATH
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda CUDA_13_HOME=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS="${MAX_JOBS:-16}" RAY_TMPDIR=/tmp/ray RAY_DEDUP_LOGS=0
export CONDA_DEFAULT_ENV=minisgl-cuda130   # satisfies the driver's env-name check
export PYTHONUNBUFFERED=1
# One node, all 8 GPUs on NVLink/NVSwitch: NCCL GIN is available and required.
# EP_DISABLE_GIN=0 (default) lets DeepEP use the full NVLink symmetric-memory path.
# Override to 1 only when testing without RDMA (e.g. single-GPU smoke tests).
export EP_DISABLE_GIN="${EP_DISABLE_GIN:-0}"
mkdir -p "$HOME" /work/bin /tmp/ray

fail() { echo "ERROR: $*"; echo "=== AFD RUN FAIL ==="; exit 1; }

echo "=== GPUs ==="
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
if [ "${RUN_VLLM_ALIGNMENT:-0}" = "1" ]; then
  echo "=== installing vLLM 0.19.0+cu130 (alignment) ==="
  uv pip install --python "$PY" --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
    || fail "vllm install"
fi
uv pip install --python "$PY" --reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" "setuptools==80.10.2" "fsspec==2026.2.0" \
  || fail "pip overrides"
export PATH=/work/venv/bin:/work/bin:$PATH

cd /work/FastAFD || fail "missing checkout"
uv pip install --python "$PY" --no-deps -e . || fail "pip install -e"
export PYTHONPATH=/work/FastAFD/python
echo "commit: $(cat .commit 2>/dev/null || echo unknown)"

# --- shims for tools missing from the CUDA image -----------------------------
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
    if "-p" in argv:                      # ps -o stat= -p PID
        pid = argv[argv.index("-p") + 1]
        info = state_and_ppid(pid)
        if info is None:
            return 1
        print(info[0])
        return 0
    return 0

def pgrep(argv):
    if "-P" not in argv:                  # only `pgrep -P PID` is used
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

# --- model: warm the PVC-backed HF cache once, serially ----------------------
MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
echo "=== downloading $MODEL to HF_HOME=${HF_HOME:-<default>} ==="
"$PY" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))" "$MODEL" || fail "model download"

# --- Ray head + AFD run -------------------------------------------------------
# nproc reports the whole node (e.g. 224). Ray prestarts ~1 worker per CPU, and
# that exhausts the node's thread limit (pthread_create EAGAIN kills the raylet),
# so advertise only a small CPU count; each AFD rank actor needs 1 CPU.
RAY_CPUS="${RAY_CPUS:-32}"
echo "=== ray start (gpus=$GPU_COUNT cpus=$RAY_CPUS) ==="
ray start --head --num-gpus="$GPU_COUNT" --num-cpus="$RAY_CPUS" \
  --object-store-memory=2000000000 --include-dashboard=false --disable-usage-stats \
  || fail "ray start"

export CLEAN_FIRST=0 CLEAN_AFTER=0 RUN_VLLM_ALIGNMENT="${RUN_VLLM_ALIGNMENT:-0}"
export PROMPT_KIND="${PROMPT_KIND:-hello}"
export MODEL_PATH="$MODEL"

echo "=== running H100 AFD launch script (attn dp${ATTN_DP_SIZE}/tp${ATTN_TP_SIZE}, mlp dp${MLP_DP_SIZE}/tp${MLP_TP_SIZE}/ep${AFD_MLP_EP_SIZE}) ==="
bash scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_h100_1node_deepep_alignment.sh
rc=$?

echo "=== key log lines ==="
RUN_DIR="$(ls -dt reports/afd_* 2>/dev/null | head -1)"
echo "run_dir=$RUN_DIR"
if [ -n "$RUN_DIR" ]; then
  grep -rhE "new_afd_deepep|afd_deepep_adapters|deepep_extension:(build_done|init_jit_done)|afd_(eg|ag)_decode_graph|graph (capture|replay)|sm_90a|Traceback|Error" \
    "$RUN_DIR"/afd.log "$RUN_DIR"/ray_logs 2>/dev/null | cut -c1-300 | sort -u | head -80
  echo "--- sample.log tail ---"; tail -n 30 "$RUN_DIR/sample.log" 2>/dev/null
  echo "--- afd.log tail ---"; tail -n 60 "$RUN_DIR/afd.log" 2>/dev/null | cut -c1-300
fi
[ "$rc" = "0" ] && echo "=== AFD RUN PASS ===" || echo "=== AFD RUN FAIL (rc=$rc) ==="

KEEP="${KEEP_ALIVE_S:-600}"
echo "=== keeping pod alive ${KEEP}s for kubectl cp/exec ==="
sleep "$KEEP"
exit "$rc"
