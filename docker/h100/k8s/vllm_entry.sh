#!/usr/bin/env bash
# Runs inside docker.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 as an arbitrary
# non-root UID (OpenShift restricted SCC). The fetch-uv init container already
# placed a static uv binary in /work/bin.
#
# 1. installs Python 3.12 + torch 2.10.0+cu130 + vLLM 0.19.0+cu130 with uv
# 2. warms the PVC-backed HF cache with the model
# 3. starts `vllm serve` (Triton MoE, eager) and waits until it answers
# 4. sends a chat completion and a logprobs request as a self-test
#
# Prints "=== VLLM RUN PASS ===" / "=== VLLM RUN FAIL ===", then keeps serving
# for KEEP_ALIVE_S so you can `kubectl port-forward pod/<name> 8000:8000`.
set -uo pipefail

export HOME=/work/home PATH=/work/bin:$PATH
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export TRITON_CACHE_DIR=/work/triton VLLM_CACHE_ROOT=/work/vllm XDG_CACHE_HOME=/work/cache
export VLLM_USE_FLASHINFER_MOE_FP8=0 VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
export PYTHONUNBUFFERED=1
mkdir -p "$HOME" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TP_SIZE:-1}"
MAXLEN="${VLLM_MAX_MODEL_LEN:-4096}"
EXTRA="${VLLM_EXTRA_ARGS:---enforce-eager --moe-backend triton}"

fail() { echo "ERROR: $*"; echo "=== VLLM RUN FAIL ==="; exit 1; }

echo "=== GPUs ==="
nvidia-smi -L || fail "no GPU visible"

echo "=== env setup ==="
uv python install 3.12 || fail "uv python install"
uv venv /work/venv --python 3.12 || fail "uv venv"
PY=/work/venv/bin/python
uv pip install --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.10.0+cu130" "triton==3.6.0" huggingface_hub setuptools wheel ninja \
  "https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu130-cp38-abi3-manylinux_2_35_x86_64.whl" \
  || fail "pip install"
uv pip install --python "$PY" --reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" "setuptools==80.10.2" "fsspec==2026.2.0" \
  || fail "pip overrides"
export PATH=/work/venv/bin:/work/bin:$PATH
# vLLM/Triton JIT needs a C compiler and the CUDA headers; both ship in the image.
echo "vllm: $("$PY" -c 'import vllm; print(vllm.__version__)')"

echo "=== downloading $MODEL to HF_HOME=${HF_HOME:-<default>} ==="
"$PY" -c "
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1]))" "$MODEL" || fail "model download"

echo "=== vllm serve $MODEL (tp=$TP port=$PORT) ==="
# shellcheck disable=SC2086
vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" $EXTRA \
  > /work/vllm_server.log 2>&1 &
VPID=$!
tail -n +1 -F /work/vllm_server.log &   # stream the server log into `kubectl logs`

# The CUDA image has no curl, so probe with the venv's Python.
ready() {
  "$PY" -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=5)" \
    "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}
deadline=$((SECONDS + ${VLLM_WAIT_READY_TIMEOUT:-1800}))
until ready; do
  kill -0 "$VPID" 2>/dev/null || fail "vllm exited before ready"
  [ "$SECONDS" -lt "$deadline" ] || fail "vllm not ready before timeout"
  sleep 5
done
echo "vLLM ready on :$PORT"

echo "=== self-test ==="
rc=0
"$PY" - "$PORT" "$MODEL" <<'EOF' || rc=$?
import json, sys, urllib.request

port, model = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}/v1"

def post(path, body):
    req = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=300))

chat = post("/chat/completions", {
    "model": model,
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "max_tokens": 16, "temperature": 0,
})
text = chat["choices"][0]["message"]["content"]
print("chat reply:", repr(text))

comp = post("/completions", {
    "model": model, "prompt": "The capital of France is", "max_tokens": 4,
    "temperature": 0, "logprobs": 5,
})
choice = comp["choices"][0]
print("completion:", repr(choice["text"]), "top logprobs:", choice["logprobs"]["top_logprobs"][0])

# Qwen3 starts its chat reply with a <think> block, so 16 tokens never reach the
# answer: only require a non-empty chat reply, and check correctness on the greedy
# completion, whose first top logprob must be " Paris".
assert text.strip(), "empty chat reply"
top = choice["logprobs"]["top_logprobs"][0]
assert max(top, key=top.get).strip() == "Paris", f"unexpected top token: {top}"
print("self-test OK")
EOF
[ "$rc" = "0" ] && echo "=== VLLM RUN PASS ===" || echo "=== VLLM RUN FAIL (rc=$rc) ==="

KEEP="${KEEP_ALIVE_S:-1800}"
echo "=== serving for ${KEEP}s; use: kubectl port-forward pod/\$POD 8000:$PORT ==="
sleep "$KEEP"
kill "$VPID" 2>/dev/null
exit "$rc"
