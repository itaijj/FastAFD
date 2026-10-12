#!/usr/bin/env bash
# Runs inside docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 as an arbitrary
# non-root UID (OpenShift restricted SCC). Init containers have already put
# the FastAFD checkout in /work/FastAFD and a static uv binary in /work/bin.
# Installs Python 3.12 + the environment.cuda130.yml pip set with uv, then runs
# the SM90 expert-path test. Prints "=== TEST PASS ===" / "=== TEST FAIL ===".
set -uo pipefail

export HOME=/work/home PATH=/work/bin:$PATH
export UV_CACHE_DIR=/work/uv-cache UV_PYTHON_INSTALL_DIR=/work/python
export CUDA_HOME=/usr/local/cuda CUDA_PATH=/usr/local/cuda
export CUDA_NVCC_EXECUTABLE=/usr/local/cuda/bin/nvcc TORCH_CUDA_ARCH_LIST=9.0a
export MINISGL_LOCAL_CACHE_BASE=/work/cache MINISGL_DEEPGEMM_BUILD_DIR=/work/cache/deepgemm
export TVM_FFI_CACHE_DIR=/work/cache/tvm_ffi MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$HOME" /work/cache

fail() { echo "ERROR: $*"; echo "=== TEST FAIL ==="; exit 1; }

echo "=== env setup ==="
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv || fail "no GPU visible"
uv python install 3.12 || fail "uv python install"
uv venv /work/venv --python 3.12 || fail "uv venv"
PY=/work/venv/bin/python
uv pip install --python "$PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.10.0+cu130" "triton==3.6.0" "nvidia-nccl-cu13==2.28.9" \
  "flashinfer-python==0.6.6" "apache-tvm-ffi==0.1.9" "transformers==4.57.3" \
  "setuptools>=75" wheel ninja "pybind11>=2.13" pyzmq msgpack "pydantic>=2" \
  || fail "pip install"
uv pip install --python "$PY" --reinstall --no-deps \
  "nvidia-nccl-cu13==2.30.4" "setuptools==80.10.2" "fsspec==2026.2.0" \
  || fail "pip overrides"
export PATH=/work/venv/bin:$PATH

cd /work/FastAFD || fail "missing checkout"
echo "commit: $(cat /work/FastAFD/.commit 2>/dev/null || echo unknown)"
"$PY" -c 'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "cap", torch.cuda.get_device_capability())' \
  || fail "torch import"

echo "=== running scripts/validate/test_h100_deepgemm_psum_moe.py ==="
"$PY" scripts/validate/test_h100_deepgemm_psum_moe.py
rc=$?
[ "$rc" = "0" ] && echo "=== TEST PASS ===" || echo "=== TEST FAIL (rc=$rc) ==="
exit "$rc"
