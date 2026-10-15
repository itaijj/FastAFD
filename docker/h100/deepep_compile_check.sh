#!/usr/bin/env bash
# Compile-only check of the DeepEP V2 extension (nvcc 13, sm_90) without a GPU.
#   podman run --rm -v $PWD:/workspace/FastAFD:Z fastafd-h100 bash docker/h100/deepep_compile_check.sh
set -uo pipefail
cd /workspace/FastAFD
export MINISGL_DEEPEP_BUILD_DIR="${MINISGL_DEEPEP_BUILD_DIR:-/tmp/deepep_moe}"
export MAX_JOBS="${MAX_JOBS:-8}"
python - <<'PY'
import sys, torch
sys.path.insert(0, "python")
torch.cuda.get_device_capability = lambda *a, **k: (9, 0)  # no GPU in build env
from minisgl.kernel.deepep_moe import ensure_deepep_moe_built
print("DEEPEP_BUILT", ensure_deepep_moe_built(force_rebuild=True))
PY
echo "DEEPEP_EXIT=$?"
