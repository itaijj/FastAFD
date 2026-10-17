#!/usr/bin/env bash
# Compile-only repro of the minisgl tvm_ffi JIT kernels used by the H100 test.
#   podman run --rm -v "$(pwd)":/workspace/FastAFD:Z fastafd-h100 bash docker/h100/jit_compile_check.sh
set -uo pipefail
cd /workspace/FastAFD
export TVM_FFI_CACHE_DIR=/tmp/tvm_ffi
export TVM_FFI_CUDA_ARCH_LIST="${TVM_FFI_CUDA_ARCH_LIST:-9.0}"
python - <<'PY'
import sys, traceback
sys.path.insert(0, "python")
import importlib
ok = True
for mod, name in (
    ("fp8_quant", "_jit_per_token_cast_fp8_packed"),
    ("fp8_quant", "_jit_psum_silu_mul_fp8_packed"),
    ("fp8_quant", "_jit_fp8_to_uint8"),
    ("fp8_quant", "_jit_add_rmsnorm_fp8_quant"),
    ("moe_topk", "_jit_moe_topk_softmax_module"),
    ("moe_align", "_jit_moe_align_module"),
    ("minimax_qk_norm", "_jit_module"),
):
    try:
        getattr(importlib.import_module(f"minisgl.kernel.{mod}"), name)()
        print("JIT_OK", mod, name)
    except Exception as e:
        ok = False
        msg = str(e)
        lines = [l for l in msg.splitlines() if "Call parameter" not in l and "addrspacecast" not in l]
        print("JIT_FAIL", mod, name, "\n".join(lines[-15:]))
try:
    from minisgl.kernel.deepgemm import _build_extension_path
    print("AOT_OK deepgemm", _build_extension_path())
except Exception:
    ok = False
    traceback.print_exc()
sys.exit(0 if ok else 1)
PY
