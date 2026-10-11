"""Single-GPU H100 (SM90) check of the AFD EG expert path (no DeepEP).

Runs ``DeepGEMMGroupedRunner`` psum-layout FP8 MoE exactly as the EG rank does
(AG-style packed-UE8M0 activation quant -> grouped FP8 GEMM -> SiLU*mul+quant
-> grouped FP8 GEMM) and compares against an FP32 torch reference.

Usage: python scripts/validate/test_h100_deepgemm_psum_moe.py
Exit code 0 on pass (or skip on non-SM90), 1 on failure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

GRAN = 128


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return float(1 - 2 * (x * y).sum() / denom) if denom > 0 else 0.0


def dequant_block_weight(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # w: [E, N, K] fp8, s: [E, N/128, K/128] fp32
    s_full = s.repeat_interleave(GRAN, dim=1).repeat_interleave(GRAN, dim=2)
    return w.float() * s_full[:, : w.shape[1], : w.shape[2]]


def dequant_packed_act(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    # x: [M, K] fp8, s: [M, ceil(K/512)] int32 packed UE8M0 (MN-major strides)
    m, k = x.shape
    groups = k // GRAN
    exp = s.contiguous().view(torch.uint8).reshape(m, -1)[:, :groups].to(torch.int32)
    scale = (exp << 23).view(torch.float32).repeat_interleave(GRAN, dim=1)
    return x.float() * scale


def make_qwen_fp8_weight(e: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    from minisgl.kernel import deepgemm as deep_gemm

    ws, ss = [], []
    for _ in range(e):
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.05
        w_q, s = deep_gemm.per_block_cast_to_fp8(w.float(), use_ue8m0=False, gran_k=GRAN)
        ws.append(w_q)
        ss.append(s)
    return torch.stack(ws), torch.stack(ss).float()


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        print("SKIP: requires an SM90 (H100) GPU")
        return 0

    from minisgl.kernel import deepgemm as deep_gemm
    from minisgl.kernel.fp8_quant import per_token_cast_to_fp8
    from minisgl.layers.moe.moe_runner.base import MoeRunnerConfig
    from minisgl.layers.moe.moe_runner.deepgemm_grouped import DeepGEMMGroupedRunner

    torch.manual_seed(0)
    hidden, inter = 2048, 768  # Qwen3-30B-A3B
    counts = [37, 0, 130, 5, 256, 1]  # uneven + empty experts
    num_experts = len(counts)

    alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

    # psum layout: expert e occupies [align(psum[e-1]), psum[e])
    psum, starts, end = [], [], 0
    for c in counts:
        start = (end + alignment - 1) // alignment * alignment
        starts.append(start)
        end = start + c
        psum.append(end)
    m = psum[-1]
    valid = torch.zeros(m, dtype=torch.bool, device="cuda")
    for s, c in zip(starts, counts):
        valid[s : s + c] = True

    x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16)
    x[~valid] = 0
    topk_w = torch.rand(m, device="cuda", dtype=torch.float32)
    x_q, x_s = per_token_cast_to_fp8(
        x, use_ue8m0=True, gran_k=GRAN, use_packed_ue8m0=True, backend="cuda"
    )

    w1, w1_s = make_qwen_fp8_weight(num_experts, 2 * inter, hidden)
    w2, w2_s = make_qwen_fp8_weight(num_experts, hidden, inter)
    layer = SimpleNamespace(
        gate_up_proj=w1,
        gate_up_proj_scale=w1_s,
        down_proj=w2,
        down_proj_scale=w2_s,
        quant=SimpleNamespace(method="fp8", weight_block_size=(GRAN, GRAN)),
        activation="silu",
    )
    disp = SimpleNamespace(
        hidden_states=x_q,
        hidden_states_scale=x_s,
        recv_count=torch.tensor(psum, dtype=torch.int32, device="cuda"),
        topk_weights=topk_w,
    )

    out = DeepGEMMGroupedRunner(MoeRunnerConfig()).apply(disp, layer)
    torch.cuda.synchronize()

    # Reference uses the runner's UE8M0-requantized weights (updated in place).
    w1_dq = dequant_block_weight(layer.gate_up_proj, layer.gate_up_proj_scale)
    w2_dq = dequant_block_weight(layer.down_proj, layer.down_proj_scale)
    x_dq = dequant_packed_act(x_q, x_s)
    ref = torch.zeros((m, hidden), device="cuda", dtype=torch.float32)
    for e, (s, c) in enumerate(zip(starts, counts)):
        if c == 0:
            continue
        h = x_dq[s : s + c] @ w1_dq[e].t()
        g, u = h[:, :inter], h[:, inter:]
        a = torch.nn.functional.silu(g) * u * topk_w[s : s + c, None]
        ref[s : s + c] = a @ w2_dq[e].t()

    got = out.float()[valid]
    want = ref[valid]
    diff = calc_diff(got, want)
    max_abs = float((got - want).abs().max())
    finite = bool(torch.isfinite(got).all())
    ok = finite and diff < 1e-3
    print(
        f"{'PASS' if ok else 'FAIL'}: calc_diff={diff:.3e} max_abs={max_abs:.3e} "
        f"finite={finite} m={m} counts={counts} alignment={alignment}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
