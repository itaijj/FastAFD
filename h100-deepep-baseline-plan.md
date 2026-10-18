# H100 (SM90) Off-the-Shelf AFD MoE Baseline Plan

## Top-Level Overview

**Goal:** Run AFD (attention ranks = AG, expert ranks = EG) on a single 8xH100 node with Qwen3-30B-A3B-FP8, without the SM100-only MegaMoE M2N megakernel.

**Approach:** Reuse the existing non-megakernel `deepep` AFD backend: DeepEP V2 ElasticBuffer dispatch/combine (`DeepEPM2NAdapter`) plus the DeepGEMM grouped FP8 GEMM runner (`DeepGEMMGroupedRunner`). It is already the default (`MINISGL_AFD_MOE_BACKEND` defaults to `deepep`) and most of it is arch-aware. The work is to close the SM90 gaps found during research, add an H100 launch configuration, make the megakernel backend fail clearly on non-SM100, and validate correctness.

**Topology for validation:** 4 AG ranks (attention dp=1, tp=4) and 4 EG ranks (mlp dp=4, tp=1, ep=4) on one 8xH100 node.

**In scope:**
- SM90 scale-recipe fix in the DeepGEMM expert path.
- H100 launch script (FlashInfer attention, page_size 1).
- Fail-fast guard for `megamoe_m2n` on non-SM100.
- One single-GPU SM90 expert-path unit test.
- An end-to-end vLLM alignment run.

**Non-goals:**
- Porting the MegaMoE megakernel to SM90.
- Performance tuning.
- Multi-node runs.
- Models other than Qwen3-30B-A3B-FP8.
- FP4 weights.

**Known facts from research:**
- AG dispatch payload: `per_token_cast_to_fp8(..., use_ue8m0=True, gran_k=128, use_packed_ue8m0=True)` produces FP8 activations plus packed int32 UE8M0 scales (`afd_attention_worker.py` `_fp8_dispatch_quant`).
- EG compute path:
  - `DeepGEMMGroupedRunner._apply_psum_layout` -> `deep_gemm.m_grouped_fp8_gemm_nt_contiguous(..., use_psum_layout=True)` with no explicit recipe.
  - Then `persistent_psum_silu_mul_quant`, which emits packed int32 scales, then a second grouped GEMM.
- The DeepGEMM C++ code has an SM90 dispatch for the contiguous/psum GEMM (`sm90_m_grouped_fp8_gemm_contiguous_1d2d`, float SFA required).
- `transform_sf_into_required_layout` converts packed int32 UE8M0 scales (gran 1x128) to float on SM90.
- However, `get_default_recipe` on SM90 asserts that SFA and SFB are both float. Packed int32 SFA therefore fails unless a recipe is passed explicitly.
- `get_theoretical_mk_alignment_for_contiguous_layout` returns the legacy 128 on non-SM100. This matches the 128-group quant kernels.
- DeepEP builds with `TORCH_CUDA_ARCH_LIST` set to the current device. Its SM100-only instructions (`fadd2`, `add.rn.f32.bf16`, `longlong4` loads) have `#else` fallbacks.
- `normalize_attention_backend_page_size` picks `fi` on non-SM100. Decode CUDA graph capture with `fi` requires `page_size == 1`. The existing Qwen3-30B scripts hardcode `--attention-backend trtllm` with page size 64.

## Sub-Tasks

### Sub-Task 1: SM90 scale-recipe fix in the DeepGEMM expert runner

- **Status:** [x] done (code; GPU validation pending via Sub-Task 3/5)
- **Implementation notes:**
  - The fix was made in C++ rather than in Python. With a recipe of `None`, the bindings call `get_default_recipe`, so this was the smallest change. No change to `deepgemm_grouped.py` was needed.
  - `csrc/utils/layout.hpp` `get_default_recipe`: on arch 9, SFA may now be `kFloat` or `kInt`. SFB must still be float. It returns `{1,128,128}`.
  - `csrc/apis/layout.hpp`, SM90 `(INT, 1, 128)` unpack branch: added `.contiguous()` before `.view(torch::kUInt8)`. The packed SFA arrives MN-major (strides `(1, align(m,4))`), and a dtype view needs a unit last-dim stride.
  - Weight scales: `_get_fp8_weights_for_deepgemm` produces float 128x128 UE8M0 block scales. On SM90 these take the existing float `(128,128)` SFB `check_sf_layout` path, so no change was needed.
  - Note: `apis/gemm.hpp` also has an uncommitted SM90 `k_grouped_fp8_gemm_tn_contiguous` dispatch from earlier work. This AFD path doesn't use it, and it was left untouched.
- **Intent:** On SM90, the grouped FP8 GEMM calls in the EG expert path must not hit the default-recipe assert when activation scales are packed int32 UE8M0. Without this fix, every EG expert call on H100 will crash.
- **Expected Outcomes:**
  - On SM90, `DeepGEMMGroupedRunner` grouped FP8 GEMM calls (contiguous/psum and masked) run with packed int32 activation scales and UE8M0-requantized weight scales.
  - SM100 behaviour is unchanged (no recipe passed, or an equivalent one).
  - Weight-scale preparation in `_get_fp8_weights_for_deepgemm` produces a layout the SM90 kernel accepts. Verify whether `transform_sf_into_required_layout` with recipe `(1,128,128)` and float block scales takes the SM90 128x128 path. If an SM90-incompatible layout is produced, adjust only for SM90.
- **Todo List:**
  1. Confirm the exact failure on SM90 by reading the call chain: `m_grouped_fp8_fp4_gemm_nt_contiguous` -> `transform_sf_pair_into_required_layout` -> `get_default_recipe`.
  2. In `DeepGEMMGroupedRunner._invoke_fp8_contiguous` and `_invoke_fp8_grouped`, pass an explicit recipe `(1, 128, 128)` when running on SM90. Alternatively pass `recipe_a=(1,128)` and `recipe_b=(128,128)`, whichever matches the C++ binding signature.
  3. Check `_get_fp8_weights_for_deepgemm` on SM90:
     - `_requant_weight_ue8m0_inplace` uses `deep_gemm.per_block_cast_to_fp8(use_ue8m0=True)`, which yields float scales rounded to powers of two.
     - Make sure the pre-transformed `w1_scale`/`w2_scale` are accepted by the SM90 GEMM. Pre-transforming with `(1,128,128)` on SM90 may yield an MN-major per-row tensor, while SM90 expects 128x128 block SFB. Pick the SM90-correct path.
  4. Detect the arch once with the existing helper `minisgl.utils.is_sm90_supported` / `is_sm100_supported`, or `torch.cuda.get_device_capability()`.
- **Relevant Context:**
  - `python/minisgl/layers/moe/moe_runner/deepgemm_grouped.py`: `_invoke_fp8_grouped`, `_invoke_fp8_contiguous`, `_get_fp8_weights_for_deepgemm`, `_apply_psum_layout`.
  - `python/minisgl/kernel/csrc/deepgemm/csrc/apis/gemm.hpp`: `m_grouped_fp8_fp4_gemm_nt_contiguous`, `m_grouped_fp8_fp4_gemm_nt_masked`, and their `m.def` signatures near the end of the file.
  - `python/minisgl/kernel/csrc/deepgemm/csrc/apis/layout.hpp`: `transform_sf_into_required_layout` (SM90 int32 -> float unpack branch).
  - `python/minisgl/kernel/csrc/deepgemm/csrc/utils/layout.hpp`: `get_default_recipe`.
  - `python/minisgl/kernel/deepgemm.py`: Python wrapper exports (`per_block_cast_to_fp8`, etc.).

### Sub-Task 2: Fail-fast guard for `megamoe_m2n` on non-SM100

- **Status:** [x] done
- **Implementation notes:** `afd_worker_base.py` `_init_afd_runtime` raises `RuntimeError` at the top of the `megamoe_m2n` branch when `torch.cuda.get_device_capability()[0] != 10`. This runs before any symmetric-memory or collective setup, so all ranks fail the same way.
- **Intent:** Selecting `MINISGL_AFD_MOE_BACKEND=megamoe_m2n` on H100 should raise a clear error at adapter-init time, not hit `DG_HOST_UNREACHABLE` deep inside a collective, where it can hang the other ranks.
- **Expected Outcomes:**
  - On a non-SM100 device, initialising AFD with `megamoe_m2n` raises a `RuntimeError` that names the arch requirement and suggests `MINISGL_AFD_MOE_BACKEND=deepep`.
  - SM100 behaviour is unchanged.
- **Todo List:**
  1. Add an arch check at the top of the `megamoe_m2n` branch in `_init_afd_runtime`, or in `MegaMoEM2NAfdAdapter.__init__`, using `is_sm100_supported()`.
  2. Make sure the check runs before any collective (symmetric-memory allocation) so that every rank fails consistently.
- **Relevant Context:**
  - `python/minisgl/afd_worker_base.py` `_init_afd_runtime` (backend selection around `afd_env("MOE_BACKEND", "deepep")`).
  - `python/minisgl/moe/megamoe_m2n_afd.py` `MegaMoEM2NAfdAdapter.__init__`.
  - `python/minisgl/utils/arch.py` `is_sm100_supported`.

### Sub-Task 3: Single-GPU SM90 expert-path unit test

- **Status:** [x] done (PASS on H100 80GB HBM3, k8s pod `h100-psum-moe-test`, commit `a195bc9`)
- **Result:** `PASS: calc_diff=3.520e-04 max_abs=6.399e-01 finite=True m=769 counts=[37, 0, 130, 5, 256, 1] alignment=128`
- **Fix found while running:** with nvcc 13, the tvm_ffi JIT kernels (`per_token_fp8_packed`, `psum_silu_mul_fp8_packed`, ...) failed in NVVM with `Call parameter type does not match function signature!` on `std::source_location::current()` and `parse Invalid record`. The cause is the nvcc 12.1 `consteval -> constexpr` workaround in `kernel/csrc/include/minisgl/utils.h`: together with `--expt-relaxed-constexpr` it makes `source_location::current()` device-visible. The workaround now applies only when `__CUDACC_VER_MAJOR__ < 13`. All 7 JIT kernels and the DeepGEMM extension compile with nvcc 13 (`docker/h100/jit_compile_check.sh`).
- **How to rerun:** `GPU_COUNT=1 bash docker/h100/k8s/deploy_test.sh` (one-shot pod; base image `nvidia/cuda:13.0.2-devel-ubuntu24.04`, deps installed with uv at startup).
- **Implementation notes:**
  - Location: `scripts/validate/test_h100_deepgemm_psum_moe.py`. The repo has no `tests/`; move it if preferred. `py_compile` passes.
  - Run: `python scripts/validate/test_h100_deepgemm_psum_moe.py`. It skips on non-SM90.
  - Coverage: 6 experts with counts `[37,0,130,5,256,1]`, including an empty expert and uneven counts. Runs `DeepGEMMGroupedRunner.apply` on the psum layout and compares against an FP32 reference with `calc_diff < 1e-3`.
- **Intent:** Check the H100 expert compute path in isolation, without DeepEP or multi-rank setup, so recipe and scale-layout bugs show up quickly.
- **Expected Outcomes:**
  - A standalone test script that, on one H100, builds synthetic Qwen-style FP8 expert weights with 128x128 block scales, plus BF16 tokens grouped per expert in the psum layout with 128 alignment.
  - The script quantizes activations exactly as the AG side does and runs the `DeepGEMMGroupedRunner` psum path.
  - It compares the output against a torch FP32 reference (dequantize, then SiLU-and-mul, then matmul) and reports a pass/fail similarity metric.
  - The test passes on SM90. It is skipped, not failed, on non-SM90 devices.
- **Todo List:**
  1. Pick a location. The repo has no `tests/` directory; consider `scripts/validate/` or a new `tests/` folder. Confirm with the user during implementation.
  2. Build inputs:
     - Small shapes compatible with Qwen3-30B-A3B (hidden 2048, moe_intermediate 768; few experts such as 4, top_k 2).
     - Include empty experts and uneven token counts.
  3. Drive the runner through a minimal dispatch-output-like object that provides `hidden_states`, `hidden_states_scale`, `recv_count` (psum) and `topk_weights`, plus a minimal layer object that provides `gate_up_proj`, `down_proj`, `*_scale`, `quant` and `activation`.
  4. Write the reference: dequantize FP8 weights and activations, run SwiGLU, apply the top-k weights the same way the runner does, then compare.
  5. Report a DeepGEMM-style `calc_diff` (or cosine similarity) with a threshold of about 1e-3, plus max abs error.
- **Relevant Context:**
  - `python/minisgl/layers/moe/moe_runner/deepgemm_grouped.py` `_apply_psum_layout`.
  - `python/minisgl/kernel/fp8_quant.py`: `per_token_cast_to_fp8_cuda`, `psum_silu_mul_fp8_quant_cuda`.
  - `python/minisgl/kernel/deepgemm_fused_quant.py` `persistent_psum_silu_mul_quant`.
  - `python/minisgl/afd_attention_worker.py` `_fp8_dispatch_quant`.

### Sub-Task 4: H100 launch script for 4 AG + 4 EG

- **Status:** [x] done (script; run pending in Sub-Task 5)
- **Implementation notes:**
  - New script: `scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_h100_1node_deepep_alignment.sh`. `bash -n` passes and the script is executable.
  - Defaults:
    - Topology: attention dp1/tp4; MLP dp4/tp1/ep4.
    - MoE: `deepep` + `deep_gemm`, `MINISGL_AFD_MOE_BACKEND=deepep`.
    - Attention: `--attention-backend fi --memory-ratio 0.65`, page_size 1, naive cache.
    - Run: mb=1, NSYS=0, 64 prompts x 1024 tokens, 16 output tokens.
    - Graph batch sizes `8,16,32,64`.
    - vLLM alignment on GPU 0 with TP=1 (Triton MoE, enforce-eager).
  - `MINISGL_AFD_MOE_BACKEND` reaches the workers through `afd_support.py` and `supervisor_ray.py`.
- **Intent:** Provide a repeatable single-node 8xH100 launch for Qwen3-30B-A3B-FP8 that avoids Blackwell-only options.
- **Expected Outcomes:**
  - A new script under `scripts/experiments/afd/qwen3_30b/`. It reuses the existing `run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh` driver and overrides these environment variables:
    - attention dp=1, tp=4; mlp dp=4, tp=1, ep=4;
    - `AFD_MOE_A2A_BACKEND=deepep`, `AFD_MOE_RUNNER_BACKEND=deep_gemm`, `MINISGL_AFD_MOE_BACKEND=deepep`;
    - `--attention-backend fi`, `MINISGL_PAGE_SIZE=1`, `AFD_CACHE_TYPE=naive`;
    - a modest prompt count and length for a first bring-up;
    - `RUN_VLLM_ALIGNMENT=1` with vLLM on a GPU set that is free after AFD shuts down, using `VLLM_EXTRA_ARGS` with the Triton MoE backend (already the default).
  - The script does not change the existing Blackwell scripts.
- **Todo List:**
  1. Create the H100 wrapper script, following the style of `run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh` (env-var defaults, then `exec` the driver).
  2. Set `AFD_SERVER_EXTRA_ARGS` to `--attention-backend fi` with an appropriate `--memory-ratio`.
  3. Default `NSYS=0` and `AFD_NUM_MB=1` for the first bring-up. Leave `mb=2` as an override.
  4. Document in a script comment that this is the H100 off-the-shelf baseline (no megakernel).
- **Relevant Context:**
  - `scripts/experiments/afd/qwen3_30b/run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh` (driver; supports all needed env overrides).
  - `scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh` (wrapper style).
  - `python/minisgl/afd_support.py`: `normalize_attention_backend_page_size`, `_supports_decode_graph_capture`.

### Sub-Task 5: End-to-end H100 bring-up and alignment validation

- **Status:** [-] in progress. The AFD run and the vLLM alignment both PASS on 2 AG + 2 EG (H100, k8s pod `h100-afd-run`, commit `3e138ee`). The 4 AG + 4 EG topology and a realistic-prompt alignment run are still to be done.
- **Findings so far (k8s run on an 8×H100 node `<h100-node>`, namespace `<namespace>`):**
  - Pod entry: `docker/h100/k8s/afd_entry.sh`; deploy with `docker/h100/k8s/deploy_afd.sh`; status helper `docker/h100/k8s/status.sh`.
  - **Ray raylet crash (fixed):** `ray start --num-cpus=$(nproc)` advertised the node's 224 CPUs. Ray prestarted about one worker per CPU and `pthread_create` failed with `EAGAIN`, which killed the raylet. Now capped via `RAY_CPUS` (default 32).
  - **DeepEP `ElasticBuffer` on H100 (fixed):** the constructor asserts `ncclCommQueryProperties().railedGinType != NONE` (`kernels/backend/nccl.cu`). The pod has the RDMA devices (after requesting `rdma/roce_gdr`), but NCCL GIN is still unavailable, so the assert fails.
    - Fix: the assert is skipped when `EP_DISABLE_GIN=1`. The existing code already skips `ginContextCount` setup for that variable, so single-node NVLink use needs no GIN.
    - `EP_DISABLE_GIN` is added to the Ray env passthrough lists (`supervisor_ray.py`, `afd_support.py`). The pod entry exports `EP_DISABLE_GIN=1`.
    - The pod requests `rdma/roce_gdr: 1` (`RDMA` knob in `deploy_afd.sh`). This is harmless but was not enough alone.
  - **Result with the fixes:** all 4 ranks initialise, the DeepEP and DeepGEMM extensions JIT-build, AG and EG decode CUDA graphs are captured for bs 8/16/32/64 and replayed, all 32 prompts generate 8 tokens each, and the script prints `=== AFD RUN PASS ===`.
  - **vLLM alignment (2 AG + 2 EG, `RUN_VLLM_ALIGNMENT=1`, vLLM 0.19.0+cu130 x86_64 wheel, Triton MoE, enforce-eager, 1 GPU):** 32 prompts, 256 generated tokens; `overall_top1_agreement_rate=1.0`, `overall_top10_hit_rate=1.0`, `overall_top100_hit_rate=1.0`, `overall_avg_rank_in_vllm=1.0`, `overall_max_rank_in_vllm=1.0`. This meets the default `ALIGN_MIN_TOP10_RATE=1.0`.
  - **Caveat:** this run used `PROMPT_KIND=hello` (repeated "hello"), 512-token prompts and 8 output tokens, so it is an easy case. A `realistic` prompt run (downloads text, needs internet) is still needed before calling alignment fully validated.
  - **4 AG + 4 EG not run yet:** no single H100 node had 8 free GPUs (GPUs are shared; the most free on one node was 7). Run it with `ATTN_DP=1 ATTN_TP=4 MLP_DP=4 EP=4 GPU_COUNT=8 RUN_VLLM_ALIGNMENT=1 bash docker/h100/k8s/deploy_afd.sh` when a node frees up (`docker/h100/k8s/gpu_free.sh`).
  - Not measured: decode step time (the run had `NSYS=0` and no timing capture).
- **Intent:** Prove the full AFD pipeline runs on H100 and gives output aligned with vLLM. Fix any remaining SM90 issues that can only be seen at runtime.
- **Expected Outcomes:**
  - The Sub-Task 4 script completes:
    - All 4 AG and 4 EG ranks initialise. The DeepEP ElasticBuffer reports its NVLink domain in `deepep_m2n:new_afd_deepep` logs.
    - Decode CUDA graphs are captured and replayed.
    - All prompts generate the expected number of tokens.
  - The vLLM alignment summary meets the existing thresholds (`ALIGN_MIN_TOP10_RATE`, default 1.0 in the alignment wrappers). Agree an acceptable top-1 rate with the user.
  - Any further SM90 fixes are recorded in this plan file under this sub-task.
- **Todo List:**
  1. Run the Sub-Task 3 unit test on H100 first; it must pass.
  2. Run the H100 launch script. Check `afd.log` and the `ray_logs` per-rank logs for: DeepEP/DeepGEMM JIT build success for `sm_90a`, `afd_deepep_adapters`, graph capture, and replay counts.
  3. If a JIT kernel (`topk_softmax_group_local`, the FP8 quant kernels, DeepEP) fails to compile or run on SM90, record it here and fix it with the smallest arch guard possible.
  4. If DeepEP ElasticBuffer fails on SM90 (for example hybrid or QP settings), record it. Try the smallest config change in `DeepEPMoeElasticBuffer` first (such as `allow_hybrid_mode` on a single node), and only then consider anything larger.
  5. Collect the alignment summary and a basic decode step time for reference.
- **Relevant Context:**
  - `python/minisgl/kernel/deepep_moe.py` `DeepEPMoeElasticBuffer` (ElasticBuffer config: `allow_hybrid_mode`, `num_allocated_qps`).
  - `python/minisgl/moe/deepep_m2n_adapter.py` `DeepEPM2NAdapter`.
  - `python/minisgl/afd_expert_worker.py` (EG per-layer dispatch -> `run_experts` -> combine pipeline).
  - `python/minisgl/afd_attention_worker.py` (AG graph capture path: `afd_dispatch_payload`, `adapter.dispatch`, `adapter.combine`).
  - `scripts/validate/compare_minisgl_vllm.py` (sample/score alignment).
