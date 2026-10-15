#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_common_dir="$SCRIPT_DIR"
while [[ "$_common_dir" != "/" && ! -f "$_common_dir/scripts/lib/common.sh" ]]; do
  _common_dir="$(dirname "$_common_dir")"
done
# shellcheck source=/dev/null
source "$_common_dir/scripts/lib/common.sh"
fastafd_init_paths "$SCRIPT_DIR"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

# H100 (SM90) off-the-shelf AFD baseline -- no MegaMoE M2N megakernel.
#   Single 8xH100 node, Qwen3-30B-A3B-FP8.
#   AG: attention dp=1, tp=4 (4 GPUs); EG: MLP dp=4, tp=1, ep=4 (4 GPUs).
#   MoE: DeepEP V2 ElasticBuffer dispatch/combine + DeepGEMM grouped FP8 GEMM
#        (MINISGL_AFD_MOE_BACKEND=deepep).
#   Attention: FlashInfer (trtllm attention is Blackwell-only); page_size=1 so
#        decode CUDA graphs can be captured with the fi backend.
#   mb=1, no NSYS, modest workload for first bring-up, then vLLM TP=1 alignment.

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
export PROMPT_LEN="${PROMPT_LEN:-1024}"
export NUM_PROMPTS="${NUM_PROMPTS:-64}"
export MAX_TOKENS="${MAX_TOKENS:-16}"
export AFD_MAX_NEW_TOKENS="${AFD_MAX_NEW_TOKENS:-$MAX_TOKENS}"
export SAMPLE_CONCURRENCY="${SAMPLE_CONCURRENCY:-$NUM_PROMPTS}"
export BATCH_SUBMIT="${BATCH_SUBMIT:-1}"
export IGNORE_EOS="${IGNORE_EOS:-1}"

export ATTN_DP_SIZE="${ATTN_DP_SIZE:-1}"
export ATTN_TP_SIZE="${ATTN_TP_SIZE:-4}"
export MLP_DP_SIZE="${MLP_DP_SIZE:-4}"
export MLP_TP_SIZE="${MLP_TP_SIZE:-1}"
export AFD_MLP_EP_SIZE="${AFD_MLP_EP_SIZE:-4}"
export AFD_NUM_MB="${AFD_NUM_MB:-1}"
export AFD_BATCH_SIZE="${AFD_BATCH_SIZE:-$NUM_PROMPTS}"
export MINISGL_MAX_RUNNING_REQUESTS="${MINISGL_MAX_RUNNING_REQUESTS:-$NUM_PROMPTS}"
export MINISGL_MAX_SEQ_LEN="${MINISGL_MAX_SEQ_LEN:-$((PROMPT_LEN + MAX_TOKENS + 64))}"
export MINISGL_CUDA_GRAPH_MAX_BS="${MINISGL_CUDA_GRAPH_MAX_BS:-64}"
export AFD_DECODE_GRAPH_BS="${AFD_DECODE_GRAPH_BS:-8,16,32,64}"
export AFD_MAX_BATCHED_TOKENS="${AFD_MAX_BATCHED_TOKENS:-16384}"

export AFD_MOE_A2A_BACKEND="${AFD_MOE_A2A_BACKEND:-deepep}"
export AFD_MOE_RUNNER_BACKEND="${AFD_MOE_RUNNER_BACKEND:-deep_gemm}"
export MINISGL_AFD_MOE_BACKEND="${MINISGL_AFD_MOE_BACKEND:-deepep}"

export MINISGL_PAGE_SIZE="${MINISGL_PAGE_SIZE:-1}"
export AFD_CACHE_TYPE="${AFD_CACHE_TYPE:-naive}"
export AFD_SERVER_EXTRA_ARGS="${AFD_SERVER_EXTRA_ARGS:---attention-backend fi --memory-ratio 0.65}"

export NSYS="${NSYS:-0}"

export RUN_VLLM_ALIGNMENT="${RUN_VLLM_ALIGNMENT:-1}"
export VLLM_GPUS="${VLLM_GPUS:-0}"
export VLLM_TP_SIZE="${VLLM_TP_SIZE:-1}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MINISGL_MAX_SEQ_LEN}"
export VLLM_PROMPT_LOGPROBS="${VLLM_PROMPT_LOGPROBS:-0}"
export VLLM_SCORE_CONCURRENCY="${VLLM_SCORE_CONCURRENCY:-16}"
export VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager --moe-backend triton}"
export VLLM_USE_FLASHINFER_MOE_FP8="${VLLM_USE_FLASHINFER_MOE_FP8:-0}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export ALIGN_MIN_TOP10_RATE="${ALIGN_MIN_TOP10_RATE:-1.0}"

export CLEAN_FIRST="${CLEAN_FIRST:-1}"
export CLEAN_AFTER="${CLEAN_AFTER:-1}"
export POST_SHUTDOWN_SLEEP="${POST_SHUTDOWN_SLEEP:-20}"
export AFD_STOP_TIMEOUT_S="${AFD_STOP_TIMEOUT_S:-60}"

export RUN_DIR="${RUN_DIR:-$CALIB_DIR/reports/afd_qwen3_30b_a3b_fp8_h100_deepep_attn${ATTN_DP_SIZE}dp_tp${ATTN_TP_SIZE}_mlp${MLP_DP_SIZE}dp_ep${AFD_MLP_EP_SIZE}_mb${AFD_NUM_MB}_align_${TIMESTAMP}}"

exec "$SCRIPT_DIR/run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh"
