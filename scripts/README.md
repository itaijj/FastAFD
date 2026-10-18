# FastAFD Scripts

This directory is organized by workflow.

## Public Entry Points

Quickstart:

```bash
./scripts/quickstart/qwen3_30b_a3b_sample.sh
```

Correctness alignment:

```bash
./scripts/validate/qwen3_30b_a3b_alignment.sh
./scripts/validate/qwen3_30b_a3b_fastafd_alignment.sh
```

Large-scale AFD preset:

```bash
./scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh
```

Published Qwen3-235B and MiniMax M2.5 result presets live under
`scripts/experiments/afd/qwen3_235b/` and `scripts/experiments/afd/minimax_m25/`;
they dispatch through `scripts/run_afd_qwen3_30b_a3b_fp8_3node_mb2_nsys_alignment.sh`.

Serving benchmark (4 GPUs vs 8 GPUs vs 2 nodes x 4 GPUs; 256 in / 1024 out; concurrency 64/128/256):

```bash
IMAGE=<cuda image> PVC_NAME=<model pvc> PYTHON=<python with transformers> \
  bash scripts/bench/run_sweep.sh
# preview without touching the cluster:
DRY_RUN=1 bash scripts/bench/run_sweep.sh
# single config / reuse a running server:
CONFIGS=B CONCURRENCIES="64" bash scripts/bench/run_sweep.sh
SKIP_DEPLOY=1 CONFIGS=B BENCH_HOST=<ip> BENCH_PORT=<port> bash scripts/bench/run_sweep.sh
```

Needs a kubectl context on the H100 cluster with free GPUs, a PVC holding the model, and (config C)
two RoCE-capable nodes. Results land in `reports/bench_<timestamp>/` (`summary.md`, `summary.csv`,
one JSON per run). `IMAGE` is only needed for the single-node configs. See the header of
`scripts/bench/run_sweep.sh` for all knobs.

## Layout

- `bench/`: serving benchmark client (`serve_bench.py`), sweep driver (`run_sweep.sh`), summary (`summarize.py`).
- `quickstart/`: mini-sgl sampling workflow.
- `serve/`: mini-sgl, FastAFD, and vLLM server launchers.
- `validate/`: mini-sgl/vLLM and FastAFD/vLLM alignment pipelines.
- `experiments/afd/qwen3_30b/`: the FastAFD launcher the presets dispatch to.
- `experiments/afd/qwen3_235b/`: Qwen3-235B published-result presets.
- `experiments/afd/minimax_m25/`: MiniMax M2.5 published-result presets.
- `experiments/vllm/`: the sharded vLLM alignment scorer.
- `data_gen/`: prompt generation helper.
- `lib/`: shared shell helpers.
