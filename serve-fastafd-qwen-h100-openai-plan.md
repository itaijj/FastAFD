# Serve FastAFD Qwen3-30B on H100 via OpenAI API (K8s Persistent Service)

## Top-Level Overview

**Goal:** Deploy the FastAFD miniSGL runtime serving `Qwen/Qwen3-30B-A3B-FP8` on a single 8×H100 node as a **persistent Kubernetes Pod+Service** (not a one-shot benchmark pod), exposing an OpenAI-compatible HTTP endpoint at `http://fastafd-qwen30b-service:<port>/v1/` accessible in-cluster and from outside the cluster via a NodePort.

**What already exists:**
- The H100 AFD backend (DeepEP + DeepGEMM) is validated and working (see `h100-deepep-baseline-plan.md` sub-tasks 1–5 done/in-progress).
- The miniSGL OpenAI-compatible API server is live at `python/minisgl/server/api_server.py` with `/v1/chat/completions`, `/v1/completions`, `/v1/models`.
- The one-shot pod entry script at `docker/h100/k8s/afd_entry.sh` installs deps, downloads the model, starts Ray, and runs the AFD launch script — but it runs a benchmark, then shuts down.
- The `deploy-afd-k8s/` skill exists for the vLLM-plugin AFD path; the FastAFD miniSGL path does not yet have an equivalent persistent-serve pod/service.

**Approach:** Create a minimal persistent K8s serve layer directly in `docker/h100/k8s/`:
1. A new entrypoint script that starts the miniSGL AFD server and keeps it alive (no benchmark loop, no one-shot exit).
2. A K8s Service (`fastafd-qwen30b-service`) of type **`NodePort`** — reachable both in-cluster via the ClusterIP and externally via `<nodeIP>:<nodePort>`.
3. A K8s Pod manifest (`fastafd-qwen30b`, persistent, `restartPolicy: Always`) wired to that Service.
4. A deploy script that creates the Service first, then the Pod.

**Topology (from `qwen_run.txt` / validated H100 run):**
- 4 AG ranks: attention `dp=1, tp=4`
- 4 EG ranks: MLP `dp=4, tp=1, ep=4`
- Total: 8 H100 GPUs on a single node
- MoE: `MINISGL_AFD_MOE_BACKEND=deepep`, runner `deep_gemm`, attention `fi`, `page_size=1`

**Scope:**
- Single-pod (all 8 ranks on one node); no multi-node disaggregation.
- Model: `Qwen/Qwen3-30B-A3B-FP8` only.
- The existing dep-install and Ray setup from `afd_entry.sh` is reused as-is.
- No changes to the Python source; only new shell and YAML artifacts.

**Non-goals:**
- Performance tuning or multi-node serving.
- K8s Deployment rollout strategy (Pod is fine for GPU serving).
- Ingress / external-cluster routing.
- Qwen3-235B or any other model.

---

## Sub-Tasks

### Sub-Task 1: Persistent serve entrypoint script

- **Status:** [x] done — `docker/h100/k8s/afd_serve_entry.sh`
- **Intent:** The existing `afd_entry.sh` runs a benchmark then exits. We need a variant that starts the miniSGL AFD API server and **blocks forever** — ready to receive OpenAI API requests in-cluster, never shutting down on its own.
- **Expected Outcomes:**
  - New file `docker/h100/k8s/afd_serve_entry.sh`.
  - Reuses the dep-install, `EP_DISABLE_GIN`, model-download, and Ray startup blocks from `afd_entry.sh` verbatim (or by `source`/copy).
  - After Ray is up, launches `fastafd_server.sh` (the same script that `run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh` uses at line 309) in **serve mode** — i.e. without a benchmark run that terminates the server afterward.
  - Server binds on `0.0.0.0` (not `127.0.0.1`) so the K8s Service can forward to it.
  - Prints `=== AFD SERVE READY ===` once `/v1/models` returns 200, then enters `sleep infinity` (keeping the container alive while the server runs as a background process).
  - If the server exits unexpectedly, the script exits non-zero so the pod enters `Failed` state and an operator can inspect logs.
  - Topology and port are parameterised via environment variables with the same names used in `afd_entry.sh` (`ATTN_DP_SIZE`, `ATTN_TP_SIZE`, `MLP_DP_SIZE`, `AFD_MLP_EP_SIZE`, `GPU_COUNT`) plus a new `SERVE_PORT` (default `8000`).
- **Todo List:**
  1. Copy the install + Ray startup block from `docker/h100/k8s/afd_entry.sh` into the new `afd_serve_entry.sh`.
  2. Replace the benchmark run block with a `fastafd_server.sh` call using `--host 0.0.0.0 --port "$SERVE_PORT" --launch-backend ray` and all H100 DeepEP flags (same as the H100 alignment script, minus the benchmark-specific flags like `--afd-max-new-tokens` and `--afd-prompt-file`).
  3. Wait for `http://127.0.0.1:$SERVE_PORT/v1/models` to return 200 (re-use the existing curl-loop pattern from `run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh` lines 351–364).
  4. Print `=== AFD SERVE READY ===`, then `sleep infinity` in the foreground while monitoring the server PID; exit non-zero if it dies.
- **Relevant Context:**
  - `docker/h100/k8s/afd_entry.sh` — source for the install/download/Ray blocks.
  - `scripts/experiments/afd/qwen3_30b/run_qwen3_30b_a3b_fp8_afd_long8k_dp4_ep4_attn1_tp4.sh` lines 308–364 — `fastafd_server.sh` launch pattern and readiness curl loop.
  - `scripts/experiments/afd/qwen3_30b/run_afd_qwen3_30b_a3b_fp8_h100_1node_deepep_alignment.sh` — all H100-specific env-var overrides (`fi` attention, `page_size=1`, `deepep` backend).
  - `python/minisgl/server/args.py` `ServerArgs.server_host` — confirms `--host 0.0.0.0` is the right flag name.

---

### Sub-Task 2: K8s Service and Pod manifests

- **Status:** [x] done — `docker/h100/k8s/fastafd-serve-pod.yaml`, `docker/h100/k8s/fastafd-serve-service.yaml`
- **Intent:** Provide the K8s YAML for the NodePort Service that exposes the miniSGL API port in-cluster and externally, and a Pod manifest with `restartPolicy: Always` that runs `afd_serve_entry.sh`.
- **Expected Outcomes:**
  - New file `docker/h100/k8s/fastafd-serve-pod.yaml` — a Pod (not Deployment; GPU scheduler affinity works better with explicit Pods on most clusters) with:
    - `restartPolicy: Always` so crash-restart is handled.
    - GPU resource request for `nvidia.com/gpu: "$TEMPLATE_GPU"` (8).
    - Environment variables forwarding `ATTN_DP_SIZE`, `ATTN_TP_SIZE`, `MLP_DP_SIZE`, `AFD_MLP_EP_SIZE`, `GPU_COUNT`, `SERVE_PORT`, `EP_DISABLE_GIN`, `MODEL_PATH`, `HF_HOME`, `RAY_CPUS`.
    - The `dshm`, `work`, and model PVC volume mounts from the existing `afd-pod.yaml`.
    - A ConfigMap volume for `afd_serve_entry.sh`.
    - Label `app: fastafd-serve` so the Service selector matches.
    - `readinessProbe` HTTP GET on `/v1/models` at `$SERVE_PORT` so the Service only routes after the server is warm.
  - New file `docker/h100/k8s/fastafd-serve-service.yaml` — a `NodePort` Service selecting `app: fastafd-serve`:
    - `port: ${TEMPLATE_SERVE_PORT}` → `targetPort: ${TEMPLATE_SERVE_PORT}` (in-cluster ClusterIP access)
    - `nodePort: ${TEMPLATE_NODE_PORT}` (external access via `<nodeIP>:<nodePort>`; blank means K8s auto-assigns from the 30000–32767 range)
- **Todo List:**
  1. Write `fastafd-serve-pod.yaml` using `envsubst`-compatible `${TEMPLATE_*}` placeholders, following the conventions in `docker/h100/k8s/afd-pod.yaml`. Key differences from the existing afd-pod: `restartPolicy: Always`, command points to `afd_serve_entry.sh` (mounted as a ConfigMap), and adds a `readinessProbe`.
  2. Write `fastafd-serve-service.yaml` as a ClusterIP Service with selector `app: fastafd-serve` and port `${TEMPLATE_SERVE_PORT}`.
  3. Verify `envsubst` variable names are consistent between the deploy script (Sub-Task 3) and both YAML templates.
- **Relevant Context:**
  - `docker/h100/k8s/afd-pod.yaml` — volume/env/resource conventions to follow.
  - `deploy-afd-k8s/templates/service-client.yaml` — minimal ClusterIP service pattern.
  - `python/minisgl/server/api_server.py` `app.get("/v1/models")` — endpoint the readiness probe should hit.

---

### Sub-Task 3: Persistent serve deploy script

- **Status:** [x] done — `docker/h100/k8s/deploy_serve.sh`
- **Intent:** One-command script to apply the Service and Pod into the cluster, wait for the pod to reach `Running`, then wait for the readiness probe (or `=== AFD SERVE READY ===` in logs), and print the in-cluster endpoint.
- **Expected Outcomes:**
  - New file `docker/h100/k8s/deploy_serve.sh`.
  - Accepts the same env-var knobs as `deploy_afd.sh` (pod name, PVC, branch, GPU count, topology, RDMA flag) plus `SERVE_PORT` (default `8000`) and `NODE_PORT` (default empty → auto-assigned by K8s).
  - Pod name is `fastafd-qwen30b` (not parameterised; matches the user-specified name).
  - Creates the `afd_serve_entry.sh` ConfigMap, applies the Service first, then the Pod.
  - Waits for the Pod to reach `Running` (bounded 15 min) and then polls logs for `=== AFD SERVE READY ===` (bounded 60 min to allow dep install + model download + JIT build + CUDA graph capture).
  - Prints the final endpoint: `http://fastafd-qwen30b-service:<SERVE_PORT>/v1/` and the `kubectl` command to verify: `kubectl run ...`.
  - Does **not** delete the Service on completion — leaves both Service and Pod running.
- **Todo List:**
  1. Write `deploy_serve.sh` following the style of `deploy_afd.sh` (envsubst template, ConfigMap for the entry script, namespace fsGroup detection).
  2. Apply Service before Pod (matches `deploy-afd-k8s` convention: services must exist before pods start).
  3. Add the bounded wait loop for `=== AFD SERVE READY ===` (not `=== AFD RUN PASS ===` which is benchmark-specific).
  4. Print teardown instructions: `kubectl delete pod $POD_NAME` and `kubectl delete service fastafd-qwen30b-service`.
- **Relevant Context:**
  - `docker/h100/k8s/deploy_afd.sh` — conventions for envsubst, ConfigMap, namespace fsGroup.
  - `deploy-afd-k8s/references/deploy.md` step 4f — wait-for-ready pattern to follow.
  - `docker/h100/k8s/status.sh` — existing helper for checking pod status.

---

### Sub-Task 4: Smoke-test the live endpoint

- **Status:** [x] done — `docker/h100/k8s/smoke_test_serve.sh`
- **Intent:** Verify the in-cluster OpenAI endpoint accepts requests correctly after the serve pod is ready.
- **Expected Outcomes:**
  - A short helper script `docker/h100/k8s/smoke_test_serve.sh` that, given `SERVICE_HOST` and `SERVE_PORT`, sends a single `/v1/chat/completions` request via `curl` and checks the response contains a non-empty `choices[0].message.content`.
  - Prints `=== SMOKE TEST PASS ===` or `=== SMOKE TEST FAIL ===`.
- **Todo List:**
  1. Write `smoke_test_serve.sh` using the existing `curl` shim pattern (pure Python fallback where `curl` is absent) already in `afd_entry.sh`.
  2. The test prompt should be short (e.g. `"Say hello in one word."`) with `max_tokens: 10`.
  3. Parse the JSON response with Python inline (matching the pattern already used in the benchmark scripts) and assert `choices[0].message.content` is non-empty.
- **Relevant Context:**
  - `python/minisgl/server/api_server.py` `v1_completions` — the endpoint being tested.
  - `docker/h100/k8s/afd_entry.sh` lines 62–113 — the `_procshim.py` curl fallback already in use.

---

## Architecture Summary

```
In-cluster consumer
        |
        | HTTP OpenAI API (:8000/v1/...)
        v
 fastafd-qwen30b-service (NodePort)
 also reachable: <nodeIP>:<nodePort>
        |
        v
 fastafd-qwen30b (restartPolicy: Always, nvidia.com/gpu: 8)
   container: afd_serve_entry.sh
        |
        +-- uv venv install (once per pod lifecycle)
        +-- HF model download (cached on PVC)
        +-- Ray head (local, 8 GPUs, capped CPUs)
        +-- python -m minisgl.server.launch --mode afd-serve
              |
              +-- AfdCoordinator (Ray actor)
              +-- AfdAttentionWorker x2 (AG, tp=2, GPUs 0-3)
              +-- AfdExpertWorker x2   (EG, dp=2, ep=2, GPUs 2-7)
              +-- FastAPI uvicorn (0.0.0.0:8000)
```

## New Files

| File | Purpose |
|------|---------|
| `docker/h100/k8s/afd_serve_entry.sh` | Pod entrypoint: install, download, Ray, serve, block |
| `docker/h100/k8s/fastafd-serve-pod.yaml` | K8s Pod manifest (`restartPolicy: Always`, readinessProbe) |
| `docker/h100/k8s/fastafd-serve-service.yaml` | K8s NodePort Service |
| `docker/h100/k8s/deploy_serve.sh` | One-command deploy + wait |
| `docker/h100/k8s/smoke_test_serve.sh` | Post-deploy smoke test |
