#!/usr/bin/env bash
# End-to-end FastAFD serving benchmark: for each config, deploy -> smoke test ->
# run serve_bench.py at each concurrency -> tear down; then write summary.csv/.md.
#
# Configs (all: Qwen3-30B-A3B-FP8, server cap 256 running requests):
#   A  1 node x 4 GPUs   attn dp1/tp2 + mlp dp2/tp1/ep2   (docker/h100/k8s/deploy_serve.sh)
#   B  1 node x 8 GPUs   attn dp1/tp4 + mlp dp4/tp1/ep4   (docker/h100/k8s/deploy_serve.sh)
#   C  2 nodes x 4 GPUs  head: attn dp1/tp4, worker: mlp dp1/tp4/ep4
#                                                          (docker/h100/k8s/deploy_serve_multinode.sh)
# Workload: 256 input tokens, 1024 output tokens, concurrency 64/128/256 (closed loop).
#
# Prerequisites:
#   * kubectl context/namespace pointing at the H100 cluster; free GPUs (see docker/h100/k8s/gpu_free.sh).
#     The configs run one after another and each is torn down before the next starts.
#   * PVC_NAME: PVC holding the HF model cache.
#   * IMAGE: container image for configs A and B (required by deploy_serve.sh). Not used by C.
#   * Config C needs two H100 nodes with RoCE (rdma/roce_gdr, multi-nic-compute).
#   * PYTHON: an interpreter with `transformers` installed (the client builds a 256-token prompt)
#     that can reach the NodePort (BENCH_HOST/BENCH_PORT override the discovered endpoint).
#
# Usage:
#   IMAGE=... PVC_NAME=... bash scripts/bench/run_sweep.sh
#   CONFIGS="B" CONCURRENCIES="64" DRY_RUN=1 bash scripts/bench/run_sweep.sh      # print only
#   SKIP_DEPLOY=1 CONFIGS=B BENCH_HOST=10.0.0.5 BENCH_PORT=31234 bash scripts/bench/run_sweep.sh
#
# Environment knobs (all optional):
#   CONFIGS="A B C"  CONCURRENCIES="64 128 256"  RESULTS_DIR (default reports/bench_<timestamp>)
#   INPUT_LEN=256  OUTPUT_LEN=1024  NUM_REQUESTS (default: 3 x concurrency, chosen by the client)
#   WARMUP_REQUESTS (default: equal to the concurrency, 32 tokens each)
#   MODEL_PATH (default Qwen/Qwen3-30B-A3B-FP8)  TOKENIZER (default: MODEL_PATH)  PYTHON (default python3)
#   SKIP_DEPLOY=1   use an already-running server (needs BENCH_HOST/BENCH_PORT; use a single config)
#   SKIP_TEARDOWN=1 leave the deployment running after each config
#   C_POD_NAME (default h100-afd-serve)  DRY_RUN=1
#   CLIENT_MODE=local|pod  where the load generator runs (default local).
#     pod: runs serve_bench.py in a CPU-only pod (bench-client) in the same namespace, talking to the
#          in-cluster Service names. Use this when the NodePorts are not reachable from this machine.
#          The pod mounts PVC_NAME (HF cache, for the tokenizer) and pip-installs transformers.
#   CLIENT_IMAGE (default docker.io/python:3.12-slim)  RUN_TIMEOUT (seconds per run, default 5400)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
K8S="$REPO_ROOT/docker/h100/k8s"

CONFIGS="${CONFIGS:-A B C}"
CONCURRENCIES="${CONCURRENCIES:-64 128 256}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-1024}"
MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
TOKENIZER="${TOKENIZER:-$MODEL}"
PYTHON="${PYTHON:-python3}"
C_BASE="${C_POD_NAME:-h100-afd-serve}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DEPLOY="${SKIP_DEPLOY:-0}"
SKIP_TEARDOWN="${SKIP_TEARDOWN:-0}"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/reports/bench_$(date +%Y%m%d_%H%M%S)}"
CLIENT_MODE="${CLIENT_MODE:-local}"
CLIENT_IMAGE="${CLIENT_IMAGE:-docker.io/python:3.12-slim}"
CLIENT_POD="bench-client"
RUN_TIMEOUT="${RUN_TIMEOUT:-5400}"
PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
SERVE_PORT_IN="${SERVE_PORT:-8000}"

# Server cap shared by all three configs so only the topology differs.
CAP_ENV=(MAX_RUNNING_REQUESTS=256 DECODE_GRAPH_BS=8,16,32,64,128,256 CUDA_GRAPH_MAX_BS=256 MODEL_PATH="$MODEL")

CURRENT_CONFIG=""

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ $*"
    return 0
  fi
  "$@"
}

label_of() {
  case "$1" in
    A) echo 1n4g ;;
    B) echo 1n8g ;;
    C) echo 2n4g ;;
    *) echo "unknown config '$1' (expected A, B or C)" >&2; return 1 ;;
  esac
}

deploy_config() {
  case "$1" in
    A) run env "${CAP_ENV[@]}" GPU_COUNT=4 ATTN_DP=1 ATTN_TP=2 MLP_DP=2 MLP_TP=1 EP=2 \
         bash "$K8S/deploy_serve.sh" ;;
    B) run env "${CAP_ENV[@]}" GPU_COUNT=8 ATTN_DP=1 ATTN_TP=4 MLP_DP=4 MLP_TP=1 EP=4 \
         bash "$K8S/deploy_serve.sh" ;;
    C) run env "${CAP_ENV[@]}" POD_NAME="$C_BASE" GPU_COUNT=4 ATTN_DP=1 ATTN_TP=4 MLP_DP=1 MLP_TP=4 EP=4 \
         bash "$K8S/deploy_serve_multinode.sh" ;;
  esac
}

teardown_config() {
  [[ "$SKIP_DEPLOY" == "1" || "$SKIP_TEARDOWN" == "1" ]] && return 0
  echo "=== tearing down config $1 ==="
  case "$1" in
    A|B)
      run kubectl delete pod fastafd-qwen30b --ignore-not-found
      run kubectl delete service fastafd-qwen30b-service --ignore-not-found
      run kubectl delete configmap fastafd-serve-entry --ignore-not-found ;;
    C)
      run kubectl delete pod "${C_BASE}-head" "${C_BASE}-worker" --ignore-not-found
      run kubectl delete service "${C_BASE}-head" "${C_BASE}-worker" "${C_BASE}-api" --ignore-not-found
      run kubectl delete configmap "${C_BASE}-head-entry" "${C_BASE}-worker-entry" "${C_BASE}-local-patch" --ignore-not-found ;;
  esac
}

# --- in-cluster client pod (CLIENT_MODE=pod) -------------------------------------
create_client_pod() {
  local ns fsgroup
  echo "=== creating client pod $CLIENT_POD ($CLIENT_IMAGE) ==="
  ns="$(kubectl config view --minify -o jsonpath='{..namespace}')"
  fsgroup="$(kubectl get ns "$ns" -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null | cut -d/ -f1)"
  kubectl delete pod "$CLIENT_POD" --ignore-not-found >/dev/null
  kubectl create configmap bench-client-src --from-file=serve_bench.py="$HERE/serve_bench.py" \
    --dry-run=client -o yaml | kubectl apply -f - || return 1
  kubectl apply -f - <<EOF || return 1
apiVersion: v1
kind: Pod
metadata:
  name: $CLIENT_POD
  labels: {app: bench-client}
spec:
  restartPolicy: Never
  securityContext:
    fsGroup: ${fsgroup:-1000}
  volumes:
    - {name: model-storage, persistentVolumeClaim: {claimName: $PVC}}
    - {name: src, configMap: {name: bench-client-src}}
  containers:
    - name: client
      image: $CLIENT_IMAGE
      command: ["/bin/bash", "-c"]
      args:
        - |
          set -e
          export HOME=/tmp
          pip install -q --no-cache-dir --target /tmp/pylib transformers
          touch /tmp/ready
          exec sleep infinity
      env:
        - {name: HF_HOME, value: /models/.hf_home}
        - {name: HF_HUB_OFFLINE, value: "1"}
        - {name: PYTHONPATH, value: /tmp/pylib}
        - {name: PYTHONUNBUFFERED, value: "1"}
      resources:
        requests: {cpu: "4", memory: 8Gi}
        limits: {cpu: "16", memory: 16Gi}
      volumeMounts:
        - {name: model-storage, mountPath: /models}
        - {name: src, mountPath: /bench, readOnly: true}
EOF
  kubectl wait pod "$CLIENT_POD" --for=jsonpath='{.status.phase}'=Running --timeout=10m || return 1
  local deadline=$((SECONDS + 600))
  until kubectl exec "$CLIENT_POD" -- test -f /tmp/ready 2>/dev/null; do
    (( SECONDS < deadline )) || { echo "client pod did not finish pip install" >&2; kubectl logs "$CLIENT_POD" --tail=30 >&2; return 1; }
    sleep 5
  done
  echo "client pod ready"
}

delete_client_pod() {
  [[ "$CLIENT_MODE" == "pod" && "$DRY_RUN" != "1" ]] || return 0
  kubectl delete pod "$CLIENT_POD" --ignore-not-found >/dev/null 2>&1
  kubectl delete configmap bench-client-src --ignore-not-found >/dev/null 2>&1
}

# Run one serve_bench.py invocation inside the client pod: start it detached, poll for
# completion (so no single kubectl exec has to stay open for the whole run), then pull the
# log and JSON back. Args: <label> <conc> <serve_bench args...>
pod_bench() {
  local label="$1" conc="$2"; shift 2
  local tag="${label}_c${conc}" quoted rc="" deadline=$((SECONDS + RUN_TIMEOUT))
  printf -v quoted '%q ' "$@"
  kubectl exec "$CLIENT_POD" -- bash -c "rm -f /tmp/$tag.done /tmp/$tag.json; \
    setsid nohup bash -c 'python /bench/serve_bench.py $quoted --output /tmp/$tag.json > /tmp/$tag.log 2>&1; echo \$? > /tmp/$tag.done' </dev/null >/dev/null 2>&1 &" \
    || return 1
  while true; do
    rc="$(kubectl exec "$CLIENT_POD" -- cat "/tmp/$tag.done" 2>/dev/null || true)"
    [[ -n "$rc" ]] && break
    if (( SECONDS >= deadline )); then
      echo "run $tag timed out after ${RUN_TIMEOUT}s" >&2
      kubectl exec "$CLIENT_POD" -- pkill -f serve_bench.py 2>/dev/null || true
      rc=124; break
    fi
    sleep 20
  done
  kubectl exec "$CLIENT_POD" -- cat "/tmp/$tag.log" 2>/dev/null | tee "$RESULTS_DIR/$tag.log"
  kubectl exec "$CLIENT_POD" -- cat "/tmp/$tag.json" > "$RESULTS_DIR/$tag.json" 2>/dev/null \
    || { rm -f "$RESULTS_DIR/$tag.json"; echo "no result JSON for $tag" >&2; }
  [[ "$rc" == "0" ]]
}

# Sets HOST and PORT for the client.
resolve_endpoint() {
  local pod svc
  if [[ "$CLIENT_MODE" == "pod" ]]; then
    case "$1" in
      A|B) HOST=fastafd-qwen30b-service ;;
      C)   HOST="${C_BASE}-api" ;;
    esac
    PORT="$SERVE_PORT_IN"; return 0
  fi
  if [[ -n "${BENCH_HOST:-}" && -n "${BENCH_PORT:-}" ]]; then
    HOST="$BENCH_HOST"; PORT="$BENCH_PORT"; return 0
  fi
  if [[ "$SKIP_DEPLOY" == "1" ]]; then
    echo "SKIP_DEPLOY=1 requires BENCH_HOST and BENCH_PORT" >&2; return 1
  fi
  case "$1" in
    A|B) pod=fastafd-qwen30b; svc=fastafd-qwen30b-service ;;
    C)   pod="${C_BASE}-head"; svc="${C_BASE}-api" ;;
  esac
  if [[ "$DRY_RUN" == "1" ]]; then
    HOST="<node-ip>"; PORT="<nodePort>"; return 0
  fi
  HOST="${BENCH_HOST:-$(kubectl get pod "$pod" -o jsonpath='{.status.hostIP}')}"
  PORT="${BENCH_PORT:-$(kubectl get service "$svc" -o jsonpath='{.spec.ports[0].nodePort}')}"
  [[ -n "$HOST" && -n "$PORT" ]] || { echo "could not resolve endpoint for $pod/$svc" >&2; return 1; }
}

# Returns non-zero only if the config could not be deployed/reached; individual
# benchmark failures are recorded in their result JSON and do not abort the config.
run_config() {
  local cfg="$1" label conc
  label="$(label_of "$cfg")" || return 1
  CURRENT_CONFIG="$cfg"
  echo ""
  echo "################ config $cfg ($label) ################"

  if [[ "$SKIP_DEPLOY" != "1" ]]; then
    deploy_config "$cfg" || { echo "deploy failed for config $cfg" >&2; return 1; }
  fi
  resolve_endpoint "$cfg" || return 1
  echo "endpoint: http://$HOST:$PORT"

  if [[ "$CLIENT_MODE" == "pod" && "$DRY_RUN" != "1" ]]; then
    kubectl exec -i "$CLIENT_POD" -- env SERVE_HOST="$HOST" SERVE_PORT="$PORT" bash -s \
      < "$K8S/smoke_test_serve.sh" || { echo "smoke test failed for config $cfg" >&2; return 1; }
  else
    run env SERVE_HOST="$HOST" SERVE_PORT="$PORT" bash "$K8S/smoke_test_serve.sh" \
      || { echo "smoke test failed for config $cfg" >&2; return 1; }
  fi

  for conc in $CONCURRENCIES; do
    local args=(--host "$HOST" --port "$PORT" --model "$MODEL" --tokenizer "$TOKENIZER"
                --concurrency "$conc" --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN"
                --warmup-requests "${WARMUP_REQUESTS:-$conc}"
                --label "$label" --output "$RESULTS_DIR/${label}_c${conc}.json")
    [[ -n "${NUM_REQUESTS:-}" ]] && args+=(--num-requests "$NUM_REQUESTS")
    echo "--- $label concurrency=$conc ---"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "+ [$CLIENT_MODE] $PYTHON $HERE/serve_bench.py ${args[*]}"
    elif [[ "$CLIENT_MODE" == "pod" ]]; then
      # --output is added by pod_bench (path inside the pod); drop the local one.
      local pargs=("${args[@]}")
      local idx
      for idx in "${!pargs[@]}"; do [[ "${pargs[$idx]}" == "--output" ]] && { unset 'pargs[idx]' 'pargs[idx+1]'; break; }; done
      pod_bench "$label" "$conc" "${pargs[@]}" \
        || echo "benchmark run $label c=$conc reported failures (kept in its result JSON)" >&2
    else
      "$PYTHON" "$HERE/serve_bench.py" "${args[@]}" 2>&1 | tee "$RESULTS_DIR/${label}_c${conc}.log"
      [[ ${PIPESTATUS[0]} -eq 0 ]] || echo "benchmark run $label c=$conc reported failures (kept in its result JSON)" >&2
    fi
  done
  return 0
}

cleanup() {
  [[ -n "$CURRENT_CONFIG" ]] && teardown_config "$CURRENT_CONFIG"
  CURRENT_CONFIG=""
}
trap 'echo "interrupted"; cleanup; delete_client_pod; exit 130' INT TERM

# --- pre-flight ---------------------------------------------------------------
for cfg in $CONFIGS; do label_of "$cfg" >/dev/null || exit 2; done
if [[ "$DRY_RUN" != "1" ]]; then
  if [[ "$SKIP_DEPLOY" != "1" ]]; then
    command -v kubectl >/dev/null || { echo "kubectl not found" >&2; exit 2; }
    command -v envsubst >/dev/null || { echo "envsubst not found" >&2; exit 2; }
    if [[ " $CONFIGS " =~ " A " || " $CONFIGS " =~ " B " ]]; then
      : "${IMAGE:?IMAGE must be set for configs A and B}"
    fi
  fi
  if [[ "$CLIENT_MODE" == "pod" ]]; then
    command -v kubectl >/dev/null || { echo "kubectl not found" >&2; exit 2; }
  else
    "$PYTHON" -c "import transformers" 2>/dev/null \
      || { echo "PYTHON='$PYTHON' cannot import transformers (needed to build the prompt)" >&2; exit 2; }
  fi
fi
[[ "$CLIENT_MODE" == "local" || "$CLIENT_MODE" == "pod" ]] || { echo "CLIENT_MODE must be local or pod" >&2; exit 2; }
[[ "$DRY_RUN" == "1" ]] || mkdir -p "$RESULTS_DIR"
echo "results_dir=$RESULTS_DIR configs=[$CONFIGS] concurrencies=[$CONCURRENCIES] in=$INPUT_LEN out=$OUTPUT_LEN dry_run=$DRY_RUN"

# --- sweep --------------------------------------------------------------------
if [[ "$CLIENT_MODE" == "pod" && "$DRY_RUN" != "1" ]]; then
  create_client_pod || { echo "could not create the client pod" >&2; delete_client_pod; exit 1; }
elif [[ "$CLIENT_MODE" == "pod" ]]; then
  echo "+ create client pod $CLIENT_POD ($CLIENT_IMAGE, PVC $PVC)"
fi
failed_configs=()
for cfg in $CONFIGS; do
  if ! run_config "$cfg"; then
    failed_configs+=("$cfg")
    if [[ "$DRY_RUN" != "1" ]]; then
      echo "config $cfg: deploy/smoke test failed; see sweep output above" > "$RESULTS_DIR/$(label_of "$cfg").failed"
    fi
  fi
  cleanup
done

delete_client_pod

# --- summary ------------------------------------------------------------------
if [[ "$DRY_RUN" == "1" ]]; then
  echo "+ $PYTHON $HERE/summarize.py $RESULTS_DIR"
else
  "$PYTHON" "$HERE/summarize.py" "$RESULTS_DIR" || echo "no results to summarize" >&2
fi
if (( ${#failed_configs[@]} )); then
  echo "FAILED configs: ${failed_configs[*]}" >&2
  exit 1
fi
echo "=== SWEEP DONE === $RESULTS_DIR"
