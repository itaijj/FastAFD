#!/usr/bin/env bash
# Deploy the persistent 2-node AFD OpenAI serve: attention head pod (4 GPUs) +
# MLP worker pod (4 GPUs) + NodePort Service for the API. Leaves everything running.
#
# Benchmark preset (config C, 2 nodes x 4 GPUs, 256-request cap):
#   MAX_RUNNING_REQUESTS=256 DECODE_GRAPH_BS=8,16,32,64,128,256 CUDA_GRAPH_MAX_BS=256 \
#     PVC_NAME=<pvc> bash docker/h100/k8s/deploy_serve_multinode.sh
#
# Topology defaults: attn dp1/tp4 on the head, mlp dp1/tp4/ep4 on the worker.
#
# Key overrides (all optional):
#   POD_NAME     base name (default: h100-afd-serve) -> <base>-head, <base>-worker, <base>-api
#   REPO_URL / BRANCH   git checkout for the pods (default: itaijj/FastAFD @ h100-deepep-baseline)
#   PVC_NAME     PVC with the HF model cache (default: deepseek-v2-lite-pvc)
#   GPU_COUNT    GPUs per pod (default 4)
#   ATTN_DP ATTN_TP MLP_DP MLP_TP EP    topology (defaults 1 4 1 4 4)
#   SERVE_PORT   API port (default 8000)      NODE_PORT  NodePort (default: auto-assigned)
#   MAX_RUNNING_REQUESTS (64)  DECODE_GRAPH_BS (8,16,32,64)  CUDA_GRAPH_MAX_BS (64)
#   KEEP_ALIVE_S worker lifetime in seconds (default 86400; the worker exits after this)
#   RDMA, HEAD_NODE, WORKER_NODE, EXCLUDE_NODES, DEEPEP_TIMEOUT_S, HF_SECRET, HF_SECRET_KEY,
#   MODEL_PATH, RAY_HEAD_PORT, LOCAL_PATCH=1   same meaning as deploy_afd_multinode.sh
#   WAIT_READY_S seconds to wait for AFD SERVE READY (default 5400)
#
# Teardown:
#   kubectl delete pod <base>-head <base>-worker; kubectl delete service <base>-head <base>-worker <base>-api
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="${POD_NAME:-h100-afd-serve}"
export TEMPLATE_POD="$BASE"
export TEMPLATE_REPO="${REPO_URL:-https://github.com/itaijj/FastAFD.git}"
export TEMPLATE_BRANCH="${BRANCH:-h100-deepep-baseline}"
export TEMPLATE_PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
export TEMPLATE_GPU="${GPU_COUNT:-4}"
export TEMPLATE_ATTN_DP="${ATTN_DP:-1}"
export TEMPLATE_ATTN_TP="${ATTN_TP:-4}"
export TEMPLATE_MLP_DP="${MLP_DP:-1}"
export TEMPLATE_MLP_TP="${MLP_TP:-4}"
export TEMPLATE_EP="${EP:-4}"
export TEMPLATE_SERVE_PORT="${SERVE_PORT:-8000}"
export TEMPLATE_MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
export TEMPLATE_DECODE_GRAPH_BS="${DECODE_GRAPH_BS:-8,16,32,64}"
export TEMPLATE_CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"
export TEMPLATE_KEEP_ALIVE_S="${KEEP_ALIVE_S:-86400}"
export TEMPLATE_RDMA="${RDMA:-1}"
export TEMPLATE_RAY_PORT="${RAY_HEAD_PORT:-6379}"
export TEMPLATE_MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
# Each pod JIT-builds its kernels independently, so ranks reach the first DeepEP
# barrier further apart than on one node; pods have no restart to retry warm.
export TEMPLATE_DEEPEP_TIMEOUT_S="${DEEPEP_TIMEOUT_S:-900}"
export TEMPLATE_HF_SECRET="${HF_SECRET:-hf-token-secret}"
export TEMPLATE_HF_SECRET_KEY="${HF_SECRET_KEY:-token}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"   # comma-separated hostnames to avoid
export TEMPLATE_EXCLUDE_NODES="\"${EXCLUDE_NODES//,/\", \"}\""
node_expr() {
  if [[ -n "$1" ]]; then
    echo "- {key: kubernetes.io/hostname, operator: In, values: [\"$1\"]}"
  else
    echo "# no hostname pin"
  fi
}
export TEMPLATE_HEAD_NODE_EXPR="$(node_expr "${HEAD_NODE:-}")"
export TEMPLATE_WORKER_NODE_EXPR="$(node_expr "${WORKER_NODE:-}")"
if [[ -n "${NODE_PORT:-}" ]]; then
  export TEMPLATE_NODE_PORT_LINE="nodePort: ${NODE_PORT}"
else
  export TEMPLATE_NODE_PORT_LINE="# nodePort: auto-assigned"
fi

NS="$(kubectl config view --minify -o jsonpath='{..namespace}')"
FS_GROUP="$(kubectl get ns "$NS" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
export TEMPLATE_FSGROUP="${FS_GROUP:-1000}"

echo "namespace=$NS"
echo "head=${BASE}-head  worker=${BASE}-worker  api_service=${BASE}-api"
echo "gpus_per_pod=$TEMPLATE_GPU  attn=dp${TEMPLATE_ATTN_DP}/tp${TEMPLATE_ATTN_TP}  mlp=dp${TEMPLATE_MLP_DP}/tp${TEMPLATE_MLP_TP}/ep${TEMPLATE_EP}"
echo "max_running=$TEMPLATE_MAX_RUNNING_REQUESTS  graph_bs=$TEMPLATE_DECODE_GRAPH_BS  cuda_graph_max_bs=$TEMPLATE_CUDA_GRAPH_MAX_BS"
echo "pvc=$TEMPLATE_PVC  rdma=$TEMPLATE_RDMA  model=$TEMPLATE_MODEL  repo=$TEMPLATE_REPO@$TEMPLATE_BRANCH"
echo "head_node=${HEAD_NODE:-<any>}  worker_node=${WORKER_NODE:-<any>}  exclude=${EXCLUDE_NODES:-<none>}"

kubectl get pvc "$TEMPLATE_PVC" >/dev/null \
  || { echo "PVC $TEMPLATE_PVC not found in namespace $NS"; exit 1; }

# ConfigMaps for the entry scripts (the worker script is the unchanged one-shot-run worker)
kubectl create configmap "${BASE}-head-entry" \
  --from-file=afd_serve_entry_head.sh="$HERE/afd_serve_entry_head.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create configmap "${BASE}-worker-entry" \
  --from-file=afd_entry_worker.sh="$HERE/afd_entry_worker.sh" \
  --dry-run=client -o yaml | kubectl apply -f -

# Optional: ship local python/ and scripts/ changes not yet on REPO_URL@BRANCH.
if [[ "${LOCAL_PATCH:-0}" == "1" ]]; then
  REPO_ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
  BASE_SHA="$(git ls-remote "$TEMPLATE_REPO" "refs/heads/$TEMPLATE_BRANCH" | cut -f1)"
  [[ -n "$BASE_SHA" ]] || { echo "branch $TEMPLATE_BRANCH not found on $TEMPLATE_REPO"; exit 1; }
  git -C "$REPO_ROOT" cat-file -e "$BASE_SHA^{commit}" 2>/dev/null \
    || { echo "remote tip $BASE_SHA is not in the local repo; git fetch first"; exit 1; }
  PATCH_FILE="$(mktemp)"
  git -C "$REPO_ROOT" diff --binary "$BASE_SHA" -- python scripts > "$PATCH_FILE"
  echo "local patch vs $TEMPLATE_BRANCH@${BASE_SHA:0:7}: $(wc -l < "$PATCH_FILE") lines"
  kubectl create configmap "${BASE}-local-patch" --from-file=local.patch="$PATCH_FILE" \
    --dry-run=client -o yaml | kubectl apply -f -
  rm -f "$PATCH_FILE"
else
  kubectl delete configmap "${BASE}-local-patch" --ignore-not-found
fi

# Tear down any previous run with the same base name
kubectl delete pod "${BASE}-head" "${BASE}-worker" --ignore-not-found
kubectl delete service "${BASE}-head" "${BASE}-worker" "${BASE}-api" --ignore-not-found

envsubst '${TEMPLATE_POD} ${TEMPLATE_FSGROUP} ${TEMPLATE_REPO} ${TEMPLATE_BRANCH} \
  ${TEMPLATE_PVC} ${TEMPLATE_GPU} ${TEMPLATE_ATTN_DP} ${TEMPLATE_ATTN_TP} \
  ${TEMPLATE_MLP_DP} ${TEMPLATE_MLP_TP} ${TEMPLATE_EP} \
  ${TEMPLATE_KEEP_ALIVE_S} ${TEMPLATE_RDMA} \
  ${TEMPLATE_EXCLUDE_NODES} ${TEMPLATE_RAY_PORT} \
  ${TEMPLATE_HEAD_NODE_EXPR} ${TEMPLATE_WORKER_NODE_EXPR} \
  ${TEMPLATE_MODEL} ${TEMPLATE_DEEPEP_TIMEOUT_S} \
  ${TEMPLATE_HF_SECRET} ${TEMPLATE_HF_SECRET_KEY} \
  ${TEMPLATE_SERVE_PORT} ${TEMPLATE_NODE_PORT_LINE} \
  ${TEMPLATE_MAX_RUNNING_REQUESTS} ${TEMPLATE_DECODE_GRAPH_BS} ${TEMPLATE_CUDA_GRAPH_MAX_BS}' \
  < "$HERE/afd-multinode-serve.yaml" | kubectl apply -f -

# Wait for the head pod to start, then for the server to come up.
echo "=== waiting for pod/${BASE}-head to reach Running (timeout 15m) ==="
kubectl wait pod "${BASE}-head" --for=jsonpath='{.status.phase}'=Running --timeout=15m \
  || { kubectl describe pod "${BASE}-head" | tail -40; exit 1; }

WAIT_READY_S="${WAIT_READY_S:-5400}"
echo "=== waiting for AFD SERVE READY in ${BASE}-head logs (timeout ${WAIT_READY_S}s) ==="
deadline=$(( $(date +%s) + WAIT_READY_S ))
while true; do
  logs="$(kubectl logs "pod/${BASE}-head" -c afd 2>/dev/null || true)"
  if grep -q "AFD SERVE READY" <<<"$logs"; then
    echo "pod/${BASE}-head is ready"
    break
  fi
  if grep -q "AFD SERVE FAIL" <<<"$logs"; then
    echo "AFD SERVE FAIL detected in head logs:" >&2
    tail -50 <<<"$logs" >&2
    exit 1
  fi
  phase="$(kubectl get pod "${BASE}-head" -o jsonpath='{.status.phase}' 2>/dev/null || echo Unknown)"
  if [[ "$phase" == "Failed" || "$phase" == "Succeeded" ]]; then
    echo "head pod ended with phase $phase" >&2
    tail -50 <<<"$logs" >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "Timed out waiting for AFD SERVE READY" >&2
    tail -100 <<<"$logs" >&2
    exit 1
  fi
  sleep 15
done

NODE_IP="$(kubectl get pod "${BASE}-head" -o jsonpath='{.status.hostIP}' 2>/dev/null || echo '<node-ip>')"
ASSIGNED_NODE_PORT="$(kubectl get service "${BASE}-api" -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo '<nodePort>')"

echo ""
echo "=== ${BASE} READY ==="
echo "  in-cluster:  http://${BASE}-api:${TEMPLATE_SERVE_PORT}/v1"
echo "  node-port:   http://${NODE_IP}:${ASSIGNED_NODE_PORT}/v1"
echo ""
echo "Smoke test:"
echo "  SERVE_HOST=${NODE_IP} SERVE_PORT=${ASSIGNED_NODE_PORT} bash docker/h100/k8s/smoke_test_serve.sh"
echo ""
echo "Teardown:"
echo "  kubectl delete pod ${BASE}-head ${BASE}-worker"
echo "  kubectl delete service ${BASE}-head ${BASE}-worker ${BASE}-api"
echo "  kubectl delete configmap ${BASE}-head-entry ${BASE}-worker-entry ${BASE}-local-patch --ignore-not-found"
