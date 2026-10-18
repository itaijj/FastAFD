#!/usr/bin/env bash
# Deploy the FastAFD miniSGL persistent serve pod (fastafd-qwen30b) and its
# NodePort Service (fastafd-qwen30b-service) into the current kubectl namespace.
#
# Usage:
#   bash docker/h100/k8s/deploy_serve.sh
#
# Key overrides (all optional):
#   REPO_URL     git repo to clone for the FastAFD checkout (default: itaijj/FastAFD on GitHub)
#   BRANCH       branch/tag to use (default: h100-deepep-baseline)
#   PVC_NAME     name of the PVC holding the HF model cache (default: deepseek-v2-lite-pvc)
#   GPU_COUNT    GPUs to request; default 8 (4 AG + 4 EG)
#   ATTN_DP      attention DP size (default 1)
#   ATTN_TP      attention TP size (default 4)
#   MLP_DP       MLP DP size (default 4)
#   MLP_TP       MLP TP size (default 1)
#   EP           MLP EP size (default 4)
#   SERVE_PORT   miniSGL API port inside the container (default 8000)
#   NODE_PORT    K8s NodePort (default: empty = auto-assigned by K8s in 30000-32767)
#   RAY_CPUS     CPU count advertised to Ray (default 32)
#   MAX_RUNNING_REQUESTS  server cap on concurrently running requests (default 64)
#   DECODE_GRAPH_BS       CSV of decode CUDA-graph batch sizes (default 8,16,32,64)
#   CUDA_GRAPH_MAX_BS     max CUDA-graph batch size (default 64)
#   IMAGE        container image; REQUIRED (no default)
#   MODEL_PATH   HuggingFace model id or PVC path (default: Qwen/Qwen3-30B-A3B-FP8)
#   RDMA         set to 1 to request rdma/roce_gdr resource (default 1)
#   HF_SECRET    Secret holding the HF token, optional (default: hf-token-secret)
#   HF_SECRET_KEY key inside HF_SECRET (default: token)
#
# Example -- full 8xH100 topology:
#   IMAGE=registry.example.com/cuda:13.0.2-devel \
#   GPU_COUNT=8 ATTN_TP=4 MLP_DP=4 EP=4 \
#   bash docker/h100/k8s/deploy_serve.sh
#
# Benchmark presets (concurrency up to 256; see afd-serve-benchmark-plan.md):
#   config A, 1 node x 4 GPUs:
#     GPU_COUNT=4 ATTN_TP=2 MLP_DP=2 EP=2 \
#     MAX_RUNNING_REQUESTS=256 DECODE_GRAPH_BS=8,16,32,64,128,256 CUDA_GRAPH_MAX_BS=256
#   config B, 1 node x 8 GPUs:
#     GPU_COUNT=8 ATTN_TP=4 MLP_DP=4 EP=4 \
#     MAX_RUNNING_REQUESTS=256 DECODE_GRAPH_BS=8,16,32,64,128,256 CUDA_GRAPH_MAX_BS=256
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
export TEMPLATE_IMAGE="${IMAGE:?IMAGE must be set to the container image}"
export TEMPLATE_MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
export TEMPLATE_REPO="${REPO_URL:-https://github.com/itaijj/FastAFD.git}"
export TEMPLATE_BRANCH="${BRANCH:-h100-deepep-baseline}"
export TEMPLATE_PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
export TEMPLATE_GPU="${GPU_COUNT:-8}"
export TEMPLATE_ATTN_DP="${ATTN_DP:-1}"
export TEMPLATE_ATTN_TP="${ATTN_TP:-4}"
export TEMPLATE_MLP_DP="${MLP_DP:-4}"
export TEMPLATE_MLP_TP="${MLP_TP:-1}"
export TEMPLATE_EP="${EP:-4}"
export TEMPLATE_SERVE_PORT="${SERVE_PORT:-8000}"
export TEMPLATE_RAY_CPUS="${RAY_CPUS:-32}"
export TEMPLATE_MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
export TEMPLATE_DECODE_GRAPH_BS="${DECODE_GRAPH_BS:-8,16,32,64}"
export TEMPLATE_CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-64}"
export TEMPLATE_HF_SECRET="${HF_SECRET:-hf-token-secret}"
export TEMPLATE_HF_SECRET_KEY="${HF_SECRET_KEY:-token}"

# NodePort: if NODE_PORT is set, emit "nodePort: <n>"; otherwise omit the line
# so K8s auto-assigns from 30000-32767.
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

POD_NAME="fastafd-qwen30b"
SVC_NAME="fastafd-qwen30b-service"
CONFIGMAP_NAME="fastafd-serve-entry"

echo "namespace=$NS"
echo "pod=$POD_NAME"
echo "service=$SVC_NAME"
echo "image=$TEMPLATE_IMAGE"
echo "model=$TEMPLATE_MODEL"
echo "pvc=$TEMPLATE_PVC"
echo "gpus=$TEMPLATE_GPU"
echo "topology=attn dp${TEMPLATE_ATTN_DP}/tp${TEMPLATE_ATTN_TP} mlp dp${TEMPLATE_MLP_DP}/tp${TEMPLATE_MLP_TP}/ep${TEMPLATE_EP}"
echo "serve_port=$TEMPLATE_SERVE_PORT"
if [[ -n "${NODE_PORT:-}" ]]; then
  echo "node_port=${NODE_PORT}"
else
  echo "node_port=auto-assigned"
fi
echo ""
echo "WARNING: this will DELETE and recreate pod/$POD_NAME if it exists."
echo "The Service ($SVC_NAME) is kept/updated idempotently."
echo ""

# ---------------------------------------------------------------------------
# Pre-flight: PVC must exist
# ---------------------------------------------------------------------------
kubectl get pvc "$TEMPLATE_PVC" >/dev/null \
  || { echo "PVC $TEMPLATE_PVC not found in namespace $NS"; exit 1; }

# ---------------------------------------------------------------------------
# Step 1: ConfigMap for the entrypoint script
# ---------------------------------------------------------------------------
echo "=== applying ConfigMap $CONFIGMAP_NAME ==="
kubectl create configmap "$CONFIGMAP_NAME" \
  --from-file=afd_serve_entry.sh="$HERE/afd_serve_entry.sh" \
  --dry-run=client -o yaml | kubectl apply -f -

# ---------------------------------------------------------------------------
# Step 2: Service (before pod -- pod binds rendezvous on startup)
# ---------------------------------------------------------------------------
echo "=== applying Service $SVC_NAME ==="
envsubst '${TEMPLATE_SERVE_PORT} ${TEMPLATE_NODE_PORT_LINE}' \
  < "$HERE/fastafd-serve-service.yaml" | kubectl apply -f -

# ---------------------------------------------------------------------------
# Step 3: Pod
# ---------------------------------------------------------------------------
echo "=== (re)deploying pod $POD_NAME ==="
kubectl delete pod "$POD_NAME" --ignore-not-found
envsubst '${TEMPLATE_IMAGE} ${TEMPLATE_MODEL} ${TEMPLATE_GPU} ${TEMPLATE_PVC} ${TEMPLATE_FSGROUP} ${TEMPLATE_SERVE_PORT} ${TEMPLATE_ATTN_DP} ${TEMPLATE_ATTN_TP} ${TEMPLATE_MLP_DP} ${TEMPLATE_MLP_TP} ${TEMPLATE_EP} ${TEMPLATE_RAY_CPUS} ${TEMPLATE_MAX_RUNNING_REQUESTS} ${TEMPLATE_DECODE_GRAPH_BS} ${TEMPLATE_CUDA_GRAPH_MAX_BS} ${TEMPLATE_REPO} ${TEMPLATE_BRANCH} ${TEMPLATE_HF_SECRET} ${TEMPLATE_HF_SECRET_KEY}' \
  < "$HERE/fastafd-serve-pod.yaml" | kubectl apply -f -

# ---------------------------------------------------------------------------
# Step 4: Wait for pod Running (up to 15 min)
# ---------------------------------------------------------------------------
echo "=== waiting for pod/$POD_NAME to reach Running (timeout 15m) ==="
kubectl wait pod "$POD_NAME" \
  --for=jsonpath='{.status.phase}'=Running \
  --timeout=15m \
  || { kubectl describe pod "$POD_NAME" | tail -40; exit 1; }
echo "pod/$POD_NAME is Running"

# ---------------------------------------------------------------------------
# Step 5: Wait for "=== AFD SERVE READY ===" in logs (up to 60 min)
# ---------------------------------------------------------------------------
echo "=== waiting for AFD SERVE READY in pod logs (timeout 60m) ==="
deadline=$(( $(date +%s) + 3600 ))
while true; do
  logs="$(kubectl logs "pod/$POD_NAME" 2>/dev/null || true)"
  if echo "$logs" | grep -q "AFD SERVE READY"; then
    echo "pod/$POD_NAME is ready"
    break
  fi
  if echo "$logs" | grep -qE "AFD SERVE FAIL|ERROR:"; then
    echo "ERROR detected in pod logs:" >&2
    echo "$logs" | tail -50 >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "Timed out waiting for AFD SERVE READY" >&2
    kubectl logs "$POD_NAME" --tail=200 >&2
    exit 1
  fi
  sleep 15
done

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
NODE_IP="$(kubectl get pod "$POD_NAME" \
  -o jsonpath='{.status.hostIP}' 2>/dev/null || echo '<node-ip>')"
ASSIGNED_NODE_PORT="$(kubectl get service "$SVC_NAME" \
  -o jsonpath='{.spec.ports[0].nodePort}' 2>/dev/null || echo '<nodePort>')"
CLUSTER_IP="$(kubectl get service "$SVC_NAME" \
  -o jsonpath='{.spec.clusterIP}' 2>/dev/null || echo '<clusterIP>')"

echo ""
echo "=== fastafd-qwen30b READY ==="
echo "  in-cluster:  http://${SVC_NAME}:${TEMPLATE_SERVE_PORT}/v1"
echo "  in-cluster:  http://${CLUSTER_IP}:${TEMPLATE_SERVE_PORT}/v1"
echo "  node-port:   http://${NODE_IP}:${ASSIGNED_NODE_PORT}/v1"
echo ""
echo "Quick health check:"
echo "  kubectl run curl-test --image=curlimages/curl --rm -it --restart=Never -- \\"
echo "    curl -s http://${SVC_NAME}:${TEMPLATE_SERVE_PORT}/v1/models"
echo ""
echo "Smoke test (from outside cluster):"
echo "  SERVE_HOST=${NODE_IP} SERVE_PORT=${ASSIGNED_NODE_PORT} bash docker/h100/k8s/smoke_test_serve.sh"
echo ""
echo "Teardown:"
echo "  kubectl delete pod $POD_NAME"
echo "  kubectl delete service $SVC_NAME"
echo "  kubectl delete configmap $CONFIGMAP_NAME"
