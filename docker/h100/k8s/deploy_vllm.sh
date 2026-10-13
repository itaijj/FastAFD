#!/usr/bin/env bash
# Deploy a plain vLLM server (no FastAFD) on H100 as a one-shot pod, following
# deploy-afd-k8s conventions: envsubst template, ConfigMap entry script,
# namespace fsGroup, Service before Pod, PVC-backed HF cache.
#
#   bash docker/h100/k8s/deploy_vllm.sh                         # Qwen3-30B-A3B-FP8 on 1 GPU
#   GPU_COUNT=4 bash docker/h100/k8s/deploy_vllm.sh             # tensor-parallel over 4 GPUs
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 bash docker/h100/k8s/deploy_vllm.sh
#
# Then:  kubectl logs -f pod/$POD_NAME -c vllm
#        kubectl port-forward pod/$POD_NAME 8000:8000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEMPLATE_POD="${POD_NAME:-h100-vllm}"
export TEMPLATE_PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
export TEMPLATE_MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
export TEMPLATE_GPU="${GPU_COUNT:-1}"
export TEMPLATE_PORT="${PORT:-8000}"
export TEMPLATE_MAXLEN="${MAX_MODEL_LEN:-4096}"
export TEMPLATE_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---enforce-eager --moe-backend triton}"
export TEMPLATE_KEEP_ALIVE_S="${KEEP_ALIVE_S:-1800}"
# Nodes known to fail CUDA init (error 802), comma-separated.
EXCLUDE_NODES="${EXCLUDE_NODES:-pokprod-b93r43s1}"
export TEMPLATE_EXCLUDE_NODES="\"${EXCLUDE_NODES//,/\", \"}\""

NS="$(kubectl config view --minify -o jsonpath='{..namespace}')"
FS_GROUP="$(kubectl get ns "$NS" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
export TEMPLATE_FSGROUP="${FS_GROUP:-1000}"

echo "namespace=$NS pod=$TEMPLATE_POD gpus=$TEMPLATE_GPU model=$TEMPLATE_MODEL pvc=$TEMPLATE_PVC"

kubectl get pvc "$TEMPLATE_PVC" >/dev/null
kubectl create configmap "${TEMPLATE_POD}-entry" --from-file=vllm_entry.sh="$HERE/vllm_entry.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl delete pod "$TEMPLATE_POD" --ignore-not-found
envsubst '${TEMPLATE_POD} ${TEMPLATE_FSGROUP} ${TEMPLATE_PVC} ${TEMPLATE_MODEL} ${TEMPLATE_GPU} ${TEMPLATE_PORT} ${TEMPLATE_MAXLEN} ${TEMPLATE_EXTRA_ARGS} ${TEMPLATE_KEEP_ALIVE_S} ${TEMPLATE_EXCLUDE_NODES}' \
  < "$HERE/vllm-pod.yaml" | kubectl apply -f -
