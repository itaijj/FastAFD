#!/usr/bin/env bash
# Deploy the H100 AFD (AG + EG) bring-up as a one-shot pod (deploy-afd-k8s
# conventions: envsubst template, ConfigMap entry script, namespace fsGroup).
# Creates the pod and returns; follow with:  kubectl logs -f pod/$POD_NAME -c afd
#
#   # full validation topology: 4 AG (dp1/tp4) + 4 EG (dp4/ep4) = 8 GPUs
#   ATTN_DP=1 ATTN_TP=4 MLP_DP=4 EP=4 GPU_COUNT=8 bash docker/h100/k8s/deploy_afd.sh
#   # smaller bring-up when 8 free GPUs on one node are not available: 2 AG + 2 EG
#   ATTN_DP=1 ATTN_TP=2 MLP_DP=2 EP=2 GPU_COUNT=4 bash docker/h100/k8s/deploy_afd.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEMPLATE_POD="${POD_NAME:-h100-afd-run}"
export TEMPLATE_REPO="${REPO_URL:-https://github.com/itaijj/FastAFD.git}"
export TEMPLATE_BRANCH="${BRANCH:-h100-deepep-baseline}"
export TEMPLATE_PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
export TEMPLATE_ATTN_DP="${ATTN_DP:-1}"
export TEMPLATE_ATTN_TP="${ATTN_TP:-2}"
export TEMPLATE_MLP_DP="${MLP_DP:-2}"
export TEMPLATE_EP="${EP:-2}"
export TEMPLATE_GPU="${GPU_COUNT:-$(( TEMPLATE_ATTN_DP * TEMPLATE_ATTN_TP + TEMPLATE_MLP_DP ))}"
export TEMPLATE_NUM_PROMPTS="${NUM_PROMPTS:-32}"
export TEMPLATE_PROMPT_LEN="${PROMPT_LEN:-512}"
export TEMPLATE_MAX_TOKENS="${MAX_TOKENS:-8}"
export TEMPLATE_KEEP_ALIVE_S="${KEEP_ALIVE_S:-900}"
# Nodes known to fail CUDA init (error 802), comma-separated.
EXCLUDE_NODES="${EXCLUDE_NODES:-pokprod-b93r43s1}"
export TEMPLATE_EXCLUDE_NODES="\"${EXCLUDE_NODES//,/\", \"}\""

NS="$(kubectl config view --minify -o jsonpath='{..namespace}')"
FS_GROUP="$(kubectl get ns "$NS" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
export TEMPLATE_FSGROUP="${FS_GROUP:-1000}"

echo "namespace=$NS pod=$TEMPLATE_POD gpus=$TEMPLATE_GPU attn=dp${TEMPLATE_ATTN_DP}/tp${TEMPLATE_ATTN_TP} mlp=dp${TEMPLATE_MLP_DP}/ep${TEMPLATE_EP} pvc=$TEMPLATE_PVC"

kubectl get pvc "$TEMPLATE_PVC" >/dev/null
kubectl create configmap "${TEMPLATE_POD}-entry" --from-file=afd_entry.sh="$HERE/afd_entry.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl delete pod "$TEMPLATE_POD" --ignore-not-found
envsubst '${TEMPLATE_POD} ${TEMPLATE_FSGROUP} ${TEMPLATE_REPO} ${TEMPLATE_BRANCH} ${TEMPLATE_PVC} ${TEMPLATE_GPU} ${TEMPLATE_ATTN_DP} ${TEMPLATE_ATTN_TP} ${TEMPLATE_MLP_DP} ${TEMPLATE_EP} ${TEMPLATE_NUM_PROMPTS} ${TEMPLATE_PROMPT_LEN} ${TEMPLATE_MAX_TOKENS} ${TEMPLATE_KEEP_ALIVE_S} ${TEMPLATE_EXCLUDE_NODES}' \
  < "$HERE/afd-pod.yaml" | kubectl apply -f -
