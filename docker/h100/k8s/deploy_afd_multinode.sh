#!/usr/bin/env bash
# Deploy the 2-pod multi-node AFD run: attn-only head (4 GPUs) + MLP-only worker (4 GPUs).
# Creates two headless Services, two ConfigMaps, and two Pods, then returns.
#
# Full 8-GPU topology split across two 4-GPU nodes:
#   ATTN_DP=1 ATTN_TP=4 MLP_DP=1 MLP_TP=4 EP=4  (each pod gets 4 GPUs)
#
#   ATTN_DP=1 ATTN_TP=4 MLP_DP=1 MLP_TP=4 EP=4 GPU_COUNT=4 RUN_VLLM_ALIGNMENT=1 \
#     bash docker/h100/k8s/deploy_afd_multinode.sh
#
# Follow with:
#   kubectl logs -f pod/${POD_NAME:-h100-afd-mn}-head  -c afd   # server + results
#   kubectl logs -f pod/${POD_NAME:-h100-afd-mn}-worker -c afd  # worker join log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="${POD_NAME:-h100-afd-mn}"
export TEMPLATE_POD="$BASE"
export TEMPLATE_REPO="${REPO_URL:-https://github.com/itaijj/FastAFD.git}"
export TEMPLATE_BRANCH="${BRANCH:-h100-deepep-baseline}"
export TEMPLATE_PVC="${PVC_NAME:-deepseek-v2-lite-pvc}"
# GPUs *per pod* — each pod gets its own 4-GPU node.
export TEMPLATE_GPU="${GPU_COUNT:-4}"
export TEMPLATE_ATTN_DP="${ATTN_DP:-1}"
export TEMPLATE_ATTN_TP="${ATTN_TP:-4}"
export TEMPLATE_MLP_DP="${MLP_DP:-1}"
export TEMPLATE_MLP_TP="${MLP_TP:-4}"
export TEMPLATE_EP="${EP:-4}"
export TEMPLATE_NUM_PROMPTS="${NUM_PROMPTS:-32}"
export TEMPLATE_PROMPT_LEN="${PROMPT_LEN:-512}"
export TEMPLATE_MAX_TOKENS="${MAX_TOKENS:-8}"
export TEMPLATE_KEEP_ALIVE_S="${KEEP_ALIVE_S:-900}"
export TEMPLATE_RDMA="${RDMA:-1}"
export TEMPLATE_VLLM_ALIGN="${RUN_VLLM_ALIGNMENT:-0}"
export TEMPLATE_RAY_PORT="${RAY_HEAD_PORT:-6379}"
EXCLUDE_NODES="${EXCLUDE_NODES:-pokprod-b93r43s1}"
export TEMPLATE_EXCLUDE_NODES="\"${EXCLUDE_NODES//,/\", \"}\""

NS="$(kubectl config view --minify -o jsonpath='{..namespace}')"
FS_GROUP="$(kubectl get ns "$NS" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
export TEMPLATE_FSGROUP="${FS_GROUP:-1000}"

echo "namespace=$NS"
echo "head=${BASE}-head  worker=${BASE}-worker"
echo "gpus_per_pod=$TEMPLATE_GPU  attn=dp${TEMPLATE_ATTN_DP}/tp${TEMPLATE_ATTN_TP}  mlp=dp${TEMPLATE_MLP_DP}/tp${TEMPLATE_MLP_TP}/ep${TEMPLATE_EP}"
echo "pvc=$TEMPLATE_PVC  rdma=$TEMPLATE_RDMA  vllm_align=$TEMPLATE_VLLM_ALIGN"

kubectl get pvc "$TEMPLATE_PVC" >/dev/null

# ConfigMap for head entry script
kubectl create configmap "${BASE}-head-entry" \
  --from-file=afd_entry_head.sh="$HERE/afd_entry_head.sh" \
  --dry-run=client -o yaml | kubectl apply -f -

# ConfigMap for worker entry script
kubectl create configmap "${BASE}-worker-entry" \
  --from-file=afd_entry_worker.sh="$HERE/afd_entry_worker.sh" \
  --dry-run=client -o yaml | kubectl apply -f -

# Tear down any previous run with the same base name
kubectl delete pod "${BASE}-head" "${BASE}-worker" --ignore-not-found
kubectl delete service "${BASE}-head" "${BASE}-worker" --ignore-not-found

# Apply all four resources (2 Services + 2 Pods) from the single template file
envsubst '${TEMPLATE_POD} ${TEMPLATE_FSGROUP} ${TEMPLATE_REPO} ${TEMPLATE_BRANCH} \
  ${TEMPLATE_PVC} ${TEMPLATE_GPU} ${TEMPLATE_ATTN_DP} ${TEMPLATE_ATTN_TP} \
  ${TEMPLATE_MLP_DP} ${TEMPLATE_MLP_TP} ${TEMPLATE_EP} \
  ${TEMPLATE_NUM_PROMPTS} ${TEMPLATE_PROMPT_LEN} ${TEMPLATE_MAX_TOKENS} \
  ${TEMPLATE_KEEP_ALIVE_S} ${TEMPLATE_RDMA} ${TEMPLATE_VLLM_ALIGN} \
  ${TEMPLATE_EXCLUDE_NODES} ${TEMPLATE_RAY_PORT}' \
  < "$HERE/afd-multinode.yaml" | kubectl apply -f -

echo ""
echo "Deployed. Follow logs with:"
echo "  kubectl logs -f pod/${BASE}-head   -c afd"
echo "  kubectl logs -f pod/${BASE}-worker -c afd"
echo "  POD_NAME=${BASE}-head bash $HERE/status.sh"
