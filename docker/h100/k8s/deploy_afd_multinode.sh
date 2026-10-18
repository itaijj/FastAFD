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
# 2 + 2 GPUs (attention TP=2 on one node, MLP TP=2/EP=2 on another):
#   ATTN_DP=1 ATTN_TP=2 MLP_DP=1 MLP_TP=2 EP=2 GPU_COUNT=2 \
#     PVC_NAME=qwen3-30b-a3b-pvc MODEL_PATH=Qwen/Qwen3-30B-A3B \
#     HF_SECRET=hf-token HF_SECRET_KEY=HF_TOKEN \
#     bash docker/h100/k8s/deploy_afd_multinode.sh
#
# The pods always run on different nodes (required podAntiAffinity both ways).
# HEAD_NODE / WORKER_NODE optionally pin each pod to a hostname (default: unpinned).
# LOCAL_PATCH=1 ships local python/ and scripts/ changes that are not on
# REPO_URL@BRANCH yet as a ConfigMap; the clone init container applies them.
#
# Follow with:
#   kubectl logs -f pod/${POD_NAME:-h100-afd-mn}-head  -c afd   # server + results
#   kubectl logs -f pod/${POD_NAME:-h100-afd-mn}-worker -c afd  # worker join log
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE="${POD_NAME:-h100-afd-mn}"
export TEMPLATE_POD="$BASE"
export TEMPLATE_REPO="${REPO_URL:-https://github.com/ronenkat/FastAFD.git}"
export TEMPLATE_BRANCH="${BRANCH:-fastafd-4xh100-ronenkat-test1}"
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
export TEMPLATE_MODEL="${MODEL_PATH:-Qwen/Qwen3-30B-A3B-FP8}"
# Each pod JIT-builds its kernels independently, so ranks reach the first DeepEP
# barrier further apart than on one node; pods have no restart to retry warm.
export TEMPLATE_DEEPEP_TIMEOUT_S="${DEEPEP_TIMEOUT_S:-900}"
export TEMPLATE_HF_SECRET="${HF_SECRET:-hf-token-secret}"
export TEMPLATE_HF_SECRET_KEY="${HF_SECRET_KEY:-token}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"   # comma-separated hostnames to avoid, e.g. EXCLUDE_NODES=node-a,node-b
export TEMPLATE_EXCLUDE_NODES="\"${EXCLUDE_NODES//,/\", \"}\""
# Optional hostname pins; without them only the anti-affinity applies.
node_expr() {
  if [[ -n "$1" ]]; then
    echo "- {key: kubernetes.io/hostname, operator: In, values: [\"$1\"]}"
  else
    echo "# no hostname pin"
  fi
}
export TEMPLATE_HEAD_NODE_EXPR="$(node_expr "${HEAD_NODE:-}")"
export TEMPLATE_WORKER_NODE_EXPR="$(node_expr "${WORKER_NODE:-}")"

NS="$(kubectl config view --minify -o jsonpath='{..namespace}')"
FS_GROUP="$(kubectl get ns "$NS" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
export TEMPLATE_FSGROUP="${FS_GROUP:-1000}"

echo "namespace=$NS"
echo "head=${BASE}-head  worker=${BASE}-worker"
echo "gpus_per_pod=$TEMPLATE_GPU  attn=dp${TEMPLATE_ATTN_DP}/tp${TEMPLATE_ATTN_TP}  mlp=dp${TEMPLATE_MLP_DP}/tp${TEMPLATE_MLP_TP}/ep${TEMPLATE_EP}"
echo "pvc=$TEMPLATE_PVC  rdma=$TEMPLATE_RDMA  vllm_align=$TEMPLATE_VLLM_ALIGN"
echo "model=$TEMPLATE_MODEL  repo=$TEMPLATE_REPO@$TEMPLATE_BRANCH  deepep_timeout_s=$TEMPLATE_DEEPEP_TIMEOUT_S"
echo "head_node=${HEAD_NODE:-<any>}  worker_node=${WORKER_NODE:-<any>}  exclude=${EXCLUDE_NODES:-<none>}"

kubectl get pvc "$TEMPLATE_PVC" >/dev/null

# ConfigMap for head entry script
kubectl create configmap "${BASE}-head-entry" \
  --from-file=afd_entry_head.sh="$HERE/afd_entry_head.sh" \
  --dry-run=client -o yaml | kubectl apply -f -

# ConfigMap for worker entry script
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
kubectl delete service "${BASE}-head" "${BASE}-worker" --ignore-not-found

# Apply all four resources (2 Services + 2 Pods) from the single template file
envsubst '${TEMPLATE_POD} ${TEMPLATE_FSGROUP} ${TEMPLATE_REPO} ${TEMPLATE_BRANCH} \
  ${TEMPLATE_PVC} ${TEMPLATE_GPU} ${TEMPLATE_ATTN_DP} ${TEMPLATE_ATTN_TP} \
  ${TEMPLATE_MLP_DP} ${TEMPLATE_MLP_TP} ${TEMPLATE_EP} \
  ${TEMPLATE_NUM_PROMPTS} ${TEMPLATE_PROMPT_LEN} ${TEMPLATE_MAX_TOKENS} \
  ${TEMPLATE_KEEP_ALIVE_S} ${TEMPLATE_RDMA} ${TEMPLATE_VLLM_ALIGN} \
  ${TEMPLATE_EXCLUDE_NODES} ${TEMPLATE_RAY_PORT} \
  ${TEMPLATE_HEAD_NODE_EXPR} ${TEMPLATE_WORKER_NODE_EXPR} \
  ${TEMPLATE_MODEL} ${TEMPLATE_DEEPEP_TIMEOUT_S} \
  ${TEMPLATE_HF_SECRET} ${TEMPLATE_HF_SECRET_KEY}' \
  < "$HERE/afd-multinode.yaml" | kubectl apply -f -

echo ""
echo "Deployed. Follow logs with:"
echo "  kubectl logs -f pod/${BASE}-head   -c afd"
echo "  kubectl logs -f pod/${BASE}-worker -c afd"
echo "  POD_NAME=${BASE}-head bash $HERE/status.sh"
