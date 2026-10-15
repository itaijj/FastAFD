#!/usr/bin/env bash
# Wait until two distinct H100 nodes each have >= NEED_GPUS free GPUs,
# then deploy the multinode AFD pod pair pinned to those two nodes.
#
# Usage:
#   ATTN_DP=1 ATTN_TP=4 MLP_DP=1 MLP_TP=4 EP=4 GPU_COUNT=4 \
#     RUN_VLLM_ALIGNMENT=1 KEEP_ALIVE_S=3600 \
#     bash docker/h100/k8s/wait_and_deploy_multinode.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEED_GPUS="${GPU_COUNT:-4}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
EXCLUDE_NODES="${EXCLUDE_NODES:-pokprod-b93r43s1}"

echo "=== wait_and_deploy_multinode: need 2 nodes each with >= $NEED_GPUS free GPUs ==="
echo "    excluded: $EXCLUDE_NODES"
echo "    polling every ${POLL_INTERVAL_S}s  (Ctrl-C to abort)"
echo

while true; do
  TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"
  GPU_FREE_OUT="$(bash "$HERE/gpu_free.sh" 2>/dev/null || true)"

  CANDIDATES=()
  while IFS= read -r line; do
    [[ "$line" =~ alloc=([0-9]+).*used=([0-9]+) ]] || continue
    alloc="${BASH_REMATCH[1]}"
    used="${BASH_REMATCH[2]}"
    free=$(( alloc - used ))
    node="$(echo "$line" | awk '{print $1}')"

    # Skip excluded nodes
    skip=0
    IFS=',' read -ra excl <<< "$EXCLUDE_NODES"
    for e in "${excl[@]}"; do
      [[ "$node" == "$e" ]] && skip=1 && break
    done
    (( skip )) && continue

    echo "  [$TIMESTAMP] $node  alloc=$alloc  used=$used  free=$free"
    if (( free >= NEED_GPUS )); then
      CANDIDATES+=("$node")
    fi
  done <<< "$GPU_FREE_OUT"

  if (( ${#CANDIDATES[@]} >= 2 )); then
    HEAD_NODE="${CANDIDATES[0]}"
    WORKER_NODE="${CANDIDATES[1]}"
    echo
    echo "=== Found: head=$HEAD_NODE  worker=$WORKER_NODE — deploying ==="
    export HEAD_NODE WORKER_NODE
    exec bash "$HERE/deploy_afd_multinode.sh"
  fi

  echo "  [$TIMESTAMP] only ${#CANDIDATES[@]} node(s) with >= $NEED_GPUS free — sleeping ${POLL_INTERVAL_S}s"
  sleep "$POLL_INTERVAL_S"
done
