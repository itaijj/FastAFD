#!/usr/bin/env bash
# Poll until 2 separate H100 nodes each have >= NEED_GPUS free GPUs (default 4),
# then deploy the 2-pod multi-node AFD run (attn head + MLP worker).
#
# Usage:
#   ATTN_DP=1 ATTN_TP=4 MLP_DP=1 MLP_TP=4 EP=4 GPU_COUNT=4 RUN_VLLM_ALIGNMENT=1 \
#     bash docker/h100/k8s/wait_and_deploy_multinode.sh
#
# Runs as a daemon (called via nohup setsid) — logs to /tmp/wait_multinode_deploy.log
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEED_GPUS="${GPU_COUNT:-4}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"
EXCLUDE="${EXCLUDE_NODES:-pokprod-b93r43s1}"

echo "=== wait_and_deploy_multinode: need 2 nodes with >=$NEED_GPUS free GPUs each ==="
echo "    exclude=$EXCLUDE  poll=${POLL_INTERVAL_S}s"
echo

while true; do
  TS="$(date '+%Y-%m-%d %H:%M:%S')"

  GPU_FREE_OUT="$(bash "$HERE/gpu_free.sh" 2>/dev/null || true)"

  QUALIFY=()
  while IFS= read -r line; do
    [[ "$line" =~ alloc=([0-9]+).*used=([0-9]+) ]] || continue
    node="$(echo "$line" | awk '{print $1}')"
    alloc="${BASH_REMATCH[1]}"
    used="${BASH_REMATCH[2]}"
    free=$(( alloc - used ))
    echo "  [$TS] $node  alloc=$alloc  used=$used  free=$free"
    # skip excluded node(s)
    for ex in ${EXCLUDE//,/ }; do
      [[ "$node" == "$ex" ]] && continue 2
    done
    (( free >= NEED_GPUS )) && QUALIFY+=("$node")
  done <<< "$GPU_FREE_OUT"

  echo "  [$TS] qualifying nodes: ${QUALIFY[*]:-none}"

  if (( ${#QUALIFY[@]} >= 2 )); then
    echo
    echo "=== ${QUALIFY[0]} and ${QUALIFY[1]} each have >=$NEED_GPUS free GPUs — deploying ==="
    exec bash "$HERE/deploy_afd_multinode.sh"
  fi

  echo "  [$TS] only ${#QUALIFY[@]} qualifying node(s), need 2 — sleeping ${POLL_INTERVAL_S}s"
  sleep "$POLL_INTERVAL_S"
done
