
#!/usr/bin/env bash
# Wait until one H100 node has >= NEED_GPUS free GPUs, then deploy the AFD pod.
#
# Usage (8-GPU full topology: 4 AG dp1/tp4 + 4 EG dp4/ep4):
#   ATTN_DP=1 ATTN_TP=4 MLP_DP=4 EP=4 GPU_COUNT=8 RUN_VLLM_ALIGNMENT=1 \
#     bash docker/h100/k8s/wait_and_deploy_afd.sh
#
# The script polls gpu_free.sh every POLL_INTERVAL_S seconds (default 60).
# Once a qualifying node is found it delegates straight to deploy_afd.sh with
# all inherited env vars intact (no re-export needed; they are already in env).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEED_GPUS="${GPU_COUNT:-8}"
POLL_INTERVAL_S="${POLL_INTERVAL_S:-60}"

echo "=== wait_and_deploy_afd: need $NEED_GPUS free GPUs on one H100 node ==="
echo "    polling every ${POLL_INTERVAL_S}s  (Ctrl-C to abort)"
echo

while true; do
  TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S')"

  # Parse gpu_free.sh output: lines like  "<node> alloc=N used=M"
  # Capture into a variable so gpu_free.sh's exit code (non-zero when
  # resourcequota is absent) doesn't kill the loop under set -o pipefail.
  GPU_FREE_OUT="$(bash "$HERE/gpu_free.sh" 2>/dev/null || true)"

  FOUND_NODE=""
  FOUND_FREE=0
  while IFS= read -r line; do
    # Skip lines that don't match the per-node format
    [[ "$line" =~ alloc=([0-9]+).*used=([0-9]+) ]] || continue
    alloc="${BASH_REMATCH[1]}"
    used="${BASH_REMATCH[2]}"
    free=$(( alloc - used ))
    node="$(echo "$line" | awk '{print $1}')"
    echo "  [$TIMESTAMP] $node  alloc=$alloc  used=$used  free=$free"
    if (( free >= NEED_GPUS )); then
      FOUND_NODE="$node"
      FOUND_FREE="$free"
      # Don't break — print all nodes for visibility, then act below
    fi
  done <<< "$GPU_FREE_OUT"

  if [ -n "$FOUND_NODE" ]; then
    echo
    echo "=== Node '$FOUND_NODE' has $FOUND_FREE free GPUs (need $NEED_GPUS) — deploying ==="
    exec bash "$HERE/deploy_afd.sh"
    # exec replaces this process; we never reach the lines below
  fi

  echo "  [$TIMESTAMP] no node with >= $NEED_GPUS free GPUs yet — sleeping ${POLL_INTERVAL_S}s"
  sleep "$POLL_INTERVAL_S"
done
