#!/usr/bin/env bash
# Usage: status.sh  -> prints per-rank log tails + key events for the latest run in h100-afd-run
P=${POD_NAME:-h100-afd-run}
kubectl exec $P -c afd -- bash -c '
date
R=$(ls -dt /work/FastAFD/reports/afd_*/ray_logs | head -1)
for f in $R/attention_rank0.log $R/mlp_dp0_rank0.log; do
  echo "=== $(basename $f)"; tail -n 12 "$f" | cut -c1-260
done
echo "=== errors in any rank log"
grep -hE "Traceback|Error|error:|Assertion|FAILED" $R/*.log 2>/dev/null | sort | uniq -c | sort -rn | head -8 | cut -c1-300
echo "=== gpu"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
'
