#!/usr/bin/env bash
# Summarize test-pod logs: bash docker/h100/k8s/summarize_logs.sh <pod>...
for p in "$@"; do
  f="/tmp/${p}.log"
  kubectl logs "$p" -c test > "$f" 2>&1
  echo "##### $p ($(wc -l < "$f") lines) node=$(kubectl get pod "$p" -o jsonpath='{.spec.nodeName}')"
  grep -nE "^(torch |commit:|PASS|FAIL|SKIP|=== )|H100|Traceback|Error|error:|FAILED:|\[[0-9]+/[0-9]+\]" "$f" \
    | grep -vE "Call parameter|addrspacecast" | head -40
  echo "--- first non-LLVM lines before 'Call parameter' ---"
  awk '/Call parameter/{exit} {print}' "$f" | tail -25
done
