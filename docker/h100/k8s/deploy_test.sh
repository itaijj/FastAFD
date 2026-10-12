#!/usr/bin/env bash
# Deploy the SM90 expert-path test as a one-shot pod on a k8s/OpenShift H100
# node, following the deploy-afd-k8s skill conventions (envsubst templates,
# ConfigMap-mounted entry script, namespace fsGroup, bounded waits).
#   bash docker/h100/k8s/deploy_test.sh            # deploy + wait for result
#   kubectl delete pod h100-psum-moe-test; kubectl delete cm h100-psum-moe-test-entry
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TEMPLATE_POD="${POD_NAME:-h100-psum-moe-test}"
export TEMPLATE_REPO="${REPO_URL:-https://github.com/itaijj/FastAFD.git}"
export TEMPLATE_BRANCH="${BRANCH:-h100-deepep-baseline}"
export TEMPLATE_GPU="${GPU_COUNT:-1}"
# Nodes known to fail CUDA init (error 802: fabric manager not up), comma-separated.
EXCLUDE_NODES="${EXCLUDE_NODES:-pokprod-b93r43s1}"
export TEMPLATE_EXCLUDE_NODES="\"${EXCLUDE_NODES//,/\", \"}\""

NS="$(kubectl config view --minify -o jsonpath='{..namespace}')"
FS_GROUP="$(kubectl get ns "$NS" \
  -o jsonpath='{.metadata.annotations.openshift\.io/sa\.scc\.supplemental-groups}' 2>/dev/null \
  | cut -d/ -f1)"
export TEMPLATE_FSGROUP="${FS_GROUP:-1000}"

echo "namespace=$NS pod=$TEMPLATE_POD gpus=$TEMPLATE_GPU repo=$TEMPLATE_REPO@$TEMPLATE_BRANCH fsGroup=$TEMPLATE_FSGROUP"

kubectl create configmap "${TEMPLATE_POD}-entry" --from-file=pod_entry.sh="$HERE/pod_entry.sh" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl delete pod "$TEMPLATE_POD" --ignore-not-found
envsubst '${TEMPLATE_POD} ${TEMPLATE_FSGROUP} ${TEMPLATE_REPO} ${TEMPLATE_BRANCH} ${TEMPLATE_GPU} ${TEMPLATE_EXCLUDE_NODES}' \
  < "$HERE/test-pod.yaml" | kubectl apply -f -

# Bounded wait for the pod to leave Pending (unschedulable pods never fail).
deadline=$(( $(date +%s) + 900 ))
while :; do
  phase="$(kubectl get pod "$TEMPLATE_POD" -o jsonpath='{.status.phase}')"
  [[ "$phase" != "Pending" ]] && break
  if (( $(date +%s) >= deadline )); then
    kubectl describe pod "$TEMPLATE_POD" | tail -40; exit 1
  fi
  sleep 10
done
echo "pod phase: $phase on node $(kubectl get pod "$TEMPLATE_POD" -o jsonpath='{.spec.nodeName}')"

# Result marker (budget 60 min: pip install + DeepGEMM JIT dominate).
deadline=$(( $(date +%s) + 3600 ))
while :; do
  logs="$(kubectl logs "pod/$TEMPLATE_POD" -c test 2>/dev/null || true)"
  if grep -q "=== TEST PASS ===" <<<"$logs"; then
    grep -E "^(PASS|FAIL|SKIP)|torch |commit:" <<<"$logs"; echo "RESULT: PASS"; exit 0
  fi
  if grep -qE "=== TEST FAIL|ERROR:" <<<"$logs"; then
    kubectl logs "pod/$TEMPLATE_POD" -c test --tail=200; echo "RESULT: FAIL"; exit 1
  fi
  phase="$(kubectl get pod "$TEMPLATE_POD" -o jsonpath='{.status.phase}')"
  if [[ "$phase" == "Failed" ]]; then
    for c in clone fetch-uv test; do echo "--- $c ---"; kubectl logs "pod/$TEMPLATE_POD" -c "$c" --tail=60 || true; done
    echo "RESULT: FAIL (pod Failed)"; exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    kubectl logs "pod/$TEMPLATE_POD" -c test --tail=200; echo "RESULT: TIMEOUT"; exit 1
  fi
  sleep 15
done
