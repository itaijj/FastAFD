#!/usr/bin/env bash
# Per H100 node: allocatable GPUs and GPUs already requested by running pods.
for n in $(kubectl get nodes -l nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3 -o name); do
  n=${n#node/}
  alloc=$(kubectl get node "$n" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')
  used=$(kubectl describe node "$n" | awk '/Allocated resources/,0' | awk '$1=="nvidia.com/gpu"{print $2; exit}')
  echo "$n alloc=$alloc used=${used:-0}"
done
kubectl get resourcequota -n itai-ds2 2>/dev/null
