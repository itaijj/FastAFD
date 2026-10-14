#!/usr/bin/env bash
# Smoke-test the live fastafd-qwen30b OpenAI-compatible endpoint.
#
# Usage (from outside the cluster via NodePort):
#   SERVE_HOST=<node-ip> SERVE_PORT=<nodePort> bash docker/h100/k8s/smoke_test_serve.sh
#
# Usage (from inside the cluster via ClusterIP / Service DNS):
#   SERVE_HOST=fastafd-qwen30b-service SERVE_PORT=8000 bash docker/h100/k8s/smoke_test_serve.sh
#
# Exits 0 and prints "=== SMOKE TEST PASS ===" on success.
# Exits 1 and prints "=== SMOKE TEST FAIL ===" on any failure.
set -uo pipefail

SERVE_HOST="${SERVE_HOST:-127.0.0.1}"
SERVE_PORT="${SERVE_PORT:-8000}"
BASE_URL="http://${SERVE_HOST}:${SERVE_PORT}"

# Resolve Python interpreter (python3 preferred, fall back to python)
PYTHON="$(command -v python3 2>/dev/null || command -v python 2>/dev/null)"
if [[ -z "$PYTHON" ]]; then
  echo "=== SMOKE TEST FAIL === (no python/python3 found on PATH)"
  exit 1
fi

echo "Smoke-testing FastAFD OpenAI endpoint: $BASE_URL (using $PYTHON)"

# ---------------------------------------------------------------------------
# Step 1: /v1/models must return a non-empty model list
# ---------------------------------------------------------------------------
"$PYTHON" - "$BASE_URL/v1/models" <<'PY' || { echo "=== SMOKE TEST FAIL ==="; exit 1; }
import json, sys, urllib.request

url = sys.argv[1]
try:
    resp = urllib.request.urlopen(url, timeout=10)
    data = json.loads(resp.read())
except Exception as exc:
    print(f"FAIL /v1/models: {exc}", file=sys.stderr)
    sys.exit(1)

models = data.get("data", [])
if not models:
    print("FAIL /v1/models: empty model list", file=sys.stderr)
    sys.exit(1)

print(f"  /v1/models OK: {[m['id'] for m in models]}")
PY

# ---------------------------------------------------------------------------
# Step 2: /v1/chat/completions must return a non-empty assistant message
# ---------------------------------------------------------------------------
"$PYTHON" - "$BASE_URL/v1/chat/completions" <<'PY' || { echo "=== SMOKE TEST FAIL ==="; exit 1; }
import json, sys, urllib.request

url = sys.argv[1]
payload = json.dumps({
    "model": "default",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10,
    "temperature": 0,
    "stream": False,
}).encode()

req = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    resp = urllib.request.urlopen(req, timeout=60)
    raw = resp.read()
except Exception as exc:
    print(f"FAIL /v1/chat/completions request: {exc}", file=sys.stderr)
    sys.exit(1)

# The server streams SSE chunks; collect and parse the last data: line
# that contains a choices array with finish_reason.
content_parts = []
for line in raw.decode(errors="replace").splitlines():
    if not line.startswith("data:"):
        continue
    chunk_str = line[5:].strip()
    if chunk_str in ("", "[DONE]"):
        continue
    try:
        chunk = json.loads(chunk_str)
    except json.JSONDecodeError:
        continue
    delta = chunk.get("choices", [{}])[0].get("delta", {})
    if delta.get("content"):
        content_parts.append(delta["content"])

content = "".join(content_parts).strip()
if not content:
    print(f"FAIL /v1/chat/completions: empty assistant content; raw response:\n{raw[:1000]}", file=sys.stderr)
    sys.exit(1)

print(f"  /v1/chat/completions OK: assistant said: {content!r}")
PY

echo "=== SMOKE TEST PASS ==="
