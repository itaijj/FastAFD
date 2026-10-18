#!/usr/bin/env python3
"""Closed-loop streaming load generator for a FastAFD / miniSGL OpenAI endpoint.

Keeps `--concurrency` requests in flight against POST /v1/chat/completions and
reports req/s, output tok/s and total (input + output) tok/s.

Requests are sent with `prompt` (not `messages`) so the server does not apply a
chat template and the input length is exactly `--input-len` tokens.  They use
`ignore_eos=true`, so every request decodes exactly `--output-len` tokens.

The only non-stdlib dependency is `transformers`, used to build and verify the
prompt length.

Example:
    python scripts/bench/serve_bench.py --host 127.0.0.1 --port 8000 \
        --model Qwen/Qwen3-30B-A3B-FP8 --concurrency 128 --label 8gpu \
        --output results/8gpu_c128.json
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from dataclasses import dataclass, field


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="Qwen/Qwen3-30B-A3B-FP8", help="model id sent in the request")
    p.add_argument("--tokenizer", default=None, help="HF tokenizer id/path (default: --model)")
    p.add_argument("--concurrency", type=int, required=True, help="requests kept in flight")
    p.add_argument("--input-len", type=int, default=256, help="prompt tokens")
    p.add_argument("--output-len", type=int, default=1024, help="decode tokens (max_tokens)")
    p.add_argument(
        "--num-requests",
        type=int,
        default=0,
        help="measured requests (default: 3 * concurrency)",
    )
    p.add_argument("--warmup-requests", type=int, default=0, help="unmeasured requests before the run")
    p.add_argument("--warmup-output-len", type=int, default=32, help="decode tokens for warmup requests")
    p.add_argument("--timeout", type=float, default=1800.0, help="per-request socket timeout (s)")
    p.add_argument("--label", default="", help="free-form label stored in the result (e.g. 4gpu)")
    p.add_argument("--output", default=None, help="write result JSON here")
    return p.parse_args()


# --------------------------------------------------------------------------- prompt


def build_prompt(tokenizer_name: str, input_len: int) -> str:
    """Return text that the server's tokenizer encodes to exactly `input_len` ids."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    words = (
        "The quick brown fox jumps over the lazy dog while distributed inference systems "
        "schedule attention and expert layers across many accelerators. "
    )
    ids = tok.encode(words * (input_len // 8 + 8), add_special_tokens=False)
    n = input_len
    for _ in range(64):
        text = tok.decode(ids[:n])
        # Match the server, which calls tokenizer.encode(prompt) with defaults.
        got = len(tok.encode(text))
        if got == input_len:
            return text
        n += input_len - got
    raise RuntimeError(f"could not build a prompt of exactly {input_len} tokens (last got {got})")


# --------------------------------------------------------------------------- client


@dataclass
class Result:
    ok: bool
    start: float = 0.0
    end: float = 0.0
    ttft: float = 0.0
    out_tokens: int = 0
    error: str = ""


def one_request(args: argparse.Namespace, prompt: str, output_len: int) -> Result:
    body = json.dumps(
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": output_len,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    )
    res = Result(ok=False)
    res.start = time.perf_counter()
    conn = http.client.HTTPConnection(args.host, args.port, timeout=args.timeout)
    try:
        conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            res.error = f"HTTP {resp.status}: {resp.read(200)!r}"
            return res
        done = False
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                done = True
                break
            if not payload:
                continue
            choice = json.loads(payload)["choices"][0]
            token_ids = choice.get("token_ids")
            if token_ids:
                if res.out_tokens == 0:
                    res.ttft = time.perf_counter() - res.start
                res.out_tokens += len(token_ids)
        if not done:
            res.error = "stream ended without [DONE]"
        elif res.out_tokens != output_len:
            res.error = f"got {res.out_tokens} output tokens, expected {output_len}"
        else:
            res.ok = True
    except Exception as exc:  # noqa: BLE001 - report any client-side failure
        res.error = f"{type(exc).__name__}: {exc}"
    finally:
        res.end = time.perf_counter()
        conn.close()
    return res


@dataclass
class Pool:
    args: argparse.Namespace
    prompt: str
    total: int
    output_len: int
    concurrency: int
    results: list[Result] = field(default_factory=list)
    _next: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._next >= self.total:
                    return
                self._next += 1
            r = one_request(self.args, self.prompt, self.output_len)
            with self._lock:
                self.results.append(r)

    def run(self) -> list[Result]:
        threads = [
            threading.Thread(target=self._worker, daemon=True)
            for _ in range(min(self.concurrency, self.total))
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.results


# --------------------------------------------------------------------------- metrics


def pct(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def summarize(args: argparse.Namespace, results: list[Result]) -> dict:
    good = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    out: dict = {
        "label": args.label,
        "model": args.model,
        "concurrency": args.concurrency,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_requests": len(results),
        "completed": len(good),
        "failed": len(failed),
        "errors": sorted({r.error for r in failed})[:5],
    }
    if good:
        # Window spans first-started to last-finished successful request.
        duration = max(r.end for r in good) - min(r.start for r in good)
        out_tokens = sum(r.out_tokens for r in good)
        in_tokens = len(good) * args.input_len
        ttft = sorted(r.ttft for r in good)
        e2e = sorted(r.end - r.start for r in good)
        out.update(
            duration_s=duration,
            req_per_s=len(good) / duration,
            output_tok_per_s=out_tokens / duration,
            total_tok_per_s=(out_tokens + in_tokens) / duration,
            ttft_s={"p50": pct(ttft, 50), "p95": pct(ttft, 95), "p99": pct(ttft, 99)},
            e2e_s={"p50": pct(e2e, 50), "p95": pct(e2e, 95), "p99": pct(e2e, 99)},
        )
    return out


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        print("--concurrency must be >= 1", file=sys.stderr)
        return 2
    num_requests = args.num_requests or 3 * args.concurrency

    print(f"building {args.input_len}-token prompt ...", flush=True)
    prompt = build_prompt(args.tokenizer or args.model, args.input_len)

    if args.warmup_requests > 0:
        print(f"warmup: {args.warmup_requests} requests x {args.warmup_output_len} tokens", flush=True)
        warm = Pool(args, prompt, args.warmup_requests, args.warmup_output_len, args.concurrency).run()
        bad = [r for r in warm if not r.ok]
        if bad:
            print(f"warmup: {len(bad)} failed, first error: {bad[0].error}", file=sys.stderr)

    print(
        f"run: label={args.label or '-'} concurrency={args.concurrency} requests={num_requests} "
        f"in={args.input_len} out={args.output_len}",
        flush=True,
    )
    results = Pool(args, prompt, num_requests, args.output_len, args.concurrency).run()
    summary = summarize(args, results)

    if summary["completed"]:
        print(
            f"RESULT label={args.label or '-'} conc={args.concurrency} "
            f"ok={summary['completed']}/{summary['num_requests']} "
            f"req/s={summary['req_per_s']:.3f} out_tok/s={summary['output_tok_per_s']:.1f} "
            f"total_tok/s={summary['total_tok_per_s']:.1f} "
            f"ttft_p50={summary['ttft_s']['p50']:.2f}s e2e_p50={summary['e2e_s']['p50']:.1f}s"
        )
    else:
        print(f"RESULT label={args.label or '-'} conc={args.concurrency} ALL REQUESTS FAILED: {summary['errors']}")
    if summary["failed"]:
        print(f"WARNING: {summary['failed']} failed; errors: {summary['errors']}", file=sys.stderr)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.output}")
    return 0 if summary["completed"] and not summary["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
