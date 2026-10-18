#!/usr/bin/env python3
"""Aggregate serve_bench.py result JSONs into summary.csv and summary.md.

Reads <results_dir>/<label>_c<concurrency>.json written by serve_bench.py, plus
optional <label>.failed marker files written by run_sweep.sh for configs whose
deployment failed.  Rows are configs, columns are concurrency levels.

Usage: python scripts/bench/summarize.py RESULTS_DIR
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys

# label -> description; unknown labels are shown as-is.
CONFIG_NAMES = {
    "1n4g": "1 node x 4 GPUs",
    "1n8g": "1 node x 8 GPUs",
    "2n4g": "2 nodes x 4 GPUs",
}
ORDER = ["1n4g", "1n8g", "2n4g"]

METRICS = [
    ("req_per_s", "req/s", "{:.2f}"),
    ("output_tok_per_s", "output tok/s", "{:.0f}"),
    ("total_tok_per_s", "total tok/s (input + output)", "{:.0f}"),
]


def load(results_dir: str):
    runs: dict[tuple[str, int], dict] = {}
    for path in glob.glob(os.path.join(results_dir, "*_c*.json")):
        with open(path) as f:
            data = json.load(f)
        runs[(data["label"], int(data["concurrency"]))] = data
    failed: dict[str, str] = {}
    for path in glob.glob(os.path.join(results_dir, "*.failed")):
        label = os.path.basename(path)[: -len(".failed")]
        with open(path) as f:
            failed[label] = f.read().strip() or "deploy failed"
    return runs, failed


def cell(run: dict | None, key: str, fmt: str) -> str:
    if run is None:
        return "-"
    if not run.get("completed"):
        return "FAILED"
    text = fmt.format(run[key])
    return text + "*" if run.get("failed") else text


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    results_dir = sys.argv[1]
    runs, failed = load(results_dir)
    if not runs and not failed:
        print(f"no results found in {results_dir}", file=sys.stderr)
        return 1

    labels = {label for label, _ in runs} | set(failed)
    labels = sorted(labels, key=lambda x: (ORDER.index(x) if x in ORDER else len(ORDER), x))
    concs = sorted({c for _, c in runs})

    # CSV (long form)
    csv_path = os.path.join(results_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["config", "label", "concurrency", "input_len", "output_len", "completed", "failed",
             "duration_s", "req_per_s", "output_tok_per_s", "total_tok_per_s",
             "ttft_p50_s", "ttft_p99_s", "e2e_p50_s", "e2e_p99_s"]
        )
        for label in labels:
            for conc in concs:
                r = runs.get((label, conc))
                if r is None:
                    continue
                ok = bool(r.get("completed"))
                w.writerow([
                    CONFIG_NAMES.get(label, label), label, conc, r["input_len"], r["output_len"],
                    r["completed"], r["failed"],
                    *(
                        [round(r["duration_s"], 3), round(r["req_per_s"], 4),
                         round(r["output_tok_per_s"], 2), round(r["total_tok_per_s"], 2),
                         round(r["ttft_s"]["p50"], 3), round(r["ttft_s"]["p99"], 3),
                         round(r["e2e_s"]["p50"], 3), round(r["e2e_s"]["p99"], 3)]
                        if ok else [""] * 8
                    ),
                ])

    # Markdown
    any_run = next(iter(runs.values()), None)
    lines = ["# FastAFD serving benchmark", ""]
    if any_run:
        lines += [f"Input {any_run['input_len']} tokens, output {any_run['output_len']} tokens, "
                  f"model `{any_run['model']}`, closed loop at the given concurrency.", ""]
    for key, title, fmt in METRICS:
        lines += [f"## {title}", "",
                  "| Config | " + " | ".join(f"c={c}" for c in concs) + " |",
                  "|---|" + "---:|" * len(concs)]
        for label in labels:
            name = CONFIG_NAMES.get(label, label)
            if label in failed and not any((label, c) in runs for c in concs):
                lines.append(f"| {name} | " + " | ".join(["DEPLOY FAILED"] * len(concs)) + " |")
                continue
            lines.append(f"| {name} | " + " | ".join(cell(runs.get((label, c)), key, fmt) for c in concs) + " |")
        lines.append("")
    lines += ["## Latency (p50 time to first token / p50 end-to-end, seconds)", "",
              "| Config | " + " | ".join(f"c={c}" for c in concs) + " |",
              "|---|" + "---:|" * len(concs)]
    for label in labels:
        row = []
        for c in concs:
            r = runs.get((label, c))
            row.append("-" if r is None else "FAILED" if not r.get("completed")
                       else f"{r['ttft_s']['p50']:.2f} / {r['e2e_s']['p50']:.1f}")
        lines.append(f"| {CONFIG_NAMES.get(label, label)} | " + " | ".join(row) + " |")
    lines.append("")
    notes = []
    if any(r.get("failed") and r.get("completed") for r in runs.values()):
        notes.append("`*` = some requests failed; metrics count successful requests only.")
    for label, reason in failed.items():
        notes.append(f"{CONFIG_NAMES.get(label, label)}: {reason}")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]

    md_path = os.path.join(results_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {csv_path} and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
