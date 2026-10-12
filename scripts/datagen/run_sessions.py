"""Run many prompts through Hermes one-shots with per-worker sandboxes, retries and resume."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sandbox_builder import build

HERE = Path(__file__).parent
HERMES = Path(
    os.environ.get("HERMES_BIN")
    or shutil.which("hermes")
    or (Path(os.environ.get("LOCALAPPDATA", "")) / "hermes" / "bin" / "hermes.exe")
)
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PROMPTS = HERE / (sys.argv[2] if len(sys.argv) > 2 else "prompts_big.jsonl")
RESULTS = HERE / (sys.argv[3] if len(sys.argv) > 3 else "results_big.jsonl")
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else 10_000
TIMEOUT = 480
RETRY_MARKERS = ("rate", "429", "overloaded", "529", "timeout", "connection")
# Only the toolsets the tasks need. Cuts the per-call prompt from ~24k to ~7.5k tokens, which is the
# main cost lever on routes without prompt caching. Override with HERMES_TOOLSETS=... or empty to disable.
TOOLSETS = os.environ.get("HERMES_TOOLSETS", "terminal,file,code_execution")


def run_one(idx: int, item: dict, sandbox: Path) -> dict:
    attempts = 0
    while True:
        attempts += 1
        t0 = time.time()
        try:
            cmd = [str(HERMES)] + (["-t", TOOLSETS] if TOOLSETS else []) + ["-z", item["prompt"]]
            proc = subprocess.run(cmd, cwd=str(sandbox), capture_output=True, text=True, timeout=TIMEOUT,
                                  encoding="utf-8", errors="replace")
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = -1, "", "timeout"
        ok = rc == 0 and len(out.strip()) > 0
        retry = (not ok) and attempts < 3 and any(m in (err + out).lower() for m in RETRY_MARKERS)
        if ok or not retry:
            return {"idx": idx, "cat": item["cat"], "prompt": item["prompt"], "rc": rc, "attempts": attempts,
                    "seconds": round(time.time() - t0, 1), "stdout_tail": out[-400:], "stderr_tail": err[-300:]}
        time.sleep(30 * attempts)


def main() -> None:
    sandboxes = []
    for i in range(WORKERS):
        sb = HERE / f"sandbox_{i}"
        try:
            build(sb)
        except PermissionError:  # a stale process still holds the old copy; use a fresh name
            sb = HERE / f"sandbox_{i}_{int(time.time())}"
            build(sb)
        sandboxes.append(sb)
    prompts = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()][:LIMIT]
    done: set[int] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["idx"])
    todo = [(i, p) for i, p in enumerate(prompts) if i not in done]
    print(f"{len(prompts)} prompts, {len(done)} done, {len(todo)} to run, {WORKERS} workers", flush=True)
    t_start = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, i, p, sandboxes[i % WORKERS]): i for i, p in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            ok += r["rc"] == 0
            with RESULTS.open("a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if n % 10 == 0 or r["rc"] != 0:
                rate = n / max(time.time() - t_start, 1) * 60
                print(f"[{n}/{len(todo)}] ok={ok} rc={r['rc']} att={r['attempts']} {r['seconds']}s {rate:.1f}/min "
                      f"{r['cat']}: {r['prompt'][:50]}", flush=True)
    print(f"done: {ok}/{len(todo)} succeeded in {(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

