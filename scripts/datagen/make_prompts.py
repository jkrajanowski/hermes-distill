"""Ask Claude for diverse, concrete task prompts grounded in the sandbox manifest. Writes JSONL.

Usage: ANTHROPIC_API_KEY=... python scripts/datagen/make_prompts.py 1000 prompts_big.jsonl
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from sandbox_builder import MANIFEST

HERE = Path(__file__).parent
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
OUT = HERE / (sys.argv[2] if len(sys.argv) > 2 else "prompts_big.jsonl")

FAMILIES = {
    "read": "read one or two files and answer a precise question about their content (summaries, extraction, comparisons)",
    "terminal": "answer a question that needs shell commands (counts, sizes, hashes, listing, environment info, running a script)",
    "search": "find things across files: definitions, references, patterns, counts per file, lines matching a term",
    "data": "compute something from the CSV/JSON data with python (aggregates, group-by, filters, joins, medians, percentages, top-N) and report numbers",
    "write": "create a new script or file in out/ that does a specific job, run it, and report the output",
    "patch": "make a small, precise edit to an existing file (fix a named bug, rename, add a docstring, change a config value) and verify",
    "multi": "a 3-6 step task combining investigation, editing, running tests or scripts, and reporting (e.g. fix the failing tests in data/minipkg_tests, build a tiny CLI over a CSV, validate data and write a report)",
    "chat": "a short conceptual question that needs NO tools (python, data, testing, agents, SQL, regex, fine-tuning); answerable in 1-4 sentences",
}
SHARE = {"read": 0.12, "terminal": 0.14, "search": 0.10, "data": 0.18, "write": 0.14, "patch": 0.12, "multi": 0.12, "chat": 0.08}


def ask(client: OpenAI, family: str, n: int, avoid: list[str]) -> list[str]:
    avoid_txt = "\n".join(f"- {a}" for a in avoid[-40:])
    prompt = (
        f"You write tasks for an AI coding agent that works in this sandbox:\n\n{MANIFEST}\n\n"
        f"Write {n} DISTINCT user requests of this family: {family}: {FAMILIES[family]}.\n"
        "Rules: each request is 1-2 sentences, concrete, references real files/columns/functions from the manifest, "
        "is completable in under 2 minutes, has a verifiable answer or artifact, and does not need internet. "
        "Vary phrasing, verbosity, and difficulty. Mix casual and formal tone. Never repeat a request from this list:\n"
        f"{avoid_txt}\n\nReturn ONLY a JSON array of strings."
    )
    r = client.chat.completions.create(model="claude-sonnet-5", messages=[{"role": "user", "content": prompt}],
                                       max_tokens=6000)
    text = r.choices[0].message.content or "[]"
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        items = json.loads(m.group(0) if m else text)
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and 15 < len(s) < 400]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())[:80]


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("HERMES_DISTILL_EVAL_KEY")
    if not api_key:
        raise SystemExit("set ANTHROPIC_API_KEY")
    client = OpenAI(base_url="https://api.anthropic.com/v1/", api_key=api_key)
    seen: set[str] = set()
    existing = [HERE / "prompts_seed.jsonl", HERE / "prompts_eval_seed.jsonl"]
    for p in existing:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    seen.add(norm(json.loads(line)["prompt"]))
    out: list[dict] = []
    for family, share in SHARE.items():
        want = round(TARGET * share)
        got: list[str] = []
        rounds = 0
        while len(got) < want and rounds < 8:
            rounds += 1
            batch = ask(client, family, min(40, want - len(got) + 5), got)
            for s in batch:
                k = norm(s)
                if k in seen:
                    continue
                seen.add(k)
                got.append(s)
                if len(got) >= want:
                    break
            print(f"{family}: {len(got)}/{want} after round {rounds}", flush=True)
        out.extend({"cat": family, "prompt": s} for s in got)
    import random

    random.Random(1).shuffle(out)
    OUT.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} prompts -> {OUT}")


if __name__ == "__main__":
    main()
