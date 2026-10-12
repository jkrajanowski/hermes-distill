"""Render eval JSON files (from `hermes-distill eval --out`) as a markdown table.

Usage: python scripts/results_table.py runs/eval-teacher.json runs/eval-base.json runs/eval-distilled.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = [("action_acc", "action"), ("tool_acc", "tool"), ("args_exact", "args exact"), ("json_valid", "JSON valid"),
        ("avg_latency_s", "s/turn")]


def main(paths: list[str]) -> None:
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    print("| model | turns | " + " | ".join(c[1] for c in COLS) + " |")
    print("|---|---:|" + "|".join("---:" for _ in COLS) + "|")
    for r in rows:
        cells = []
        for key, _ in COLS:
            v = r.get(key)
            cells.append("n/a" if v is None else (f"{v:.1f}%" if key != "avg_latency_s" else f"{v:.1f}"))
        print(f"| {r['model']} | {r['turns']} | " + " | ".join(cells) + " |")
    print()
    print("Per-tool accuracy (correct/gold):")
    tools = sorted({t for r in rows for t in r.get("per_tool", {})})
    print("| tool | " + " | ".join(r["model"] for r in rows) + " |")
    print("|---|" + "|".join("---:" for _ in rows) + "|")
    for t in tools:
        print(f"| {t} | " + " | ".join(
            f"{r['per_tool'][t]['correct']}/{r['per_tool'][t]['n']}" if t in r.get("per_tool", {}) else "-" for r in rows
        ) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
