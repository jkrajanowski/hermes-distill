"""Combine several `hermes-distill export` output dirs into one train/holdout set, BOM-free.

Usage:
  python scripts/build_dataset.py OUT_DIR POOL_DIR [POOL_DIR ...]

Each POOL_DIR is an `export --out` directory. Their train.messages.jsonl files are concatenated into
OUT_DIR/train.messages.jsonl; the holdout.messages.jsonl files are concatenated into
OUT_DIR/holdout.messages.jsonl; tools.json is taken from the first pool. Samples whose final
assistant turn is a provider/billing error are dropped (they are not answers).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

BAD_ENDINGS = ("usage is exhausted", "Plans & Billing", "credit balance", "anthropic reported",
               "switch to an Anthropic API key", "overloaded_error")


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]


def _is_error_ending(rec: dict) -> bool:
    last = next((m for m in reversed(rec["messages"]) if m["role"] == "assistant"), None)
    text = ((last or {}).get("content") or "").lower()
    return any(b.lower() in text for b in BAD_ENDINGS)


def main() -> None:
    out = Path(sys.argv[1])
    pools = [Path(p) for p in sys.argv[2:]]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "holdout"):
        rows = [r for p in pools for r in _rows(p / f"{name}.messages.jsonl")]
        kept = [r for r in rows if not _is_error_ending(r)]
        (out / f"{name}.messages.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
        print(f"{name}: {len(kept)} samples ({len(rows) - len(kept)} error endings dropped)")
    shutil.copy(pools[0] / "tools.json", out / "tools.json")


if __name__ == "__main__":
    main()
