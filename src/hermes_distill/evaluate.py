"""Next-action evaluation against any OpenAI-compatible endpoint.

For every assistant turn in the held-out conversations we feed the model the
conversation prefix (with the real tool schemas) and ask for the next action.
We then score:

* ``action_acc``   did it choose the right kind of step (tool call vs. answer)?
* ``tool_acc``     on tool turns, did it call the same tool as the teacher?
* ``args_exact``   on tool turns, did the arguments match exactly?
* ``json_valid``   were the emitted arguments valid JSON?

This is cheap, offline, and directly comparable between the teacher model,
the base model, and the distilled model, which is the whole point.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .train import load_records


@dataclass
class EvalConfig:
    data: Path
    model: str
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "no-key"
    max_samples: int | None = None
    max_turns_per_sample: int = 6
    max_tokens: int = 512
    temperature: float = 0.0
    timeout: float = 120.0
    strip_reasoning: bool = True


@dataclass
class EvalResult:
    turns: int = 0
    action_correct: int = 0
    tool_turns: int = 0
    tool_correct: int = 0
    args_exact: int = 0
    json_valid: int = 0
    errors: int = 0
    latency_s: float = 0.0
    per_tool: dict[str, dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        def pct(a: int, b: int) -> float | None:
            return round(100 * a / b, 1) if b else None

        return {
            "turns": self.turns,
            "action_acc": pct(self.action_correct, self.turns),
            "tool_turns": self.tool_turns,
            "tool_acc": pct(self.tool_correct, self.tool_turns),
            "args_exact": pct(self.args_exact, self.tool_turns),
            "json_valid": pct(self.json_valid, self.tool_turns),
            "errors": self.errors,
            "avg_latency_s": round(self.latency_s / self.turns, 2) if self.turns else None,
            "per_tool": self.per_tool,
        }


def _prefix(messages: list[dict[str, Any]], k: int, strip_reasoning: bool) -> list[dict[str, Any]]:
    out = []
    for m in messages[:k]:
        nm = {k2: v for k2, v in m.items() if k2 not in ("reasoning", "reasoning_content")} if strip_reasoning else dict(m)
        if nm["role"] == "assistant" and nm.get("tool_calls"):
            nm["tool_calls"] = [
                {"id": tc.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                for i, tc in enumerate(nm["tool_calls"])
            ]
            if not nm.get("content"):
                nm["content"] = ""
        if nm["role"] == "tool":
            nm = {"role": "tool", "content": nm.get("content") or "", "tool_call_id": nm.get("tool_call_id") or ""}
        out.append(nm)
    return out


def _canon(args: Any) -> str:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args.strip()
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def run_eval(cfg: EvalConfig, progress: Any | None = None) -> EvalResult:
    from openai import OpenAI

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=cfg.timeout)
    records = load_records(cfg.data, cfg.max_samples)
    res = EvalResult()
    for rec in records:
        messages = rec["messages"]
        tools = rec.get("tools") or None
        done = 0
        for k, m in enumerate(messages):
            if m["role"] != "assistant" or done >= cfg.max_turns_per_sample:
                continue
            done += 1
            gold_calls = m.get("tool_calls") or []
            gold_is_tool = bool(gold_calls)
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=cfg.model,
                    messages=_prefix(messages, k, cfg.strip_reasoning),
                    tools=tools,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
            except Exception:
                res.errors += 1
                res.turns += 1
                continue
            res.latency_s += time.time() - t0
            res.turns += 1
            pred = resp.choices[0].message
            pred_calls = list(pred.tool_calls or [])
            pred_is_tool = bool(pred_calls)
            if pred_is_tool == gold_is_tool:
                res.action_correct += 1
            if gold_is_tool:
                res.tool_turns += 1
                gold_name = gold_calls[0]["function"]["name"]
                slot = res.per_tool.setdefault(gold_name, {"n": 0, "correct": 0})
                slot["n"] += 1
                if pred_is_tool:
                    pname = pred_calls[0].function.name
                    pargs = pred_calls[0].function.arguments
                    try:
                        json.loads(pargs) if isinstance(pargs, str) else None
                        res.json_valid += 1
                    except json.JSONDecodeError:
                        pass
                    if pname == gold_name:
                        res.tool_correct += 1
                        slot["correct"] += 1
                        if _canon(pargs) == _canon(gold_calls[0]["function"]["arguments"]):
                            res.args_exact += 1
            if progress:
                progress(res)
    return res
