"""Quality filters applied to each conversation before it becomes training data.

Everything here is deterministic and cheap. The goal is a dataset that is
(a) safe to put on a GPU box or share, and (b) free of the junk that makes
small models worse: dangling tool calls, megabyte tool outputs, duplicate
sessions, and conversations that never reached an answer.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("google", re.compile(r"AIza[0-9A-Za-z_\-]{30,}")),
    ("github", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("slack", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("aws", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("telegram", re.compile(r"\b\d{8,10}:[A-Za-z0-9_\-]{35}\b")),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}")),
    ("pem", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----")),
    ("kv", re.compile(r"(?i)\b([A-Z0-9_]*(?:api[_-]?key|secret|token|password|passwd)[A-Z0-9_]*)\s*[=:]\s*['\"]?([^\s'\"]{8,})")),
]


def redact(text: str) -> tuple[str, int]:
    """Replace credential-looking strings with ``[REDACTED:<kind>]``. Returns (text, hits)."""
    if not text:
        return text, 0
    hits = 0
    for kind, pattern in _SECRET_PATTERNS:
        if kind == "kv":
            def _sub(m: re.Match[str], _kind: str = kind) -> str:
                nonlocal hits
                hits += 1
                return f"{m.group(1)}=[REDACTED:{_kind}]"
            text, n = pattern.subn(_sub, text)
        else:
            text, n = pattern.subn(f"[REDACTED:{kind}]", text)
            hits += n
    return text, hits


def _redact_value(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        total = 0
        out = {}
        for k, v in value.items():
            nv, n = _redact_value(v)
            out[k] = nv
            total += n
        return out, total
    if isinstance(value, list):
        total = 0
        out_l = []
        for v in value:
            nv, n = _redact_value(v)
            out_l.append(nv)
            total += n
        return out_l, total
    return value, 0


def redact_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    total = 0
    out = []
    for m in messages:
        nm = dict(m)
        for key in ("content", "reasoning", "reasoning_content"):
            if nm.get(key):
                nm[key], n = redact(nm[key])
                total += n
        if nm.get("tool_calls"):
            calls = []
            for tc in nm["tool_calls"]:
                args, n = _redact_value(tc["function"]["arguments"])
                total += n
                calls.append({**tc, "function": {**tc["function"], "arguments": args}})
            nm["tool_calls"] = calls
        out.append(nm)
    return out, total


# ---------------------------------------------------------------------------
# Structural cleanup
# ---------------------------------------------------------------------------

def truncate_tool_responses(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
    if max_chars <= 0:
        return messages
    out = []
    for m in messages:
        if m["role"] == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > max_chars:
            head = max_chars * 2 // 3
            tail = max_chars - head
            dropped = len(m["content"]) - max_chars
            m = {**m, "content": m["content"][:head] + f"\n...[{dropped} chars truncated]...\n" + m["content"][-tail:]}
        out.append(m)
    return out


def trim_dangling(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing turns until the conversation ends with an assistant *text* reply.

    A session that was interrupted mid tool-call, or whose last turn is the user
    typing and getting nothing back, teaches the model to stop in the wrong place.
    """
    msgs = list(messages)
    while msgs:
        last = msgs[-1]
        if last["role"] == "assistant" and not last.get("tool_calls") and (last.get("content") or "").strip():
            break
        msgs.pop()
    return msgs


def repair_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure every assistant tool call has exactly its results right after it and
    that no tool message floats without a preceding call."""
    out: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"]]
            results: list[dict[str, Any]] = []
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                results.append(messages[j])
                j += 1
            if not results:
                # Dangling call in the middle: drop the call turn but keep the text.
                if (m.get("content") or "").strip():
                    out.append({k: v for k, v in m.items() if k != "tool_calls"})
                i = j
                continue
            # Align by tool_call_id when available, else by position.
            by_id = {r.get("tool_call_id"): r for r in results if r.get("tool_call_id")}
            aligned: list[dict[str, Any]] = []
            for pos, call_id in enumerate(ids):
                r = by_id.get(call_id) if call_id else None
                if r is None and pos < len(results):
                    r = results[pos]
                if r is None:
                    r = {"role": "tool", "content": "", "tool_call_id": call_id or "", "name": ""}
                aligned.append({**r, "tool_call_id": call_id or r.get("tool_call_id", "")})
            out.append(m)
            out.extend(aligned)
            i = j
            continue
        if m["role"] == "tool":
            i += 1  # orphan result
            continue
        out.append(m)
        i += 1
    return out


# Matched against json.dumps(arguments), so a Windows backslash shows up as "\\\\" (two chars).
_ABS_PATH = re.compile(r"(?<![\w.])(?:[A-Za-z]:(?:\\\\|/)|/(?:c|d|home|Users|root|etc|tmp|var|mnt|opt)(?=[/\\\\\s\"']|$)|~(?![\w]))")
_PATH_END = ' "\'\n\t;|&<>)(,'


def _norm_path(p: str) -> str:
    p = re.sub(r"[\\/]+", "/", p).rstrip("/").lower()  # JSON-escaped backslashes collapse too
    m = re.match(r"^/([a-z])/(.*)$", p)  # git-bash /c/Users/... -> c:/users/...
    return f"{m.group(1)}:/{m.group(2)}" if m else p


def _temp_roots() -> list[str]:
    import tempfile

    from .state_db import default_hermes_home

    roots = ["/tmp", _norm_path(tempfile.gettempdir()), _norm_path(str(default_hermes_home() / "cache" / "scratch"))]
    return [r for r in roots if r]


def touches_outside_cwd(messages: list[dict[str, Any]], cwd: str | None) -> bool:
    """True if any tool call references a filesystem location outside ``cwd``.

    Absolute paths under ``cwd`` or under a temp directory are fine; any other absolute path or
    ``~`` counts as outside. Sessions like that leak the user's directory layout into the dataset
    and teach the model to wander, so the exporter drops them by default. Relative ``..`` is not
    checked: it is too common inside legitimate subdirectory work to be a useful signal.
    """
    allowed = ([_norm_path(cwd)] if cwd else []) + _temp_roots()
    for m in messages:
        for tc in m.get("tool_calls") or []:
            blob = json.dumps(tc["function"]["arguments"], ensure_ascii=False)
            for hit in _ABS_PATH.finditer(blob):
                if hit.group(0).startswith("~"):
                    return True
                end = hit.end()
                while end < len(blob) and blob[end] not in _PATH_END:
                    end += 1
                path = _norm_path(blob[hit.start():end])
                if not any(path == a or path.startswith(a + "/") for a in allowed):
                    return True
    return False


def conversation_hash(messages: list[dict[str, Any]]) -> str:
    h = hashlib.sha1()
    for m in messages:
        if m["role"] in ("user", "assistant"):
            h.update(m["role"].encode())
            h.update((m.get("content") or "").strip().encode("utf-8", "replace"))
            for tc in m.get("tool_calls") or []:
                h.update(json.dumps(tc["function"], sort_keys=True, ensure_ascii=False).encode("utf-8", "replace"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

@dataclass
class FilterConfig:
    min_assistant_turns: int = 1
    min_tool_calls: int = 0
    require_reasoning: bool = False
    require_completed: bool = True
    max_tool_response_chars: int = 6000
    max_messages: int = 400
    max_total_chars: int = 400_000
    dedupe: bool = True
    redact_secrets: bool = True
    keep_system: bool = True
    drop_outside_cwd: bool = True
    seen_hashes: set[str] = field(default_factory=set)


@dataclass
class FilterResult:
    messages: list[dict[str, Any]] | None
    reason: str | None = None
    redactions: int = 0
    tool_calls: int = 0
    assistant_turns: int = 0


def apply_filters(messages: list[dict[str, Any]], cfg: FilterConfig, cwd: str | None = None) -> FilterResult:
    msgs = [m for m in messages if cfg.keep_system or m["role"] != "system"]
    if cfg.drop_outside_cwd and touches_outside_cwd(msgs, cwd):
        return FilterResult(None, "outside_cwd")
    msgs = repair_tool_pairs(msgs)
    if cfg.require_completed:
        msgs = trim_dangling(msgs)
    msgs = truncate_tool_responses(msgs, cfg.max_tool_response_chars)
    if cfg.max_messages and len(msgs) > cfg.max_messages:
        msgs = trim_dangling(msgs[: cfg.max_messages])

    assistant_turns = sum(1 for m in msgs if m["role"] == "assistant")
    tool_calls = sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant")
    if not any(m["role"] == "user" for m in msgs):
        return FilterResult(None, "no_user_turn")
    if assistant_turns < cfg.min_assistant_turns:
        return FilterResult(None, "too_few_assistant_turns")
    if tool_calls < cfg.min_tool_calls:
        return FilterResult(None, "too_few_tool_calls")
    if cfg.require_reasoning and not any(
        (m.get("reasoning") or m.get("reasoning_content") or "").strip() for m in msgs if m["role"] == "assistant"
    ):
        return FilterResult(None, "no_reasoning")
    total_chars = sum(len(m.get("content") or "") for m in msgs)
    if cfg.max_total_chars and total_chars > cfg.max_total_chars:
        return FilterResult(None, "too_long")

    redactions = 0
    if cfg.redact_secrets:
        msgs, redactions = redact_messages(msgs)

    if cfg.dedupe:
        h = conversation_hash(msgs)
        if h in cfg.seen_hashes:
            return FilterResult(None, "duplicate")
        cfg.seen_hashes.add(h)

    return FilterResult(msgs, None, redactions, tool_calls, assistant_turns)
