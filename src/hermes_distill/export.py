"""Orchestrate: state.db -> filtered conversations -> JSONL on disk."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .convert import (
    format_tools_block,
    infer_tool_schemas,
    schemas_from_session,
    schemas_to_openai_tools,
    to_openai_messages,
    to_sharegpt,
)
from .filters import FilterConfig, apply_filters
from .hermes_tools import dump_hermes_tool_schemas, merge_schemas
from .state_db import Session, StateDB


@dataclass
class Sample:
    session: Session
    messages: list[dict[str, Any]]
    tool_names: list[str]
    tool_calls: int
    assistant_turns: int
    redactions: int


@dataclass
class ExportStats:
    sessions_seen: int = 0
    kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    by_source: Counter = field(default_factory=Counter)
    by_model: Counter = field(default_factory=Counter)
    tool_usage: Counter = field(default_factory=Counter)
    redactions: int = 0
    assistant_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_tokens: int = 0
    estimated_cost_usd: float = 0.0
    sessions_with_schemas: int = 0
    schema_sources: Counter = field(default_factory=Counter)
    hermes_tools_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessions_seen": self.sessions_seen,
            "kept": self.kept,
            "dropped": dict(self.dropped),
            "by_source": dict(self.by_source),
            "by_model": dict(self.by_model),
            "tool_usage": dict(self.tool_usage.most_common()),
            "redactions": self.redactions,
            "assistant_turns": self.assistant_turns,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "cache_tokens": self.cache_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "sessions_with_schemas": self.sessions_with_schemas,
            "schema_sources": dict(self.schema_sources),
            "hermes_tools_error": self.hermes_tools_error,
        }


def collect(
    db: StateDB,
    cfg: FilterConfig,
    sources: Iterable[str] | None = None,
    exclude_sources: Iterable[str] | None = None,
    since: float | None = None,
    include_system: bool = True,
    tools_from_hermes: bool = True,
    tools_scope: str = "used",
) -> tuple[list[Sample], ExportStats, dict[str, dict[str, Any]]]:
    """Read every qualifying session and return samples, stats, and the merged tool schema registry.

    Schema priority: the session's own tool pin (what the teacher saw), then the live Hermes
    install's ``get_tool_definitions()``, then inference from observed calls.
    ``tools_scope`` decides which tools each sample lists: ``used`` (only tools it called) or
    ``all`` (every tool the Hermes install exposes, closer to what the agent sees at inference).
    """
    stats = ExportStats()
    samples: list[Sample] = []
    pinned: dict[str, dict[str, Any]] = {}
    for session in db.sessions(sources=sources, exclude_sources=exclude_sources, since=since):
        stats.sessions_seen += 1
        real = schemas_from_session(session)
        if real:
            stats.sessions_with_schemas += 1
            for name, schema in real.items():
                pinned.setdefault(name, schema)
        rows = db.messages(session.id)
        messages = to_openai_messages(session, rows, include_system=include_system)
        result = apply_filters(messages, cfg)
        if result.messages is None:
            stats.dropped[result.reason or "unknown"] += 1
            continue
        used = sorted({tc["function"]["name"] for m in result.messages for tc in (m.get("tool_calls") or [])})
        for name in used:
            stats.tool_usage[name] += sum(
                1 for m in result.messages for tc in (m.get("tool_calls") or []) if tc["function"]["name"] == name
            )
        tool_names = session.tool_names or used
        samples.append(
            Sample(session, result.messages, tool_names, result.tool_calls, result.assistant_turns, result.redactions)
        )
        stats.kept += 1
        stats.by_source[session.source] += 1
        stats.by_model[session.model or "unknown"] += 1
        stats.redactions += result.redactions
        stats.assistant_turns += result.assistant_turns
        stats.tool_calls += result.tool_calls
        stats.input_tokens += session.input_tokens
        stats.cache_tokens += session.cache_read_tokens + session.cache_write_tokens
        stats.output_tokens += session.output_tokens
        stats.estimated_cost_usd += session.estimated_cost_usd
    hermes_live: dict[str, dict[str, Any]] = {}
    if tools_from_hermes:
        hermes_live, err = dump_hermes_tool_schemas()
        stats.hermes_tools_error = err
    inferred = infer_tool_schemas([s.messages for s in samples])
    schemas = merge_schemas(pinned, hermes_live, inferred)
    for name in schemas:
        stats.schema_sources["pin" if name in pinned else "hermes" if name in hermes_live else "inferred"] += 1
    if tools_scope == "all" and hermes_live:
        for s in samples:
            if not s.session.tool_schemas:
                s.tool_names = sorted(set(s.tool_names) | set(hermes_live))
    return samples, stats, schemas


def _split_key(session_id: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{session_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _messages_record(sample: Sample, schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tools = schemas_to_openai_tools({n: schemas[n] for n in sample.tool_names if n in schemas})
    messages = []
    for m in sample.messages:
        nm = {k: v for k, v in m.items() if k != "reasoning"}
        if m.get("reasoning") and not nm.get("reasoning_content"):
            nm["reasoning_content"] = m["reasoning"]  # Qwen3 / DeepSeek chat-template convention
        messages.append(nm)
    return {
        "messages": messages,
        "tools": tools,
        "meta": {
            "session_id": sample.session.id,
            "source": sample.session.source,
            "model": sample.session.model,
            "started_at": sample.session.started_at,
            "tool_calls": sample.tool_calls,
            "assistant_turns": sample.assistant_turns,
        },
    }


def _sharegpt_record(sample: Sample, schemas: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tools_block = format_tools_block(schemas, sample.tool_names)
    return {
        "conversations": to_sharegpt(sample.messages, tools_block),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(sample.session.started_at)),
        "model": sample.session.model or "unknown",
        "completed": True,
        "meta": {"session_id": sample.session.id, "source": sample.session.source},
    }


def write_jsonl(
    samples: list[Sample],
    schemas: dict[str, dict[str, Any]],
    out_dir: Path,
    fmt: str = "messages",
    holdout: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    render = _messages_record if fmt == "messages" else _sharegpt_record
    train_path = out_dir / f"train.{fmt}.jsonl"
    holdout_path = out_dir / f"holdout.{fmt}.jsonl"
    n_train = n_hold = 0
    with train_path.open("w", encoding="utf-8") as ftrain, holdout_path.open("w", encoding="utf-8") as fhold:
        for s in samples:
            line = json.dumps(render(s, schemas), ensure_ascii=False) + "\n"
            if holdout > 0 and _split_key(s.session.id, seed) < holdout:
                fhold.write(line)
                n_hold += 1
            else:
                ftrain.write(line)
                n_train += 1
    (out_dir / "tools.json").write_text(json.dumps(schemas, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"train": str(train_path), "holdout": str(holdout_path), "n_train": n_train, "n_holdout": n_hold,
            "tools": str(out_dir / "tools.json")}
