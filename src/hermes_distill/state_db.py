"""Read-only access to Hermes Agent's ``state.db``.

Hermes keeps every session and message in a single SQLite database under
``$HERMES_HOME`` (default ``~/.hermes/state.db``) in WAL mode. We never open the
live file for writing: the database is snapshotted with SQLite's online backup
API into a temp file first, so a running gateway is never disturbed and we get
a consistent view that includes the WAL.

Schema facts this module relies on (hermes_state_common.py, schema v25+):

* ``sessions``: id, source, model, system_prompt / system_prompt_hash,
  started_at, ended_at, end_reason, message_count, tool_call_count, token
  counters, estimated_cost_usd, title, tool_names, parent_session_id.
* ``messages``: session_id, role, content, tool_call_id, tool_calls (JSON),
  tool_name, reasoning, reasoning_content, timestamp, finish_reason, active.
* ``system_prompts``: hash -> prompt (content-addressed since v25).
* Structured (multimodal) ``content`` is stored as ``"\\x00json:" + json``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CONTENT_JSON_PREFIX = "\x00json:"


def default_hermes_home() -> Path:
    """Mirror ``hermes_constants.get_hermes_home``: ``$HERMES_HOME`` wins, else the platform default
    (``%LOCALAPPDATA%\\hermes`` on Windows, ``~/.hermes`` elsewhere, plus ``HERMES_DATA_DIR_SUFFIX``)."""
    override = os.environ.get("HERMES_HOME", "").strip()
    if override:
        return Path(os.path.expanduser(os.path.expandvars(override)))
    suffix = os.environ.get("HERMES_DATA_DIR_SUFFIX", "")
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / ("hermes" + suffix)
    return Path.home() / (".hermes" + suffix)


def default_state_db() -> Path:
    return default_hermes_home() / "state.db"


@dataclass
class Session:
    id: str
    source: str
    model: str | None
    started_at: float
    ended_at: float | None
    end_reason: str | None
    message_count: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: float
    title: str | None
    system_prompt: str | None
    tool_names: list[str] = field(default_factory=list)
    tool_schemas: list[dict[str, Any]] | None = None
    parent_session_id: str | None = None
    cwd: str | None = None


@dataclass
class Message:
    id: int
    role: str
    content: Any
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]] | None
    tool_name: str | None
    reasoning: str | None
    reasoning_content: str | None
    timestamp: float
    finish_reason: str | None


def decode_content(raw: Any) -> Any:
    """Reverse the Hermes content encoding. Multimodal parts come back as a list of dicts."""
    if isinstance(raw, str) and raw.startswith(CONTENT_JSON_PREFIX):
        try:
            return json.loads(raw[len(CONTENT_JSON_PREFIX):])
        except json.JSONDecodeError:
            return raw
    return raw


def content_to_text(content: Any) -> str:
    """Flatten any content shape to plain text. Images and other binary parts are dropped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                elif "text_summary" in part:
                    parts.append(str(part["text_summary"]))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if isinstance(content, dict):
        return content.get("text") or content.get("text_summary") or json.dumps(content, ensure_ascii=False)
    return str(content)


def _parse_json(raw: Any, default: Any) -> Any:
    if raw is None:
        return default
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _resolve_tool_names(raw: Any) -> tuple[list[str], list[dict[str, Any]] | None]:
    """``tool_names`` is either a JSON list of names, a JSON list of full tool
    definitions, a comma-separated string, or a content hash that was already
    resolved through ``system_prompts`` by the SQL join."""
    parsed = _parse_json(raw, None)
    if parsed is None:
        if isinstance(raw, str) and raw:
            return [t.strip() for t in raw.split(",") if t.strip()], None
        return [], None
    if isinstance(parsed, dict):
        parsed = parsed.get("tools") or parsed.get("names") or list(parsed.values())
    if not isinstance(parsed, list):
        return [], None
    if parsed and all(isinstance(x, dict) for x in parsed):
        names: list[str] = []
        for d in parsed:
            fn = d.get("function") if isinstance(d.get("function"), dict) else d
            if fn.get("name"):
                names.append(str(fn["name"]))
        return names, parsed
    return [str(x) for x in parsed if x], None


class StateDB:
    """Snapshot-then-read wrapper around ``state.db``.

    Usage::

        with StateDB() as db:
            for session in db.sessions():
                msgs = db.messages(session.id)
    """

    def __init__(self, path: Path | str | None = None, snapshot: bool = True):
        self.path = Path(path) if path else default_state_db()
        if not self.path.exists():
            raise FileNotFoundError(f"Hermes state database not found at {self.path}")
        self._tmpdir: tempfile.TemporaryDirectory | None = None
        self._conn: sqlite3.Connection | None = None
        self._snapshot = snapshot

    # -- lifecycle -----------------------------------------------------------------
    def __enter__(self) -> StateDB:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        if self._conn is not None:
            return
        if self._snapshot:
            self._tmpdir = tempfile.TemporaryDirectory(prefix="hermes-distill-")
            dst_path = Path(self._tmpdir.name) / "state.db"
            src = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
            try:
                dst = sqlite3.connect(dst_path)
                with dst:
                    src.backup(dst)
                dst.close()
            finally:
                src.close()
            self._conn = sqlite3.connect(f"file:{dst_path.as_posix()}?mode=ro", uri=True)
        else:
            self._conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # -- introspection -------------------------------------------------------------
    def columns(self, table: str) -> set[str]:
        return {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}

    def schema_version(self) -> int | None:
        try:
            row = self.conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            return int(row[0]) if row else None
        except sqlite3.OperationalError:
            return None

    # -- sessions ------------------------------------------------------------------
    def sessions(
        self,
        sources: Iterable[str] | None = None,
        exclude_sources: Iterable[str] | None = None,
        since: float | None = None,
        min_messages: int = 2,
    ) -> list[Session]:
        cols = self.columns("sessions")
        has_prompts = "system_prompts" in {
            r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        has_prompt_hash = "system_prompt_hash" in cols and has_prompts
        has_tool_names = "tool_names" in cols
        select = [
            "s.id", "s.source", "s.model", "s.started_at", "s.ended_at", "s.end_reason",
            "COALESCE(s.message_count, 0)", "COALESCE(s.tool_call_count, 0)",
            "COALESCE(s.input_tokens, 0)", "COALESCE(s.output_tokens, 0)",
            "COALESCE(s.estimated_cost_usd, 0)", "s.title", "s.parent_session_id",
            "COALESCE(s.cache_read_tokens, 0)" if "cache_read_tokens" in cols else "0",
            "COALESCE(s.cache_write_tokens, 0)" if "cache_write_tokens" in cols else "0",
            "s.system_prompt",  # index 15
            "p.prompt" if has_prompt_hash else "NULL",
            # tool_names may itself be a hash into system_prompts (content-addressed tool list)
            "COALESCE(tp.prompt, s.tool_names)" if (has_tool_names and has_prompts) else (
                "s.tool_names" if has_tool_names else "NULL"
            ),
            "s.cwd" if "cwd" in cols else "NULL",  # index 18
        ]
        sql = "SELECT " + ", ".join(select) + " FROM sessions s"
        if has_prompt_hash:
            sql += " LEFT JOIN system_prompts p ON p.hash = s.system_prompt_hash"
        if has_tool_names and has_prompts:
            sql += " LEFT JOIN system_prompts tp ON tp.hash = s.tool_names"
        where = ["COALESCE(s.message_count, 0) >= ?"]
        params: list[Any] = [min_messages]
        if sources:
            src = list(sources)
            where.append("s.source IN (" + ",".join("?" * len(src)) + ")")
            params.extend(src)
        if exclude_sources:
            exc = list(exclude_sources)
            where.append("s.source NOT IN (" + ",".join("?" * len(exc)) + ")")
            params.extend(exc)
        if since is not None:
            where.append("s.started_at >= ?")
            params.append(since)
        sql += " WHERE " + " AND ".join(where) + " ORDER BY s.started_at"
        out: list[Session] = []
        for r in self.conn.execute(sql, params):
            names, schemas = _resolve_tool_names(r[17])
            out.append(
                Session(
                    id=r[0], source=r[1] or "unknown", model=r[2], started_at=r[3], ended_at=r[4],
                    end_reason=r[5], message_count=r[6], tool_call_count=r[7], input_tokens=r[8],
                    output_tokens=r[9], estimated_cost_usd=float(r[10] or 0), title=r[11],
                    parent_session_id=r[12], cache_read_tokens=r[13], cache_write_tokens=r[14],
                    system_prompt=r[15] or r[16], tool_names=names, tool_schemas=schemas, cwd=r[18],
                )
            )
        return out

    # -- messages ------------------------------------------------------------------
    def messages(self, session_id: str) -> list[Message]:
        cols = self.columns("messages")
        active_clause = " AND active = 1" if "active" in cols else ""
        sql = (
            "SELECT id, role, content, tool_call_id, tool_calls, tool_name, reasoning, "
            "reasoning_content, timestamp, finish_reason FROM messages "
            f"WHERE session_id = ?{active_clause} ORDER BY id"
        )
        out: list[Message] = []
        for r in self.conn.execute(sql, (session_id,)):
            tool_calls = _parse_json(r["tool_calls"], None)
            if tool_calls is not None and not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            out.append(
                Message(
                    id=r["id"], role=r["role"], content=decode_content(r["content"]),
                    tool_call_id=r["tool_call_id"], tool_calls=tool_calls, tool_name=r["tool_name"],
                    reasoning=r["reasoning"], reasoning_content=r["reasoning_content"],
                    timestamp=r["timestamp"], finish_reason=r["finish_reason"],
                )
            )
        return out
