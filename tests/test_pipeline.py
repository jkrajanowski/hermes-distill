"""End-to-end test on a synthetic state.db built with the real Hermes schema subset."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_distill.convert import (
    TRAJECTORY_SYSTEM_PROMPT,
    format_tools_block,
    infer_tool_schemas,
    schemas_from_session,
    to_openai_messages,
    to_sharegpt,
)
from hermes_distill.export import collect, write_jsonl
from hermes_distill.filters import FilterConfig, apply_filters, redact, repair_tool_pairs, trim_dangling
from hermes_distill.state_db import CONTENT_JSON_PREFIX, StateDB

SCHEMA = """
CREATE TABLE schema_version (version INTEGER NOT NULL);
CREATE TABLE system_prompts (hash TEXT PRIMARY KEY, prompt TEXT NOT NULL);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY, source TEXT NOT NULL, model TEXT, system_prompt TEXT, system_prompt_hash TEXT,
    parent_session_id TEXT, started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
    message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, estimated_cost_usd REAL,
    title TEXT, tool_names TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
    tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL, token_count INTEGER,
    finish_reason TEXT, reasoning TEXT, reasoning_content TEXT, active INTEGER NOT NULL DEFAULT 1
);
"""

TOOL_PIN = {
    "version": 1,
    "tools": [
        {"type": "function", "function": {"name": "terminal", "description": "Execute shell commands",
                                            "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                                           "required": ["command"]}}},
    ],
}


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO schema_version VALUES (25)")
    conn.execute("INSERT INTO system_prompts VALUES ('h1', 'You are Hermes.')")
    conn.execute("INSERT INTO system_prompts VALUES ('h2', ?)", (json.dumps(TOOL_PIN),))

    def add_session(sid: str, source: str, tool_names: str | None, model: str = "anthropic/claude-sonnet-4.6"):
        conn.execute(
            "INSERT INTO sessions (id, source, model, system_prompt_hash, started_at, message_count, tool_call_count,"
            " input_tokens, output_tokens, estimated_cost_usd, tool_names)"
            " VALUES (?, ?, ?, 'h1', 1700000000, 4, 1, 1000, 200, 0.01, ?)",
            (sid, source, model, tool_names),
        )

    def add(sid: str, role: str, content, **kw):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_call_id, tool_calls, tool_name, timestamp,"
            " reasoning, active) VALUES (?, ?, ?, ?, ?, ?, 1700000001, ?, ?)",
            (sid, role, content, kw.get("tool_call_id"), kw.get("tool_calls"), kw.get("tool_name"),
             kw.get("reasoning"), kw.get("active", 1)),
        )

    # Session A: complete tool-using conversation, tool pin stored as hash.
    add_session("A", "cli", "h2")
    add("A", "user", "What Python version is installed?")
    add("A", "assistant", "", tool_calls=json.dumps([
        {"id": "call_1", "type": "function", "function": {"name": "terminal",
                                                          "arguments": json.dumps({"command": "python3 --version"})}}
    ]), reasoning="I should run python3 --version.")
    add("A", "tool", "Python 3.11.6", tool_call_id="call_1", tool_name="terminal")
    add("A", "assistant", "Python 3.11.6 is installed. My key is sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234 ok")
    add("A", "assistant", "stale compacted turn", active=0)

    # Session B: dangling tool call at the end (interrupted) + multimodal user content.
    add_session("B", "telegram", json.dumps(["terminal"]))
    add("B", "user", CONTENT_JSON_PREFIX + json.dumps([{"type": "text", "text": "look at this"},
                                                      {"type": "image_url", "image_url": {"url": "data:..."}}]))
    add("B", "assistant", "Sure.")
    add("B", "assistant", "", tool_calls=json.dumps([
        {"id": "call_2", "type": "function", "function": {"name": "terminal", "arguments": "{\"command\": \"ls\"}"}}
    ]))

    # Session C: exact duplicate of A's content (should be deduped).
    add_session("C", "cli", "h2")
    add("C", "user", "What Python version is installed?")
    add("C", "assistant", "", tool_calls=json.dumps([
        {"id": "call_9", "type": "function", "function": {"name": "terminal",
                                                          "arguments": json.dumps({"command": "python3 --version"})}}
    ]))
    add("C", "tool", "Python 3.11.6", tool_call_id="call_9", tool_name="terminal")
    add("C", "assistant", "Python 3.11.6 is installed. My key is sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234 ok")
    conn.commit()
    conn.close()
    return path


def test_reads_sessions_and_resolves_tool_pin(db_path: Path):
    with StateDB(db_path, snapshot=True) as db:
        assert db.schema_version() == 25
        sessions = {s.id: s for s in db.sessions()}
        assert set(sessions) == {"A", "B", "C"}
        a = sessions["A"]
        assert a.system_prompt == "You are Hermes."
        assert a.tool_names == ["terminal"]
        assert schemas_from_session(a)["terminal"]["parameters"]["required"] == ["command"]
        b = sessions["B"]
        assert b.tool_names == ["terminal"] and b.tool_schemas is None
        rows = db.messages("A")
        assert [r.role for r in rows] == ["user", "assistant", "tool", "assistant"]  # inactive row skipped
        assert rows[1].tool_calls[0]["function"]["name"] == "terminal"


def test_openai_and_sharegpt_conversion(db_path: Path):
    with StateDB(db_path) as db:
        a = next(s for s in db.sessions() if s.id == "A")
        msgs = to_openai_messages(a, db.messages("A"))
    assert msgs[0] == {"role": "system", "content": "You are Hermes."}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"command": "python3 --version"}
    assert msgs[2]["reasoning"] == "I should run python3 --version."
    assert msgs[3] == {"role": "tool", "content": "Python 3.11.6", "tool_call_id": "call_1", "name": "terminal"}

    schemas = infer_tool_schemas([msgs])
    traj = to_sharegpt(msgs, format_tools_block(schemas))
    assert [t["from"] for t in traj] == ["system", "human", "gpt", "tool", "gpt"]
    assert traj[0]["value"].startswith(TRAJECTORY_SYSTEM_PROMPT.split("{tools}")[0])
    assert traj[2]["value"] == (
        "<think>\nI should run python3 --version.\n</think>\n"
        "<tool_call>\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"python3 --version\"}}\n</tool_call>"
    )
    assert json.loads(traj[3]["value"].split("\n")[1]) == {"tool_call_id": "call_1", "name": "terminal",
                                                           "content": "Python 3.11.6"}
    assert traj[4]["value"].startswith("<think>\n</think>\n")


def test_filters_redact_trim_and_repair():
    text, hits = redact("token sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234 and AIzaSyFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE00")
    assert hits == 2 and "sk-ant" not in text and "AIza" not in text

    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function",
                                                             "function": {"name": "terminal", "arguments": {}}}]},
    ]
    assert trim_dangling(msgs) == []
    repaired = repair_tool_pairs(msgs + [{"role": "tool", "content": "out", "tool_call_id": "x", "name": "terminal"},
                                         {"role": "tool", "content": "orphan", "tool_call_id": "zzz"}])
    assert [m["role"] for m in repaired] == ["user", "assistant", "tool"]


def test_schema_merge_priority():
    from hermes_distill.hermes_tools import merge_schemas, normalize_defs

    live = normalize_defs([{"type": "function", "function": {"name": "terminal", "description": "live",
                                                             "parameters": {"type": "object", "properties": {}}}},
                           {"name": "web_search", "parameters": {"type": "object", "properties": {"q": {}}}}])
    assert set(live) == {"terminal", "web_search"} and live["web_search"]["description"] == ""
    merged = merge_schemas({"terminal": {"name": "terminal", "description": "pin", "parameters": {}}}, live,
                           {"terminal": {"name": "terminal", "description": "inferred", "parameters": {}},
                            "read_file": {"name": "read_file", "description": "inferred", "parameters": {}}})
    assert merged["terminal"]["description"] == "pin"
    assert merged["web_search"]["description"] == ""
    assert merged["read_file"]["description"] == "inferred"


def test_collect_and_write(db_path: Path, tmp_path: Path):
    cfg = FilterConfig()
    with StateDB(db_path) as db:
        samples, stats, schemas = collect(db, cfg, tools_from_hermes=False)
    ids = sorted(s.session.id for s in samples)
    assert ids == ["A", "B"], stats.as_dict()           # C deduped against A
    assert stats.dropped["duplicate"] == 1
    assert stats.sessions_with_schemas == 2             # A and C carry the pin
    assert "terminal" in schemas and schemas["terminal"]["description"] == "Execute shell commands"
    a = next(s for s in samples if s.session.id == "A")
    assert a.redactions == 1 and "[REDACTED:anthropic]" in a.messages[-1]["content"]
    b = next(s for s in samples if s.session.id == "B")
    assert [m["role"] for m in b.messages] == ["system", "user", "assistant"]  # dangling call trimmed
    assert b.messages[1]["content"] == "look at this"

    out = tmp_path / "out"
    r = write_jsonl(samples, schemas, out, "messages", holdout=0.0)
    assert r["n_train"] == 2 and r["n_holdout"] == 0
    rec = json.loads((out / "train.messages.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["tools"][0]["function"]["name"] == "terminal"
    assert rec["messages"][2]["reasoning_content"] == "I should run python3 --version."
    write_jsonl(samples, schemas, out, "sharegpt", holdout=0.0)
    rec2 = json.loads((out / "train.sharegpt.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec2["conversations"][0]["from"] == "system" and rec2["completed"] is True


def test_default_home_resolution(monkeypatch: pytest.MonkeyPatch):
    from hermes_distill import state_db

    monkeypatch.setenv("HERMES_HOME", "/tmp/custom-hermes")
    assert state_db.default_state_db() == Path("/tmp/custom-hermes") / "state.db"
    monkeypatch.delenv("HERMES_HOME")
    monkeypatch.setattr(state_db.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\x\AppData\Local")
    assert state_db.default_state_db() == Path(r"C:\Users\x\AppData\Local") / "hermes" / "state.db"
    monkeypatch.setattr(state_db.sys, "platform", "linux")
    assert state_db.default_state_db() == Path.home() / ".hermes" / "state.db"


def test_assistant_only_labels_mask_non_assistant_tokens():
    from hermes_distill.train import tokenize_with_assistant_labels

    class Tok:  # character-level fake fast tokenizer
        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True, truncation=True, max_length=10**6):
            return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    text = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nyo<|im_end|>\n"
    out = tokenize_with_assistant_labels(text, Tok(), 10**6, assistant_only=True)
    trained = "".join(chr(i) for i in out["labels"] if i != -100)
    assert trained == "yo<|im_end|>"
    assert out["trained_tokens"] == len("yo<|im_end|>")
    full = tokenize_with_assistant_labels(text, Tok(), 10**6, assistant_only=False)
    assert full["labels"] == full["input_ids"]


def test_outside_cwd_detection():
    from hermes_distill.filters import touches_outside_cwd

    cwd = r"C:\Users\x\sandbox_1"

    def conv(args):
        return [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "type": "function",
                                                                     "function": {"name": "terminal", "arguments": args}}]}]

    assert not touches_outside_cwd(conv({"command": "ls data/ && cat out/report.md"}), cwd)
    assert not touches_outside_cwd(conv({"path": r"C:\Users\x\sandbox_1\data\sales.csv"}), cwd)
    assert not touches_outside_cwd(conv({"command": "cd /c/Users/x/sandbox_1/data && wc -l sales.csv"}), cwd)
    assert touches_outside_cwd(conv({"command": "cd ~ && git status"}), cwd)
    assert touches_outside_cwd(conv({"path": r"C:\Users\x\other\README.md"}), cwd)
    assert touches_outside_cwd(conv({"command": "find /home -name pyproject.toml"}), cwd)
    assert touches_outside_cwd(conv({"command": "cat /etc/passwd"}), cwd)
    assert not touches_outside_cwd(conv({"command": "python -c 'print(1.5/2)'"}), cwd)
    assert not touches_outside_cwd(conv({"code": "with open('data/sales.csv') as f:\n    r = f.read()"}), cwd)
    assert not touches_outside_cwd(conv({"command": "sort data/x.log > /tmp/counts.txt && cat /tmp/counts.txt"}), cwd)
    assert not touches_outside_cwd(conv({"command": "cd data/buggy && python - > ../../out/log.txt"}), cwd)


def test_apply_filters_reasons():
    cfg = FilterConfig(min_tool_calls=1)
    res = apply_filters([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}], cfg)
    assert res.messages is None and res.reason == "too_few_tool_calls"
