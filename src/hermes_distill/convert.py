"""Convert Hermes DB rows into training-ready conversations.

Two output formats are produced:

* ``messages``: OpenAI-style chat messages with native ``tool_calls`` /
  ``tool`` roles plus a ``tools`` list. This is what chat templates
  (Qwen3, Llama 3, Hermes 4) consume via ``apply_chat_template``.
* ``sharegpt``: the exact ``from``/``value`` format Hermes itself writes to
  ``trajectory_samples.jsonl`` (``<think>`` blocks, ``<tool_call>`` and
  ``<tool_response>`` XML). Mirrors ``agent/agent_runtime_helpers.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .state_db import Message, Session, content_to_text

TRAJECTORY_SYSTEM_PROMPT = (
    "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. "
    "You may call one or more functions to assist with the user query. If available tools are not relevant in assisting "
    "with user query, just respond in natural conversational language. Don't make assumptions about what values to plug "
    "into functions. After calling & executing the functions, you will be provided with function results within "
    "<tool_response> </tool_response> XML tags. Here are the available tools:\n"
    "<tools>\n{tools}\n</tools>\n"
    "For each function call return a JSON object, with the following pydantic model json schema for each:\n"
    "{{'title': 'FunctionCall', 'type': 'object', 'properties': {{'name': {{'title': 'Name', 'type': 'string'}}, "
    "'arguments': {{'title': 'Arguments', 'type': 'object'}}}}, 'required': ['name', 'arguments']}}\n"
    "Each function call should be enclosed within <tool_call> </tool_call> XML tags.\n"
    "Example:\n<tool_call>\n{{'name': <function-name>,'arguments': <args-dict>}}\n</tool_call>"
)

_SCRATCHPAD_OPEN = "<REASONING_SCRATCHPAD>"
_SCRATCHPAD_CLOSE = "</REASONING_SCRATCHPAD>"


# ---------------------------------------------------------------------------
# DB rows -> OpenAI-style messages
# ---------------------------------------------------------------------------

def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any] | None:
    """Return ``{"id", "type": "function", "function": {"name", "arguments": dict}}`` or None."""
    if not isinstance(tc, dict):
        return None
    fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
    name = fn.get("name")
    if not name:
        return None
    return {
        "id": tc.get("id") or "",
        "type": "function",
        "function": {"name": str(name), "arguments": _parse_args(fn.get("arguments"))},
    }


def to_openai_messages(session: Session, rows: list[Message], include_system: bool = True) -> list[dict[str, Any]]:
    """Flatten DB rows into OpenAI chat messages. Reasoning is kept in a ``reasoning`` field."""
    out: list[dict[str, Any]] = []
    if include_system and session.system_prompt:
        out.append({"role": "system", "content": session.system_prompt})
    for m in rows:
        if m.role == "system":
            if not include_system:
                continue
            if out and out[0]["role"] == "system":
                continue  # the session row already provided the system prompt
            out.append({"role": "system", "content": content_to_text(m.content)})
        elif m.role == "user":
            out.append({"role": "user", "content": content_to_text(m.content)})
        elif m.role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": content_to_text(m.content)}
            calls = [c for c in (normalize_tool_call(tc) for tc in (m.tool_calls or [])) if c]
            if calls:
                msg["tool_calls"] = calls
            reasoning = m.reasoning or m.reasoning_content
            if reasoning and reasoning.strip():
                msg["reasoning"] = reasoning
            out.append(msg)
        elif m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": content_to_text(m.content),
                    "tool_call_id": m.tool_call_id or "",
                    "name": m.tool_name or "",
                }
            )
    return out


# ---------------------------------------------------------------------------
# Tool schema inference
# ---------------------------------------------------------------------------

_JSON_TYPES = {str: "string", bool: "boolean", int: "integer", float: "number", list: "array", dict: "object"}


def schemas_from_session(session: Session) -> dict[str, dict[str, Any]]:
    """Real tool definitions when Hermes persisted them on the session row."""
    out: dict[str, dict[str, Any]] = {}
    for d in session.tool_schemas or []:
        fn = d.get("function") if isinstance(d.get("function"), dict) else d
        name = fn.get("name")
        if not name:
            continue
        out[str(name)] = {
            "name": str(name),
            "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
        }
    return out


def infer_tool_schemas(conversations: list[list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """Build minimal JSON schemas from every tool call observed in the data.

    Hermes passes tool schemas to the API at call time; depending on the version
    the DB may only know tool *names*. Inferring from observed arguments gives a
    faithful-enough ``<tools>`` block for training; pass ``--tool-schemas`` to
    override with the real definitions when you have them.
    """
    props: dict[str, dict[str, set[str]]] = {}
    required: dict[str, dict[str, int]] = {}
    counts: dict[str, int] = {}
    for conv in conversations:
        for msg in conv:
            for tc in msg.get("tool_calls") or []:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                counts[name] = counts.get(name, 0) + 1
                p = props.setdefault(name, {})
                r = required.setdefault(name, {})
                for key, value in args.items():
                    p.setdefault(key, set()).add(_JSON_TYPES.get(type(value), "string"))
                    r[key] = r.get(key, 0) + 1
    schemas: dict[str, dict[str, Any]] = {}
    for name, p in props.items():
        n = counts[name]
        properties = {k: {"type": sorted(v)[0] if len(v) == 1 else "string"} for k, v in sorted(p.items())}
        req = sorted(k for k, c in required[name].items() if c == n)
        schemas[name] = {
            "name": name,
            "description": f"{name} tool (schema inferred from {n} observed calls)",
            "parameters": {"type": "object", "properties": properties, "required": req},
        }
    return schemas


def schemas_to_openai_tools(schemas: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {"name": s["name"], "description": s.get("description", ""), "parameters": s["parameters"]},
        }
        for s in schemas.values()
    ]


def format_tools_block(schemas: dict[str, dict[str, Any]], names: list[str] | None = None) -> str:
    """JSON list of tool definitions in the shape Hermes embeds in ``<tools>``."""
    selected = [schemas[n] for n in (names or list(schemas)) if n in schemas]
    return json.dumps(
        [{"name": s["name"], "description": s.get("description", ""), "parameters": s["parameters"], "required": None}
         for s in selected],
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# OpenAI-style messages -> ShareGPT (Hermes trajectory format)
# ---------------------------------------------------------------------------

def convert_scratchpad_to_think(content: str) -> str:
    if not content or _SCRATCHPAD_OPEN not in content:
        return content
    return content.replace(_SCRATCHPAD_OPEN, "<think>").replace(_SCRATCHPAD_CLOSE, "</think>")


def _with_think_block(content: str) -> str:
    return content if "<think>" in content else "<think>\n</think>\n" + content


def _gpt_prefix(msg: dict[str, Any]) -> str:
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    if reasoning and reasoning.strip():
        return f"<think>\n{reasoning}\n</think>\n"
    return ""


def _tool_call_turn(msg: dict[str, Any]) -> str:
    content = _gpt_prefix(msg)
    if msg.get("content") and msg["content"].strip():
        content += convert_scratchpad_to_think(msg["content"]) + "\n"
    for tc in msg.get("tool_calls") or []:
        payload = {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
        content += f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>\n"
    return _with_think_block(content).rstrip()


def _tool_responses(assistant: dict[str, Any], messages: list[dict[str, Any]], start: int) -> tuple[list[str], int]:
    blocks: list[str] = []
    j = start
    calls = assistant.get("tool_calls") or []
    while j < len(messages) and messages[j]["role"] == "tool":
        tool_msg = messages[j]
        content: Any = tool_msg.get("content") or ""
        try:
            if isinstance(content, str) and content.strip().startswith(("{", "[")):
                content = json.loads(content)
        except json.JSONDecodeError:
            pass
        idx = len(blocks)
        name = calls[idx]["function"]["name"] if idx < len(calls) else (tool_msg.get("name") or "unknown")
        payload = json.dumps(
            {"tool_call_id": tool_msg.get("tool_call_id", ""), "name": name, "content": content}, ensure_ascii=False
        )
        blocks.append(f"<tool_response>\n{payload}\n</tool_response>")
        j += 1
    return blocks, j


def to_sharegpt(messages: list[dict[str, Any]], tools_block: str) -> list[dict[str, Any]]:
    """Mirror of the Hermes ``convert_to_trajectory_format`` operating on OpenAI-style messages."""
    trajectory: list[dict[str, Any]] = [
        {"from": "system", "value": TRAJECTORY_SYSTEM_PROMPT.format(tools=tools_block)}
    ]
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]
        if role == "system":
            i += 1
            continue
        if role == "assistant":
            if msg.get("tool_calls"):
                trajectory.append({"from": "gpt", "value": _tool_call_turn(msg)})
                blocks, j = _tool_responses(msg, messages, i + 1)
                if blocks:
                    trajectory.append({"from": "tool", "value": "\n".join(blocks)})
                    i = j - 1
            else:
                content = _gpt_prefix(msg) + convert_scratchpad_to_think(msg.get("content") or "")
                trajectory.append({"from": "gpt", "value": _with_think_block(content).strip()})
        elif role == "user":
            trajectory.append({"from": "human", "value": msg.get("content") or ""})
        elif role == "tool":
            # Orphan tool result (its assistant turn was dropped); keep the data visible.
            trajectory.append({"from": "tool", "value": msg.get("content") or ""})
        i += 1
    return trajectory


_THINK_RE = re.compile(r"<think>\s*</think>\s*")


def strip_empty_think(text: str) -> str:
    return _THINK_RE.sub("", text)
