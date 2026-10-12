"""Pull the real tool definitions out of a local Hermes install.

Hermes only persists a per-session tool pin for some session kinds (MCP tool
agent). For everything else the DB knows tool *names* at best. The schemas the
teacher actually saw live in ``model_tools.get_tool_definitions()`` inside the
Hermes checkout, so we run that in Hermes' own virtualenv and read the result
back. Best effort: any failure returns an empty dict and the exporter falls
back to inferred schemas.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .state_db import default_hermes_home

_DUMP_SCRIPT = r"""
import json, sys
out = sys.argv[1]
try:
    import model_tools
    defs = model_tools.get_tool_definitions()
except Exception as exc:  # pragma: no cover - diagnostic path
    defs = {"error": repr(exc)}
with open(out, "w", encoding="utf-8") as f:
    json.dump(defs, f)
"""


def find_hermes_install() -> Path | None:
    """The Hermes source checkout: ``$HERMES_INSTALL_DIR``, ``hermes --version``, then defaults."""
    env = os.environ.get("HERMES_INSTALL_DIR", "").strip()
    if env and (Path(env) / "model_tools.py").exists():
        return Path(env)
    exe = shutil.which("hermes")
    if exe:
        try:
            out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60).stdout
            m = re.search(r"Install directory:\s*(.+)", out)
            if m and (Path(m.group(1).strip()) / "model_tools.py").exists():
                return Path(m.group(1).strip())
        except (subprocess.SubprocessError, OSError):
            pass
    for cand in (default_hermes_home() / "hermes-agent", Path.home() / ".hermes" / "hermes-agent"):
        if (cand / "model_tools.py").exists():
            return cand
    return None


def find_hermes_python(install: Path) -> Path | None:
    """Interpreter of the Hermes runtime venv (newest environment wins)."""
    home = default_hermes_home()
    bin_dir, exe = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    candidates = [
        *(home / "installs").glob(f"*/environments/*/venv/{bin_dir}/{exe}"),
        install / "venv" / bin_dir / exe,
        home / "venv" / bin_dir / exe,
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return None
    # Newest environment directory wins (a `hermes pm install --extra` creates a fresh one).
    return max(existing, key=lambda p: p.parents[2].stat().st_mtime)


def normalize_defs(defs: Any) -> dict[str, dict[str, Any]]:
    """OpenAI-style ``{"type": "function", "function": {...}}`` or bare dicts -> ``{name: schema}``."""
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(defs, list):
        return out
    for d in defs:
        if not isinstance(d, dict):
            continue
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


def dump_hermes_tool_schemas(timeout: float = 240.0) -> tuple[dict[str, dict[str, Any]], str | None]:
    """Return ``({name: schema}, error)``. ``error`` explains an empty result."""
    install = find_hermes_install()
    if install is None:
        return {}, "Hermes install not found (set HERMES_INSTALL_DIR)"
    python = find_hermes_python(install)
    if python is None:
        return {}, f"Hermes venv python not found under {default_hermes_home()}"
    with tempfile.TemporaryDirectory(prefix="hermes-distill-tools-") as tmp:
        out = Path(tmp) / "tools.json"
        try:
            proc = subprocess.run(
                [str(python), "-c", _DUMP_SCRIPT, str(out)],
                cwd=str(install), capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return {}, f"could not run Hermes python: {exc}"
        if not out.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return {}, "tool dump produced no output: " + " | ".join(tail)
        data = json.loads(out.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "error" in data:
        return {}, f"get_tool_definitions failed: {data['error']}"
    schemas = normalize_defs(data)
    return schemas, None if schemas else "no tool definitions returned"


def merge_schemas(*layers: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge schema dicts; earlier layers win (pass most trusted first)."""
    merged: dict[str, dict[str, Any]] = {}
    for layer in layers:
        for name, schema in layer.items():
            merged.setdefault(name, schema)
    return merged
