"""``hermes-distill`` command line."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .filters import FilterConfig
from .state_db import StateDB, default_state_db

app = typer.Typer(
    name="hermes-distill",
    help="Turn your Hermes Agent session history into a fine-tuned local model, and plug it back in.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _version(value: bool) -> None:
    if value:
        console.print(f"hermes-distill {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show version."),
) -> None:
    pass


def _filter_config(
    min_assistant_turns: int, min_tool_calls: int, require_reasoning: bool, allow_incomplete: bool,
    max_tool_chars: int, no_dedupe: bool, no_redact: bool, drop_system: bool,
) -> FilterConfig:
    return FilterConfig(
        min_assistant_turns=min_assistant_turns,
        min_tool_calls=min_tool_calls,
        require_reasoning=require_reasoning,
        require_completed=not allow_incomplete,
        max_tool_response_chars=max_tool_chars,
        dedupe=not no_dedupe,
        redact_secrets=not no_redact,
        keep_system=not drop_system,
    )


def _since_epoch(days: int | None, after: str | None = None) -> float | None:
    """``--days N`` or ``--after <ISO datetime | epoch seconds>`` -> epoch lower bound."""
    if after:
        try:
            return float(after)
        except ValueError:
            from datetime import datetime

            return datetime.fromisoformat(after).timestamp()
    return time.time() - days * 86400 if days else None


def _print_stats(stats_dict: dict) -> None:
    t = Table(title="Hermes sessions", show_lines=False)
    t.add_column("metric")
    t.add_column("value", justify="right")
    for key in ("sessions_seen", "kept", "assistant_turns", "tool_calls", "redactions", "input_tokens",
                "cache_tokens", "output_tokens", "estimated_cost_usd", "sessions_with_schemas"):
        t.add_row(key, str(stats_dict[key]))
    console.print(t)
    if stats_dict.get("schema_sources"):
        console.print("tool schema sources: " + ", ".join(f"{k}={v}" for k, v in stats_dict["schema_sources"].items()))
    if stats_dict.get("hermes_tools_error"):
        console.print(f"[yellow]live Hermes tool schemas unavailable: {stats_dict['hermes_tools_error']}[/yellow]")
    for title, key in (("dropped (reason)", "dropped"), ("by source", "by_source"), ("by model", "by_model")):
        if stats_dict[key]:
            t2 = Table(title=title)
            t2.add_column("key")
            t2.add_column("n", justify="right")
            for k, v in sorted(stats_dict[key].items(), key=lambda kv: -kv[1]):
                t2.add_row(str(k), str(v))
            console.print(t2)
    if stats_dict["tool_usage"]:
        t3 = Table(title="tool usage (kept sessions)")
        t3.add_column("tool")
        t3.add_column("calls", justify="right")
        for k, v in list(stats_dict["tool_usage"].items())[:25]:
            t3.add_row(k, str(v))
        console.print(t3)


# ---------------------------------------------------------------------------
@app.command()
def stats(
    db: Path | None = typer.Option(None, help="Path to state.db (default: $HERMES_HOME/state.db)."),
    source: list[str] = typer.Option([], help="Only these sources (cli, telegram, tui, cron, ...)."),
    exclude_source: list[str] = typer.Option([], help="Exclude these sources."),
    days: int | None = typer.Option(None, help="Only sessions from the last N days."),
    after: str | None = typer.Option(None, help="Only sessions started after this ISO datetime or epoch."),
    min_assistant_turns: int = typer.Option(1),
    min_tool_calls: int = typer.Option(0),
    require_reasoning: bool = typer.Option(False),
    allow_incomplete: bool = typer.Option(False, help="Keep sessions that did not end with an assistant answer."),
    max_tool_chars: int = typer.Option(6000),
    no_dedupe: bool = typer.Option(False),
    no_redact: bool = typer.Option(False),
    drop_system: bool = typer.Option(False),
    tools_from_hermes: bool = typer.Option(True, help="Pull real tool schemas from the local Hermes install."),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Summarize what is in your Hermes history and what would survive the filters."""
    from .export import collect

    cfg = _filter_config(min_assistant_turns, min_tool_calls, require_reasoning, allow_incomplete,
                         max_tool_chars, no_dedupe, no_redact, drop_system)
    with StateDB(db) as sdb:
        console.print(f"[dim]state.db: {sdb.path}  (schema v{sdb.schema_version()})[/dim]")
        _, st, schemas = collect(sdb, cfg, source or None, exclude_source or None, _since_epoch(days, after),
                                 tools_from_hermes=tools_from_hermes)
    d = st.as_dict()
    d["tool_schemas_known"] = len(schemas)
    if as_json:
        console.print_json(json.dumps(d))
    else:
        _print_stats(d)
        console.print(f"[dim]tool schemas available: {len(schemas)}[/dim]")


@app.command()
def export(
    out: Path = typer.Option(Path("data"), help="Output directory."),
    fmt: str = typer.Option("messages", help="messages (OpenAI chat + tools) or sharegpt (Hermes trajectory)."),
    both: bool = typer.Option(False, help="Write both formats."),
    holdout: float = typer.Option(0.1, help="Fraction of sessions held out for eval (split by session id)."),
    seed: int = typer.Option(42),
    db: Path | None = typer.Option(None, help="Path to state.db (default: $HERMES_HOME/state.db)."),
    source: list[str] = typer.Option([]),
    exclude_source: list[str] = typer.Option([]),
    days: int | None = typer.Option(None),
    after: str | None = typer.Option(None, help="Only sessions started after this ISO datetime or epoch."),
    min_assistant_turns: int = typer.Option(1),
    min_tool_calls: int = typer.Option(0),
    require_reasoning: bool = typer.Option(False),
    allow_incomplete: bool = typer.Option(False),
    max_tool_chars: int = typer.Option(6000),
    no_dedupe: bool = typer.Option(False),
    no_redact: bool = typer.Option(False, help="Keep credential-looking strings (not recommended)."),
    drop_system: bool = typer.Option(False, help="Drop the Hermes system prompt from every sample."),
    tool_schemas: Path | None = typer.Option(None, help="JSON file {name: {name, description, parameters}} overriding tool definitions."),
    tools_from_hermes: bool = typer.Option(True, help="Pull real tool schemas from the local Hermes install."),
    tools_scope: str = typer.Option("used", help="Tools listed per sample: 'used' (only called tools) or 'all' (every Hermes tool)."),
    system_mode: str = typer.Option("keep", help="System prompt per sample: keep (full, ~4k tokens), short (first paragraph), drop."),
    before: str | None = typer.Option(None, help="Only sessions started before this ISO datetime or epoch."),
) -> None:
    """Export filtered conversations to JSONL (train + holdout) plus tools.json."""
    from .export import collect, write_jsonl

    if tools_scope not in ("used", "all"):
        raise typer.BadParameter("--tools-scope must be 'used' or 'all'")
    if system_mode not in ("keep", "short", "drop"):
        raise typer.BadParameter("--system-mode must be keep, short or drop")
    cfg = _filter_config(min_assistant_turns, min_tool_calls, require_reasoning, allow_incomplete,
                         max_tool_chars, no_dedupe, no_redact, drop_system)
    with StateDB(db) as sdb:
        console.print(f"[dim]state.db: {sdb.path}  (schema v{sdb.schema_version()})[/dim]")
        samples, st, schemas = collect(sdb, cfg, source or None, exclude_source or None, _since_epoch(days, after),
                                       tools_from_hermes=tools_from_hermes, tools_scope=tools_scope,
                                       system_mode=system_mode, before=_since_epoch(None, before))
    if tool_schemas:
        override = json.loads(tool_schemas.read_text(encoding="utf-8"))
        schemas.update({k: v for k, v in override.items()})
    if not samples:
        console.print("[red]Nothing to export after filtering.[/red] Try --allow-incomplete or --min-assistant-turns 1.")
        raise typer.Exit(1)
    formats = ["messages", "sharegpt"] if both else [fmt]
    results = {}
    for f in formats:
        results[f] = write_jsonl(samples, schemas, out, f, holdout, seed)
    (out / "stats.json").write_text(json.dumps(st.as_dict(), indent=2), encoding="utf-8")
    _print_stats(st.as_dict())
    for f, r in results.items():
        console.print(f"[green]{f}[/green]: {r['n_train']} train -> {r['train']}, {r['n_holdout']} holdout -> {r['holdout']}")
    console.print(f"tool schemas: {len(schemas)} -> {results[formats[0]]['tools']}")


@app.command()
def train(
    data: Path = typer.Option(Path("data/train.messages.jsonl")),
    output: Path = typer.Option(Path("runs/adapter")),
    base_model: str = typer.Option("Qwen/Qwen3-4B"),
    epochs: float = typer.Option(2.0),
    lr: float = typer.Option(2e-4),
    max_seq_len: int = typer.Option(4096, help="4k fits an 8 GB GPU with 'export --system-mode short'; 8k needs ~24 GB."),
    lora_r: int = typer.Option(16),
    lora_alpha: int = typer.Option(32),
    batch_size: int = typer.Option(1),
    grad_accum: int = typer.Option(8),
    no_4bit: bool = typer.Option(False, help="Load the base model in bf16/fp16 instead of 4-bit."),
    max_samples: int | None = typer.Option(None),
    drop_reasoning: bool = typer.Option(False, help="Train without <think> content."),
    no_chunked_loss: bool = typer.Option(False, help="Use the stock loss (materializes full logits; needs ~4 GB more VRAM at 4k)."),
    loss_chunk_size: int = typer.Option(1024),
    dry_run: bool = typer.Option(False, help="Render and tokenize only; print length stats and one example."),
) -> None:
    """QLoRA fine-tune a base model on the exported conversations."""
    from .train import TrainConfig
    from .train import dry_run as _dry
    from .train import train as _train

    cfg = TrainConfig(
        data=data, output=output, base_model=base_model, epochs=epochs, learning_rate=lr,
        max_seq_len=max_seq_len, lora_r=lora_r, lora_alpha=lora_alpha, batch_size=batch_size,
        grad_accum=grad_accum, load_4bit=not no_4bit, max_samples=max_samples, drop_reasoning=drop_reasoning,
        chunked_loss=not no_chunked_loss, loss_chunk_size=loss_chunk_size,
    )
    if dry_run:
        stats_ = _dry(cfg)
        example = stats_.pop("example", "")
        console.print_json(json.dumps(stats_))
        console.rule("first rendered example (truncated)")
        console.print(example)
        return
    summary = _train(cfg)
    console.print_json(json.dumps(summary))


@app.command()
def serve(
    adapter: Path = typer.Option(Path("runs/adapter")),
    name: str = typer.Option("hermes-distill", help="Ollama model name."),
    base_model: str | None = typer.Option(None, help="Defaults to the adapter's recorded base."),
    merged_dir: Path = typer.Option(Path("models/merged")),
    gguf: Path = typer.Option(Path("models/hermes-distill.gguf")),
    quant: str = typer.Option("q8_0", help="f16/bf16/q8_0 need only the Python converter; Q4_K_M etc. need llama-quantize."),
    llama_cpp: Path | None = typer.Option(None, help="Path to a llama.cpp checkout."),
    num_ctx: int = typer.Option(32768),
    skip_merge: bool = typer.Option(False),
    skip_gguf: bool = typer.Option(False),
    skip_ollama: bool = typer.Option(False),
) -> None:
    """Merge -> GGUF -> ollama create, then print the Hermes config to use it."""
    from .serve import hermes_config_snippet, merge_adapter, ollama_create, to_gguf

    if not skip_merge:
        console.print(f"merging {adapter} into {merged_dir} ...")
        merge_adapter(adapter, base_model, merged_dir)
    if not skip_gguf:
        console.print(f"converting {merged_dir} -> {gguf} ({quant}) ...")
        to_gguf(merged_dir, gguf, llama_cpp, quant)
    if not skip_ollama:
        console.print(f"ollama create {name} ...")
        ollama_create(name, gguf, num_ctx)
    console.rule("add this to Hermes")
    console.print(hermes_config_snippet(name))


@app.command("hermes-config")
def hermes_config(
    name: str = typer.Option("hermes-distill"),
    base_url: str = typer.Option("http://localhost:11434/v1"),
) -> None:
    """Print the config.yaml snippet that points Hermes at a local model."""
    from .serve import hermes_config_snippet

    console.print(hermes_config_snippet(name, base_url))


@app.command("eval")
def eval_cmd(
    data: Path = typer.Option(Path("data/holdout.messages.jsonl")),
    model: str = typer.Option(..., help="Model name at the endpoint (e.g. hermes-distill, qwen3:4b, gpt-4.1)."),
    base_url: str = typer.Option("http://localhost:11434/v1"),
    api_key: str = typer.Option("no-key", envvar="HERMES_DISTILL_EVAL_KEY"),
    max_samples: int | None = typer.Option(None),
    max_turns_per_sample: int = typer.Option(6),
    max_tokens: int = typer.Option(512),
    out: Path | None = typer.Option(None, help="Write the result JSON here."),
) -> None:
    """Next-action accuracy of a model against held-out Hermes sessions."""
    from .evaluate import EvalConfig, run_eval

    cfg = EvalConfig(data=data, model=model, base_url=base_url, api_key=api_key, max_samples=max_samples,
                     max_turns_per_sample=max_turns_per_sample, max_tokens=max_tokens)
    with console.status(f"evaluating {model} ..."):
        res = run_eval(cfg)
    d = res.as_dict()
    d["model"] = model
    d["data"] = str(data)
    t = Table(title=f"next-action eval: {model}")
    t.add_column("metric")
    t.add_column("value", justify="right")
    for k in ("turns", "action_acc", "tool_turns", "tool_acc", "args_exact", "json_valid", "errors", "avg_latency_s"):
        t.add_row(k, str(d[k]))
    console.print(t)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(d, indent=2), encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")


@app.command()
def tools(
    out: Path | None = typer.Option(None, help="Write {name: schema} JSON here (usable as --tool-schemas)."),
) -> None:
    """Dump the tool definitions of the local Hermes install."""
    from .hermes_tools import dump_hermes_tool_schemas, find_hermes_install, find_hermes_python

    install = find_hermes_install()
    console.print(f"[dim]install: {install}  python: {find_hermes_python(install) if install else None}[/dim]")
    schemas, err = dump_hermes_tool_schemas()
    if err:
        console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)
    t = Table(title=f"{len(schemas)} Hermes tools")
    t.add_column("tool")
    t.add_column("params")
    for name, s in sorted(schemas.items()):
        t.add_row(name, ", ".join((s["parameters"].get("properties") or {}).keys()))
    console.print(t)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(schemas, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")


@app.command()
def where() -> None:
    """Print the state.db path hermes-distill will read."""
    p = default_state_db()
    console.print(f"{p}  {'(exists)' if p.exists() else '(missing)'}")


if __name__ == "__main__":
    app()
