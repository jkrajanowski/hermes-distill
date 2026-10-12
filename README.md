# hermes-distill

**Train your own local Hermes from your agent's history.**

[Hermes Agent](https://github.com/NousResearch/hermes-agent) logs every session it runs into a local SQLite
database: user turns, tool calls, tool results, reasoning, the tool schemas it was given. That is a
tool-calling dataset shaped exactly like *your* work, produced by a frontier model. `hermes-distill`
turns it into a small local model and plugs that model back into Hermes.

```
state.db  ──export──▶  train.jsonl  ──train──▶  LoRA adapter  ──serve──▶  Ollama model  ──▶  ~/.hermes/config.yaml
                       holdout.jsonl ─────────────────────────eval───────▶  teacher vs base vs distilled
```

Why: Hermes users on cheap VPSes spend most of their money on cron jobs and routine tool use. A 4B model
that has seen a few hundred of your own sessions handles those for free, on a laptop GPU, with your data
never leaving the machine.

> Status: **alpha**. `export` and `stats` are tested against the real Hermes schema (v25+).
> `train`, `serve` and `eval` are implemented and wired up but not yet validated on a full run.
> See [Roadmap](#roadmap).

## Install

```bash
pip install hermes-distill            # exporter + CLI only
pip install "hermes-distill[train]"   # + torch / transformers / peft / trl / bitsandbytes
pip install "hermes-distill[eval]"    # + openai client for the eval harness
```

From source:

```bash
git clone https://github.com/jkrajanowski/hermes-distill && cd hermes-distill
pip install -e ".[train,eval,dev]"
```

## Quickstart

```bash
# 1. What is in my history, and what survives the quality filters?
hermes-distill stats

# 2. Export train/holdout JSONL (secrets redacted, duplicates dropped, dangling tool calls trimmed)
hermes-distill export --out data --min-tool-calls 1 --both

# 3. Check token lengths and look at one rendered example before spending GPU time
hermes-distill train --dry-run --base-model Qwen/Qwen3-4B

# 4. QLoRA on a single 8 GB GPU (defaults: 4-bit base, r=16, 4k context)
hermes-distill train --base-model Qwen/Qwen3-4B --epochs 2 --output runs/qwen3-4b

# 5. Merge, convert to GGUF, register with Ollama, print the Hermes config
hermes-distill serve --adapter runs/qwen3-4b --name hermes-distill --llama-cpp ~/llama.cpp

# 6. Compare next-action accuracy on held-out sessions
hermes-distill eval --model hermes-distill                      # your distilled model via Ollama
hermes-distill eval --model qwen3:4b                            # the untouched base
hermes-distill eval --model anthropic/claude-sonnet-4.6 \
    --base-url https://openrouter.ai/api/v1 --api-key $OPENROUTER_API_KEY   # the teacher
```

Then point Hermes at it:

```yaml
# ~/.hermes/config.yaml
model:
  default: "hermes-distill"
  provider: "custom"
  base_url: "http://localhost:11434/v1"
```

## What `export` does

| Step | Detail |
|---|---|
| Locate | Same rule as Hermes: `$HERMES_HOME`, else `~/.hermes` on Linux/macOS or `%LOCALAPPDATA%\hermes` on Windows. `hermes-distill where` prints it. |
| Snapshot | Copies `state.db` with the SQLite backup API, so a running gateway is never touched and the WAL is included. |
| Read | Sessions and `active = 1` messages; resolves content-addressed system prompts and the per-session **tool-schema pin** Hermes stores, so training uses the real tool definitions, not guesses. Falls back to schema inference from observed calls. |
| Repair | Aligns tool results to their calls by `tool_call_id`, drops orphan results, trims dangling trailing calls so every sample ends with an assistant answer. |
| Clean | Truncates giant tool outputs (head + tail), drops oversize sessions, dedupes by content hash. |
| Redact | Replaces API keys, bearer tokens, PEM blocks and `KEY=value` secrets with `[REDACTED:kind]`. |
| Write | `train.messages.jsonl` + `holdout.messages.jsonl` (OpenAI chat + `tools`, rendered through the base model's own chat template at train time) and/or `*.sharegpt.jsonl` (the exact `from`/`value` + `<think>` + `<tool_call>` format Hermes writes itself). Split is deterministic by session id. |

`stats` runs the same pipeline without writing anything and prints what was kept, what was dropped and why,
tool usage, token totals and the estimated spend you are about to distill.

## Eval

`eval` replays held-out conversations turn by turn against any OpenAI-compatible endpoint and scores whether
the model picks the same next action as the teacher: tool vs. answer, which tool, exact arguments, JSON
validity. Same harness for the teacher, the base model and the distilled model, so the numbers are comparable.

## Roadmap

- [x] Exporter grounded in the real `state.db` schema, with tests
- [x] Filters: repair, trim, truncate, dedupe, redact
- [x] QLoRA trainer (TRL + PEFT + bitsandbytes), dry-run mode
- [x] Merge, GGUF, Ollama, Hermes config
- [x] Next-action eval harness
- [ ] Validate a full run on a consumer GPU and publish numbers
- [ ] Assistant-only loss masking
- [ ] Ship as a Hermes plugin (`hermes distill ...` via the `hermes_agent.plugins` entry point)
- [ ] Rejection-sampled round 2 using Hermes' batch runner

## How it relates to Hermes' own tooling

Hermes ships `batch_runner.py` (generate trajectories from a prompt file) and `trajectory_compressor.py`.
Those produce *new* data by running the agent. `hermes-distill` works on the data you already have, and
closes the loop by training, serving and evaluating.

## License

MIT
