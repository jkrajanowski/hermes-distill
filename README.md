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

> Status: **alpha, but the whole loop has run end to end** on a live Hermes 0.21 install and an
> 8 GB laptop GPU: export, QLoRA, GGUF, Ollama, eval. See [First results](#first-results-rtx-4060-laptop-8-gb)
> for numbers and the lessons, and the [Roadmap](#roadmap) for what is next.

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
#    (Hermes' system prompt alone is ~4k tokens; the default 8k context keeps most sessions whole)
hermes-distill train --dry-run --base-model Qwen/Qwen3-4B

# 4. QLoRA on a single 8 GB GPU (defaults: 4-bit base, r=16, 8k context)
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
| Read | Sessions and `active = 1` messages; resolves content-addressed system prompts and the per-session **tool-schema pin** Hermes stores for some session kinds. |
| Tool schemas | Priority: the session pin (what the teacher saw) → the live Hermes install's `get_tool_definitions()`, run inside Hermes' own venv (`hermes-distill tools` shows it) → inference from observed calls. `--tools-scope all` lists every Hermes tool per sample instead of only the ones it called. |
| Repair | Aligns tool results to their calls by `tool_call_id`, drops orphan results, trims dangling trailing calls so every sample ends with an assistant answer. |
| Clean | Truncates giant tool outputs (head + tail), drops oversize sessions, dedupes by content hash. |
| Redact | Replaces API keys, bearer tokens, PEM blocks and `KEY=value` secrets with `[REDACTED:kind]`. |
| Write | `train.messages.jsonl` + `holdout.messages.jsonl` (OpenAI chat + `tools`, rendered through the base model's own chat template at train time) and/or `*.sharegpt.jsonl` (the exact `from`/`value` + `<think>` + `<tool_call>` format Hermes writes itself). Split is deterministic by session id. |

`stats` runs the same pipeline without writing anything and prints what was kept, what was dropped and why,
tool usage, token totals and the estimated spend you are about to distill.

## Fitting a small GPU

The Hermes system prompt is ~4k tokens on its own, and the loss needs a `seq_len x vocab` logits
tensor (152k vocab for Qwen3), so sequence length is the knob that decides whether training fits:

| GPU | `export --system-mode` | `train --max-seq-len` |
|---|---|---|
| 8 GB (RTX 4060 laptop) | `short` (first paragraph) or `drop` | 4096 |
| 16 GB | `short` | 8192 |
| 24 GB+ | `keep` (full prompt, exactly what Hermes sends at inference) | 8192 |

The trainer pins the model to one GPU on purpose. With `device_map="auto"` transformers quietly
offloads layers to CPU when VRAM runs out, which looks like "training is running" while using
15 GB of host RAM and making no progress. Better to fail with an out-of-memory error and lower
the context.

For a clean train/eval split by time, use `--before` / `--after` with an ISO datetime or epoch
seconds, for example when you generate dedicated eval sessions after exporting the training set.

## Eval

`eval` replays held-out conversations turn by turn against any OpenAI-compatible endpoint and scores whether
the model picks the same next action as the teacher: tool vs. answer, which tool, exact arguments, JSON
validity. Same harness for the teacher, the base model and the distilled model, so the numbers are comparable.

## First results (RTX 4060 laptop, 8 GB)

Setup: 61 training sessions generated by running Hermes (teacher: claude-sonnet-5) on a sandbox of
this repo plus synthetic CSVs and logs, 18 separate eval sessions generated afterwards (54 assistant
turns). QLoRA on Qwen3-4B, r=16, 2 epochs, 4k context, short system prompt, about 10 minutes per run.
Both local models are q8_0 GGUFs served by Ollama; eval budget 1024 output tokens per turn.

| model | action type | right tool | args exact | valid JSON | s/turn |
|---|---:|---:|---:|---:|---:|
| claude-sonnet-5 (teacher, replaying itself) | 96.3% | 80.6% | 19.4% | 100% | 1.9 |
| Qwen3-4B base | 70.4% | 44.4% | 8.3% | 55.6% | 15.1 |
| hermes-distill v0: full-sequence loss | 61.1% | 36.1% | 5.6% | 41.7% | 12.9 |
| hermes-distill v1: assistant-only loss | 66.7% | **52.8%** | **11.1%** | **61.1%** | **8.2** |

With 54 scored turns the confidence interval on each percentage is roughly ±13 points, so read the
table as "v1 beats the base on tool choice and speed, v0 does not", not as precise values. Per tool,
v1 picked `search_files` on 5 of 8 turns where the teacher did; the base model never picked it once.

What the first two runs taught us, in order of how much it mattered:

1. **Loss masking is not optional.** Only 13% of the tokens in a Hermes session are the assistant's
   own turns; the rest is system prompt, tool schemas and tool output. v0 trained on all of it, spent
   87% of its gradient learning to predict `ls` output, and came out *worse* than the base model.
   v1 changed exactly one thing, masking the loss to assistant turns, and moved tool accuracy from
   36% to 53% while halving latency. That is why assistant-only loss is now the default.
2. **Thinking models need output budget in the eval.** With a 512-token cap the base model scored
   48% / 19% (action / tool); at 1k it scored 70% / 44%. The cap was cutting Qwen3's `<think>` block
   off before the tool call. v0 looked *better* than base under the small cap only because it had
   learned to think less. Default is now 1024.
3. **An 8 GB card needs three memory fixes** (see above): chunked cross-entropy, no all-ones attention
   mask, GPU-only placement. Windows does not OOM; it pages VRAM through system RAM and runs 10x slower.
4. **61 sessions is a smoke test, not a dataset.** The numbers above are for checking the pipeline
   works end to end. The interesting experiment starts at a few hundred sessions.

Reproduce: `scripts/results_table.py runs/eval-*.json` renders the table from the eval JSON files.

## Roadmap

- [x] Exporter grounded in the real `state.db` schema, with tests
- [x] Filters: repair, trim, truncate, dedupe, redact
- [x] QLoRA trainer (TRL + PEFT + bitsandbytes), dry-run mode
- [x] Merge, GGUF, Ollama, Hermes config
- [x] Next-action eval harness
- [x] Validate a full run on a consumer GPU and publish numbers
- [x] Assistant-only loss masking
- [ ] Train on a few hundred real sessions and report again
- [ ] Thinking on/off as an explicit, consistent choice between training data and eval
- [ ] Ship as a Hermes plugin (`hermes distill ...` via the `hermes_agent.plugins` entry point)
- [ ] Rejection-sampled round 2 using Hermes' batch runner

## How it relates to Hermes' own tooling

Hermes ships `batch_runner.py` (generate trajectories from a prompt file) and `trajectory_compressor.py`.
Those produce *new* data by running the agent. `hermes-distill` works on the data you already have, and
closes the loop by training, serving and evaluating.

## License

MIT
