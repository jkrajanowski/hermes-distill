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

Setup: training sessions generated by running Hermes (teacher: claude-sonnet-5) on a sandbox of
this repo plus synthetic CSVs, logs, configs and a small buggy package (`scripts/datagen/`), with
18 separate eval sessions generated beforehand (54 assistant turns). QLoRA on Qwen3-4B, r=16, 2 epochs,
4k context, short system prompt. v0 and v1 used 61 sessions (10 minutes each); v2 used 429 (74 minutes);
v3 used 952 (3 hours).
All local models are q8_0 GGUFs served by Ollama; eval budget 1024 output tokens per turn.

| model | action type | right tool | args exact | valid JSON | s/turn |
|---|---:|---:|---:|---:|---:|
| claude-sonnet-5 (teacher, replaying itself) | 96.3% | 80.6% | 19.4% | 100% | 1.9 |
| Qwen3-4B base | 70.4% | 44.4% | 8.3% | 55.6% | 15.1 |
| hermes-distill v0: full-sequence loss | 61.1% | 36.1% | 5.6% | 41.7% | 12.9 |
| hermes-distill v1: assistant-only loss, 61 sessions | 66.7% | 52.8% | 11.1% | 61.1% | 8.2 |
| hermes-distill v2: assistant-only loss, 429 sessions | 96.3% | 83.3% | 19.4% | 94.4% | 2.1 |
| hermes-distill v3: assistant-only loss, **952 sessions** | **98.1%** | **91.7%** | **25.0%** | **97.2%** | **2.4** |

v2 and v3 are the same recipe as v1 with 7x and 15x the data (74 minutes and 3 hours of training on
the laptop GPU). On this eval set v2 matches the teacher's own replay on every metric, and v3 goes past
it on tool choice (92% vs 81%: the replay is Sonnet re-deciding with temperature, the student has seen
the same task families 950 times). Both answer in about 2 seconds instead of 15, because they learned
to act without a long `<think>` preamble.

Read this with two caveats. With 54 scored turns the confidence interval on each percentage is roughly
±13 points. And the eval sessions come from the same sandbox and the same task families as the training
sessions (different prompts, same files and tools), so this measures in-distribution behaviour: "a 4B model
can learn how *this* agent works on *this* kind of work", not "it generalizes to any codebase". The
per-tool table makes the point: v2 picks `search_files` on 8 of 8 turns where the teacher did; the base
model never picked it once.

A second, larger holdout (30 sessions and 118 turns, split by session id from the same generated
pool, never trained on by any version) shows the same ordering with tighter error bars (about ±9 points):

| model | action type | right tool | args exact | valid JSON | s/turn |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B base | 58.5% | 29.7% | 2.2% | 46.2% | 15.1 |
| v1 (61 sessions) | 58.5% | 41.6% | 9.0% | 64.0% | 5.3 |
| v2 (429 sessions) | 88.1% | 71.4% | 19.8% | 86.8% | 1.7 |
| v3 (952 sessions) | **92.4%** | **85.7%** | **20.9%** | **94.5%** | **1.8** |

Per tool, v3's gain over v2 is almost all `terminal` (30/35 vs 23/35) and `write_file` (5/5 vs 2/5);
`patch` stays at 0/2 for every student, which is the one tool the data has too few examples of (109
calls in 952 sessions).

### Is it just overfitting?

Two more checks, both on turns scored the same way:

- **Training-set fit**: 30 of the 65 first-day sessions, which both v2 and v3 trained on.
- **Out of distribution**: 23 new sessions on a different project. That project is a ComfyUI image
  pipeline with its own scripts, JSON manifests, workflows and logs. The student never saw it. The
  prompts were written by hand and the teacher generated the sessions the same day
  (`SANDBOX_SRC=... scripts/datagen/run_sessions.py`).

| model | training-set fit (81 turns) | 118-turn holdout | new project (85 turns) |
|---|---:|---:|---:|
| Qwen3-4B base | 50.0% | 29.7% | 22.2% |
| v1 (61 sessions) | n/a | 41.6% | 44.4% |
| v2 (429 sessions) | 84.6% | 71.4% | 75.0% |
| v3 (952 sessions) | 84.6% | 85.7% | 82.8% |

*Right-tool accuracy.*

On the new project, v3 scored 92.9% on action type, 21.9% on exact arguments and 96.9% on valid JSON, at 2.1 s per turn.

v3 scores the same on sessions it trained on, sessions it did not, and a codebase it never saw.
A model that memorized its data would show a large train-to-test gap; this one shows none. What it
learned is the agent's *policy*: which tool to reach for given the request and what is on screen.
Before acting, it searches files to orient itself, reads a file and then runs code over it, and it
answers without tools when no tool is needed. The base model gets worse as the tasks move away from
the sandbox it was asked about. The students do not.

Why the student agrees with the teacher more than the teacher agrees with itself:

- The original sessions were *sampled*, and every model in the eval is decoded greedily at temperature 0.
- Training on about a thousand sampled trajectories and then decoding greedily gives the teacher's
  most common choice in each state. On ambiguous turns that is also the choice the held-out
  trajectory most likely contains.
- So "beats the teacher" means "more consistent than the teacher", nothing more. The same effect caps
  "args exact" near 20 to 25% for every model, the teacher included.

What this does **not** show:

- These are still one-shot Hermes tasks of the same *kinds*: read, search, data, write, patch, a few
  multi-step ones. They used the same tools and the same sandbox rules. A new codebase is not a new job.
- The out-of-distribution set is small: about ±10 points at 85 turns.
- Every scored turn is conditioned on the teacher's own correct history. Nothing here tests whether
  the student recovers after its *own* wrong move, which is where a collapsed, low-variance policy
  should be weakest.

What the first runs taught us, in order of how much it mattered:

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
4. **Data is the lever once the loss is right.** 61 sessions gave a modest gain; 429 sessions closed
   the gap to the teacher on in-distribution tasks; 952 sessions added another 8 points of tool accuracy
   on the 18-session set and 14 on the larger holdout, so the curve had not flattened yet. Generating sessions costs about $0.06 each with
   prompt caching (see `scripts/datagen/README.md` for the billing and toolset gotchas that make it
   7x more without).
5. **Agent sessions need containment.** Two generated sessions escaped their sandbox: one searched the
   whole machine and patched the real repo, one ran `pip install -e .` and re-pointed a global package.
   The sandbox now carries an `AGENTS.md` with rules, and the exporter drops any session whose tool
   calls touch paths outside its working directory.

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
