# Generating sessions on purpose

`hermes-distill` works on whatever is in your Hermes history. If you want more data than
daily use produces, these scripts generate it the honest way: by actually running Hermes on
real tasks, so sessions land in `state.db` exactly like normal usage.

```bash
# 1. ~1,000 diverse task prompts grounded in the sandbox manifest (about $1 of Claude usage)
ANTHROPIC_API_KEY=sk-ant-... python scripts/datagen/make_prompts.py 1000 prompts_big.jsonl

# 2. Run them through Hermes one-shots, 6 in parallel, each in its own sandbox copy of this repo
#    plus synthetic CSV/JSON/log/config data and a small buggy package (resumable, retries on 429/529)
python scripts/datagen/run_sessions.py 6 prompts_big.jsonl results_big.jsonl

# 3. Export as usual; use --before/--after to split train and eval by time
hermes-distill export --out data --system-mode short --holdout 0.1
```

`sandbox_builder.py` is the single source of truth for what the agent sees. `MANIFEST` at the top is
also what the prompt generator reads, so adding a file there makes new task families possible.

Measured cost with Claude Sonnet 5 as the teacher: about $0.05 per session, 10 to 20 seconds each,
roughly 12 sessions per minute with 6 workers on a fresh API key.

Billing gotcha: Anthropic promotional credits did not cover sessions run through Hermes' native
Anthropic provider (they drained the purchased balance instead), but did cover the same model through
the OpenAI-compatible endpoint. If you are burning promo credit, point Hermes at it as a custom provider:

```bash
hermes config set model.provider custom
hermes config set model.base_url https://api.anthropic.com/v1
hermes config set model.default claude-sonnet-5
hermes config set model.api_mode chat_completions
# and CUSTOM_API_KEY=<your key> in the Hermes .env
```

Task families and their share of the generated prompts: data 18%, terminal 14%, write 14%, read 12%,
patch 12%, multi-step 12%, search 10%, no-tool chat 8%.
