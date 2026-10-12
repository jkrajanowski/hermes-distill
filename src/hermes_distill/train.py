"""QLoRA fine-tuning on exported conversations.

Heavy imports (torch, transformers, peft, trl) happen inside functions so the
CLI stays fast and usable on machines without a GPU stack.

Design notes
------------
* Conversations are rendered to text with the base model's own chat template
  (``apply_chat_template(messages, tools=...)``), so tool calls end up in the
  exact token format the model was pre-trained on (Qwen3, Llama 3.x, Hermes 4
  all ship tool-aware templates).
* Loss is computed over the whole rendered sequence in v0. Assistant-only
  masking needs ``{% generation %}`` markers in the template, which most
  templates lack; it is on the roadmap.
* Defaults target a single 8 GB consumer GPU: 4-bit base, LoRA r=16 on all
  linear layers, gradient checkpointing, batch 1 x grad-accum 8, 4k context.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    data: Path
    output: Path
    base_model: str = "Qwen/Qwen3-4B"
    epochs: float = 2.0
    learning_rate: float = 2e-4
    max_seq_len: int = 8192
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    batch_size: int = 1
    grad_accum: int = 8
    load_4bit: bool = True
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    logging_steps: int = 5
    save_steps: int = 200
    seed: int = 42
    max_samples: int | None = None
    drop_reasoning: bool = False


def load_records(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
                if max_samples and len(records) >= max_samples:
                    break
    return records


def _clean_for_template(messages: list[dict[str, Any]], drop_reasoning: bool) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        nm = {k: v for k, v in m.items() if k not in ("reasoning",)}
        if drop_reasoning:
            nm.pop("reasoning_content", None)
        if nm["role"] == "tool":
            nm = {"role": "tool", "content": nm.get("content") or "", "tool_call_id": nm.get("tool_call_id", ""),
                  "name": nm.get("name", "")}
        if nm["role"] == "assistant" and nm.get("tool_calls"):
            nm["tool_calls"] = [
                {"id": tc.get("id") or f"call_{i}", "type": "function",
                 "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                for i, tc in enumerate(nm["tool_calls"])
            ]
        out.append(nm)
    return out


def render_texts(records: list[dict[str, Any]], tokenizer: Any, drop_reasoning: bool = False) -> list[str]:
    texts = []
    for r in records:
        messages = _clean_for_template(r["messages"], drop_reasoning)
        tools = r.get("tools") or None
        try:
            text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
        except Exception:
            # Templates without tool support: fall back to no tools argument.
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)
    return texts


def token_stats(texts: list[str], tokenizer: Any, max_seq_len: int) -> dict[str, Any]:
    lengths = [len(tokenizer(t, add_special_tokens=False).input_ids) for t in texts]
    if not lengths:
        return {"n": 0}
    lengths_sorted = sorted(lengths)
    return {
        "n": len(lengths),
        "mean_tokens": round(sum(lengths) / len(lengths)),
        "p50_tokens": lengths_sorted[len(lengths) // 2],
        "p90_tokens": lengths_sorted[int(len(lengths) * 0.9)],
        "max_tokens": lengths_sorted[-1],
        "total_tokens": sum(lengths),
        "truncated": sum(1 for n in lengths if n > max_seq_len),
        "max_seq_len": max_seq_len,
    }


def dry_run(cfg: TrainConfig) -> dict[str, Any]:
    from transformers import AutoTokenizer

    records = load_records(cfg.data, cfg.max_samples)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    texts = render_texts(records, tokenizer, cfg.drop_reasoning)
    stats = token_stats(texts, tokenizer, cfg.max_seq_len)
    stats["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()}
    stats["example"] = texts[0][:1500] if texts else ""
    return stats


def train(cfg: TrainConfig) -> dict[str, Any]:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    records = load_records(cfg.data, cfg.max_samples)
    if not records:
        raise SystemExit(f"No records in {cfg.data}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    texts = render_texts(records, tokenizer, cfg.drop_reasoning)
    stats = token_stats(texts, tokenizer, cfg.max_seq_len)
    dataset = Dataset.from_dict({"text": texts})

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    quant = None
    if cfg.load_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    sft_kwargs: dict[str, Any] = dict(
        output_dir=str(cfg.output),
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        gradient_checkpointing=cfg.gradient_checkpointing,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        optim="paged_adamw_8bit" if cfg.load_4bit else "adamw_torch",
        seed=cfg.seed,
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )
    try:
        sft_config = SFTConfig(max_length=cfg.max_seq_len, **sft_kwargs)
    except TypeError:  # older TRL
        sft_config = SFTConfig(max_seq_length=cfg.max_seq_len, **sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(cfg.output))
    tokenizer.save_pretrained(str(cfg.output))
    summary = {
        "base_model": cfg.base_model,
        "adapter_dir": str(cfg.output),
        "train_loss": getattr(result, "training_loss", None),
        "steps": getattr(result, "global_step", None),
        "seconds": round(time.time() - t0),
        "data": stats,
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(cfg).items()},
    }
    (cfg.output / "hermes_distill_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
