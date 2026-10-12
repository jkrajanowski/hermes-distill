"""Merge the adapter, convert to GGUF, register with Ollama, point Hermes at it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def merge_adapter(adapter_dir: Path, base_model: str | None, out_dir: Path) -> Path:
    """Merge a LoRA adapter into its base model and save fp16 weights to ``out_dir``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if base_model is None:
        cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_model = cfg["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def _find_llama_cpp(explicit: Path | None) -> Path:
    candidates = [explicit, Path(os.environ.get("LLAMA_CPP_DIR", "")), Path.home() / "llama.cpp", Path("llama.cpp")]
    for c in candidates:
        if c and (c / "convert_hf_to_gguf.py").exists():
            return c
    raise SystemExit(
        "llama.cpp not found. Clone it (git clone https://github.com/ggml-org/llama.cpp) and pass --llama-cpp <dir> "
        "or set LLAMA_CPP_DIR."
    )


_CONVERTER_OUTTYPES = {"f32", "f16", "bf16", "q8_0"}


def to_gguf(merged_dir: Path, out_file: Path, llama_cpp: Path | None = None, quant: str = "q8_0") -> Path:
    """HF safetensors -> GGUF via llama.cpp's converter.

    ``f16``/``bf16``/``q8_0`` need only the Python converter (no C++ build). Other quants
    (``Q4_K_M`` etc.) go through an f16 GGUF and the ``llama-quantize`` binary.
    """
    repo = _find_llama_cpp(llama_cpp)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    converter = [sys.executable, str(repo / "convert_hf_to_gguf.py"), str(merged_dir)]
    if quant.lower() in _CONVERTER_OUTTYPES:
        subprocess.run(converter + ["--outfile", str(out_file), "--outtype", quant.lower()], check=True)
        return out_file
    f16 = out_file.with_suffix(".f16.gguf")
    subprocess.run(converter + ["--outfile", str(f16), "--outtype", "f16"], check=True)
    quantize = shutil.which("llama-quantize") or next(
        (str(p) for p in [repo / "build" / "bin" / "llama-quantize", repo / "build" / "bin" / "llama-quantize.exe",
                          repo / "llama-quantize", repo / "llama-quantize.exe"] if p.exists()),
        None,
    )
    if not quantize:
        raise SystemExit("llama-quantize binary not found; build llama.cpp or pass --quant f16.")
    subprocess.run([quantize, str(f16), str(out_file), quant], check=True)
    f16.unlink(missing_ok=True)
    return out_file


def find_ollama() -> str | None:
    """``ollama`` on PATH, else the default per-user install locations."""
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path("/usr/local/bin/ollama"),
        Path("/opt/homebrew/bin/ollama"),
        Path.home() / ".ollama" / "bin" / "ollama",
    ]
    return next((str(c) for c in candidates if c.exists()), None)


def ollama_create(name: str, gguf: Path, num_ctx: int = 32768, workdir: Path | None = None) -> Path:
    """Write a Modelfile next to the GGUF and run ``ollama create``."""
    workdir = workdir or gguf.parent
    modelfile = workdir / f"Modelfile.{name}"
    modelfile.write_text(
        f"FROM {gguf.resolve().as_posix()}\n"
        f"PARAMETER num_ctx {num_ctx}\n"
        "PARAMETER temperature 0.6\n"
        "PARAMETER top_p 0.95\n",
        encoding="utf-8",
    )
    ollama = find_ollama()
    if not ollama:
        raise SystemExit("ollama not found on PATH or in its default install location. Install it from https://ollama.com.")
    subprocess.run([ollama, "create", name, "-f", str(modelfile)], check=True)
    return modelfile


def hermes_config_snippet(model_name: str, base_url: str = "http://localhost:11434/v1") -> str:
    return (
        "# ~/.hermes/config.yaml\n"
        "model:\n"
        f'  default: "{model_name}"\n'
        '  provider: "custom"\n'
        f'  base_url: "{base_url}"\n'
        "\n"
        "# ~/.hermes/.env  (local models are slower than cloud APIs)\n"
        "HERMES_API_TIMEOUT=600\n"
    )


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
