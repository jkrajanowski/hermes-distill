# Train -> merge/GGUF/Ollama -> evals, detached-friendly. Run from the repo root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_chain.ps1 -Version v3 -Data data\v3
# Training deps live in the CUDA-enabled Python 3.10 (`py -3.10`), the eval CLI in .venv.
param(
    [Parameter(Mandatory = $true)][string]$Version,
    [Parameter(Mandatory = $true)][string]$Data,
    [string]$BaseModel = "Qwen/Qwen3-4B",
    [double]$Epochs = 2,
    [int]$MaxSeqLen = 4096,
    [int]$HoldoutSamples = 30
)
$env:PYTHONIOENCODING = "utf-8"
$status = "runs\chain-$Version.status"
"TRAINING" | Out-File $status
py -3.10 -u -m hermes_distill.cli train --data "$Data\train.messages.jsonl" --base-model $BaseModel --epochs $Epochs --max-seq-len $MaxSeqLen --output "runs\qwen3-4b-$Version" *> "runs\train-$Version.log"
if (-not (Test-Path "runs\qwen3-4b-$Version\adapter_model.safetensors")) { "TRAIN FAILED" | Out-File $status; exit 1 }
"SERVING" | Out-File $status
py -3.10 -m hermes_distill.cli serve --adapter "runs\qwen3-4b-$Version" --name "hermes-distill-$Version" --merged-dir "models\merged-$Version" --gguf "models\hermes-distill-$Version.q8_0.gguf" --quant q8_0 --llama-cpp "$env:USERPROFILE\llama.cpp" --num-ctx 8192 *> "runs\serve-$Version.log"
"EVAL" | Out-File $status
.\.venv\Scripts\hermes-distill.exe eval --data data\eval\train.messages.jsonl --model "hermes-distill-$Version" --base-url http://localhost:11434/v1 --out "runs\eval-distilled-$Version.json" *> "runs\eval-$Version-a.log"
.\.venv\Scripts\hermes-distill.exe eval --data "$Data\holdout.messages.jsonl" --model "hermes-distill-$Version" --base-url http://localhost:11434/v1 --max-samples $HoldoutSamples --out "runs\eval-hold-$Version.json" *> "runs\eval-$Version-b.log"
"DONE" | Out-File $status
