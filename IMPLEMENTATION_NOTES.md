# Parsing + Fine-Tuning Implementation Notes

This document captures what was implemented, executed, and validated for:

1. Dataset parsing (pilot run)
2. Fine-tuning pipeline setup
3. Guide benchmark summary

Date: 2026-03-25

## Files Updated

- `ocr_pipeline/process_pdfs.py`
- `finetune/train.py`
- `finetune/requirements.txt`

## 1) OCR/Data Parsing: What Was Implemented

`ocr_pipeline/process_pdfs.py` was upgraded from a static script into a production-style CLI pipeline.

### Key capabilities

- Recursive file discovery from input root.
- Extension filter (`--extensions`) with default `.pdf`.
- Pilot support (`--limit`) with deterministic sampling (`--sample-mode random --seed ...`).
- Configurable chunking:
  - `--chunk-words` (default 800)
  - `--overlap-words` (default 100)
  - `--min-chunk-words` (default 50)
- Tail trimming (`--trim-tail-pages`) to drop likely disclaimer pages.
- Normalized text cleanup.
- Structured JSONL outputs:
  - `chroma_chunks.jsonl` for RAG ingestion
  - `finetune_template.jsonl` for SFT labeling
- Per-file logging, runtime summary, and failure log generation.

### Output schema

`chroma_chunks.jsonl` rows contain:

- `id`
- `text`
- `metadata.source`
- `metadata.relative_source`
- `metadata.chunk_index`
- `metadata.file_extension`

`finetune_template.jsonl` rows contain chat-style `messages`:

- `system` prompt
- `user` context + task prompt
- `assistant` placeholder (empty string for manual completion)

## 2) Pilot Parsing Run: Executed Benchmark

### Command used

```bash
python ocr_pipeline/process_pdfs.py \
  --input-dir D:/finetune/raw_dataset \
  --output-dir D:/finetune/ocr_pipeline \
  --extensions .pdf \
  --limit 14 \
  --sample-mode random \
  --seed 3407 \
  --chunk-words 800 \
  --overlap-words 100 \
  --min-chunk-words 50 \
  --trim-tail-pages 3
```

### Pilot results

- Files discovered (`.pdf`): 116
- Files selected: 14
- Files processed: 14
- Failed files: 0
- Total chunks: 863
- Average chunks per file: 61.64
- Runtime: 8.65 seconds
- Approx input size of selected files: 52.05 MB

### Generated files

- `ocr_pipeline/chroma_chunks.jsonl`
- `ocr_pipeline/finetune_template.jsonl`

### Generated data quality stats

- `chroma_chunks.jsonl` lines: 863
- `finetune_template.jsonl` lines: 863
- Chunk word count: min 67, median 800, p95 800, max 800
- Output sizes:
  - `chroma_chunks.jsonl`: 4.17 MB
  - `finetune_template.jsonl`: 4.33 MB

## 3) Fine-Tune Phase: What Was Implemented

`finetune/train.py` was rewritten into a configurable CLI training pipeline.

### Key capabilities

- CLI arguments for model, dataset path, output, LoRA, optimizer, eval split, and export behavior.
- Runtime checks:
  - Python version guardrails
  - warning for Python 3.14 compatibility risk
  - CUDA required for actual training
- Dataset validation:
  - checks `messages` structure
  - optional filtering of empty assistant outputs
  - clear failure message when all assistant outputs are empty
- Optional dry run mode (`--dry-run`) for data verification without loading model.
- Unsloth + LoRA setup:
  - default model: `Qwen/Qwen2.5-3B-Instruct`
  - default LoRA: `r=16`, `lora_alpha=16`
- TRL SFT training setup with configurable hyperparameters.
- Optional eval split (auto disabled for very small datasets).
- Adapter save to `outputs/adapter`.
- GGUF export (`q4_k_m`) to `outputs/gguf/<name>`.

## 4) Requirements Setup

`finetune/requirements.txt` now includes practical dependency bounds for reproducibility:

- torch
- unsloth
- transformers==4.57.3
- trl==0.22.2
- datasets
- accelerate
- peft
- bitsandbytes (non-Windows marker)
- safetensors
- sentencepiece
- huggingface_hub[hf_transfer]
- tensorboard

Notes:

- Recommended runtime is Python 3.11 with CUDA-capable Linux/WSL2.
- Windows + Python 3.14 can work for validation paths, but full Unsloth stack is less predictable.

## 5) Validation Commands Executed

### Syntax validation

```bash
python -m compileall ocr_pipeline/process_pdfs.py finetune/train.py
```

### Fine-tune CLI inspection

```bash
python finetune/train.py --help
```

### Dry-run dataset validation (success)

```bash
python finetune/train.py --dry-run --allow-empty-assistant --max-samples 10
```

### Empty-assistant guardrail test (expected failure)

```bash
python finetune/train.py --dry-run --max-samples 50
```

Expected behavior was confirmed: training is blocked when assistant outputs are empty unless explicitly overridden.

## 6) Benchmark: `FINE_TUNING_GUIDE.md`

### Score

- Overall benchmark: **7.0 / 10**

### Strong areas

- Clear end-to-end architecture (OCR -> SFT -> RAG -> deployment).
- Practical baseline hyperparameters for 16 GB VRAM.
- Deployment topology is conceptually sound for OCI CPU inference.

### Gaps found

- No pinned dependency matrix in the guide.
- Version drift in model naming across files.
- Limited operational details for reproducible execution.
- No strict train/eval contract for monitoring model quality.

### What this implementation already addressed

- Added reproducible requirements in `finetune/requirements.txt`.
- Added structured training CLI with eval split support in `finetune/train.py`.
- Added deterministic parser and pilot benchmark workflow in `ocr_pipeline/process_pdfs.py`.

## 7) Recommended Next Steps

1. Fill assistant completions in `ocr_pipeline/finetune_template.jsonl` for at least 300 high-quality rows.
2. Create Python 3.11 environment and install `finetune/requirements.txt`.
3. Run a short smoke training job:

```bash
python finetune/train.py \
  --dataset-path D:/finetune/ocr_pipeline/finetune_template.jsonl \
  --max-samples 128 \
  --num-epochs 0.2 \
  --skip-gguf-export
```

4. Run full training once dataset completions are ready, then export GGUF.
5. Copy GGUF to deployment model path and start compose stack.
