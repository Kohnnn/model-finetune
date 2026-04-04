# VietCap Qwen3.5-4B Fine-tuning Development Journal

**Project**: Private AI Analyst Stack (OCR -> Fine-tune -> RAG App -> OCI Deployment)
**Date**: 2026-03-31 | **Status**: In Progress

## Reference Pipeline (Jackrong Opus Distillation)
Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF:
- Base: Qwen/Qwen3.5-27B, Framework: Unsloth LoRA SFT
- Training format: <thinking> {reasoning} 
 {answer}, <|im_start|> chat template
- Starting loss: 0.74356 -> Final: 0.23984, Context: 16,384 tokens

Datasets used: nohurry/Opus-4.6-Reasoning-3000x-filtered (2330 rows),
Roman1111111/claude-opus-4.6-10000x (9633 rows),
TeichAI/claude-4.5-opus-high-reasoning-250x (250 rows),
Jackrong/Qwen3.5-reasoning-700x (633 rows)

## Our Simplified Pipeline (Two-Phase)
Base: Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-GGUF (already distilled 4B)

Phase 1: Fine-tune on Opus reasoning datasets -> sharpen analytical thinking
  Dataset: finetune/outputs/datasets/opus_reasoning.jsonl (~5000-6000 rows)
  Command: python finetune/train.py --model-name Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled
    --dataset-path finetune/outputs/datasets/opus_reasoning.jsonl
    --learning-rate 5e-5 --weight-decay 0.05 --num-epochs 1

Phase 2: Fine-tune on VietCap private data -> domain adaptation
  Step 2a: python finetune/generate_sft_dataset.py (Ollama generation)
  Step 2b: python finetune/train.py --dataset-path finetune/outputs/datasets/vietcap_sft_generated.jsonl
    --learning-rate 2e-5 --weight-decay 0.05 --num-epochs 1

## Created: download_opus_datasets.py
- Downloads all 4 Opus datasets via `datasets` library
- Converts flat/conversation/message formats -> unified {"messages": [...]} JSONL
- Roman dataset filtered by metadata.category in ("simple logic and math", "math")
- System prompt: "You are a helpful assistant. Please reason step by step before answering."

## process_pdfs.py Syntax Fixes
- Fixed literal newline characters injected by heredoc edits in string literals
  (e.g., `text.split('` + newline + `')` -> `text.split('\n')`)
- All files verified: `python -m py_compile` passes, `ruff check` passes (0 errors)

## Noise Filters Added to process_pdfs.py
- min_chunk_words: 50 -> 200
- HEAD_SECTION_MARKERS: strip leading boilerplate (mirrors existing TAIL_SECTION_MARKERS)
- NOISE_PATTERNS regex: Figure captions, Page numbers, ToC, Termination notices,
  Source citations, Rating legends, Financial report fragments, Rating system boilerplate
- Require >= 2 ANALYTICAL_MARKERS per chunk (target price, valuation, earnings, etc.)
- Excessive number detection: reject if >30% of characters are digits (table/chart noise)

## generate_sft_dataset.py Fixes
- CRITICAL: Added out_fh.write() - file was opened at line 233 but never written (BUG)
- Added try/except around json.loads() at line 163 - was crashing on malformed JSON lines
- Added Ollama retry with exponential backoff (2^attempt seconds)

## train.py Updated Defaults
- --learning-rate: 2e-4 -> 5e-5
- --weight-decay: 0.01 -> 0.05
- --num-epochs: 3.0 -> 1.0
- --model-name: unsloth/Qwen3.5-4B -> Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled

## System Prompt (VietCap Analyst)
You are a senior equity research analyst at VietCap Securities.
Based ONLY on the provided context, deliver concise, evidence-based analytical commentary.
Cite specific data points inline as [S1], [S2] when using evidence.
Do NOT copy full sentences. Paraphrase and synthesize.
Structure: 1) Key thesis, 2) Supporting evidence, 3) Risks and caveats

## Output Paths
finetune/outputs/datasets/opus_reasoning.jsonl       - Opus reasoning data
finetune/outputs/datasets/vietcap_sft_generated.jsonl - VietCap with generated responses
finetune/outputs/qwen35_4b_opus_reasoning/merged_model/  - Phase 1 model
finetune/outputs/qwen35_4b_vietcap_final/merged_model/   - Phase 2 model
finetune/outputs/qwen35_4b_vietcap_final/gguf/            - GGUF export

Last updated: 2026-03-31