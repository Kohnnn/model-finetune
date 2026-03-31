# Development Journal

This journal records the main engineering decisions, failures, fixes, and milestones that turned the folder from a scaffold into a working private analyst stack with a trained Qwen 3.5 model.

## 2026-03-28 to 2026-03-29: Repository Read and RAG-First MVP

- Mapped the repository and confirmed the OCR pipeline was the most complete subsystem.
- Chose a RAG-first path because the corpus was already large while the fine-tuning dataset still had empty assistant completions.
- Implemented a FastAPI retrieval app, Chroma ingestion flow, Docker packaging, bootstrap script, and deployment docs.
- Brought up the local stack with Chroma, llama.cpp, and the app service.
- Added source-grounded fallbacks so the app returned extractive evidence when the local smoke-test GGUF was too weak.

## 2026-03-29: Parser Cleanup and Training Preparation

- Expanded `ocr_pipeline/process_pdfs.py` to strip disclaimer and contact sections in both English and Vietnamese.
- Added metadata such as `doc_id`, `title`, `year`, `language`, `chunk_index`, and `chunk_word_count` to retrieval and SFT outputs.
- Regenerated the cleaned OCR outputs from `raw_dataset/`.
- Final cleaned parse result:
  - supported files discovered: `8180`
  - successfully processed: `8179`
  - cleaned chunks: `23978`
  - remaining failure: `ocr_pipeline/parse_failures.log` -> `SIP-20231101-KQKD.docx`

## 2026-03-29: GPU Environment Repair for Qwen 3.5

- Verified the machine had an RTX 4060 Ti 16GB but the Python environment was still CPU-only.
- Upgraded the local `.venv` to CUDA PyTorch `2.11.0+cu128`.
- Upgraded `transformers` and refreshed `unsloth` plus `unsloth-zoo` from source so `qwen3_5` would load correctly.
- Added `finetune/setup_gpu_env.ps1` so the GPU environment is reproducible.
- Stored `HF_TOKEN` locally as a user-scoped environment variable.

## 2026-03-29 to 2026-03-30: Training Script Stabilization

- Hardened `finetune/train.py` for `unsloth/Qwen3.5-4B`.
- Added CUDA and model-config preflight checks.
- Added compatibility handling for:
  - `eval_strategy` versus `evaluation_strategy`
  - `processing_class` versus tokenizer arguments in `Trainer`
  - processor-backed tokenization for Qwen 3.5
- Added merged-model saving and Hugging Face push options.
- Added `training_summary.json` generation for completed runs.
- Added `finetune/export_gguf.py` so a completed adapter can be exported without retraining.

## 2026-03-30: Full-Corpus Draft Dataset

- Generated a full-corpus draft SFT file at `finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl`.
- This dataset uses cleaned OCR context with heuristic draft completions.
- Final draft row count: `23974`
- Average truncated context length: about `409` words.

## 2026-03-30: Full Qwen 3.5 Fine-Tune

- Ran the full local LoRA training job against the full-corpus draft set.
- Final training command used:

```powershell
.venv\Scripts\python.exe finetune/train.py \
  --dataset-path finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl \
  --output-dir finetune/outputs/qwen35_4b_full_corpus_draft23974 \
  --max-seq-length 1024 \
  --batch-size 1 \
  --gradient-accumulation 4 \
  --num-epochs 1 \
  --eval-split 0 \
  --log-steps 100 \
  --save-steps 500 \
  --warmup-steps 100 \
  --save-merged-model \
  --skip-gguf-export \
  --disable-response-only-masking
```

- Final result:
  - output dir: `finetune/outputs/qwen35_4b_full_corpus_draft23974`
  - train rows: `23974`
  - epochs: `1.0`
  - runtime: `68249.53s` (~`18.96h`)
  - train loss: `1.0765`
  - adapter size: about `0.10 GB`
  - merged model size: about `8.70 GB`

## 2026-03-30: Private Hugging Face Publication

- Created and uploaded the merged model to a private Hugging Face repository.
- Final repo: `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`
- Final repo URL: `https://huggingface.co/Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`
- Included the merged model, tokenizer/processor files, README, and training summary.

## 2026-03-31: GGUF Export

- Initial GGUF export failed because Unsloth could not auto-install or detect the local `llama.cpp` toolchain on Windows.
- Installed Visual Studio Build Tools and used the existing local CMake installation.
- Manually cloned and built `llama.cpp` under `C:\Users\Admin\.unsloth\llama.cpp`.
- Added Windows path and OpenSSL detection fixes in `finetune/export_gguf.py`.
- Final GGUF export succeeded.

Generated GGUF artifacts:

- `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.Q4_K_M.gguf`
- `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf/Qwen3.5-4B.BF16-mmproj.gguf`

Approximate GGUF directory size: `3.15 GB`

## 2026-03-31: Commit-Ready Cleanup

- Removed temporary smoke-training folders.
- Removed intermediate pilot parse outputs.
- Removed the old draft-only and CPU-seed outputs from `finetune/outputs/`.
- Initialized a local git repository with `git init`.
- Updated `.gitignore` to exclude `.agent/` and `unsloth_compiled_cache/` in addition to existing large/private assets.

## Final Artifact Ledger

- Full dataset draft set: `finetune/outputs/datasets/qwen35_full_corpus_draft.jsonl`
- Full adapter: `finetune/outputs/qwen35_4b_full_corpus_draft23974/adapter`
- Full merged model: `finetune/outputs/qwen35_4b_full_corpus_draft23974/merged_model`
- Full training summary: `finetune/outputs/qwen35_4b_full_corpus_draft23974/training_summary.json`
- GGUF export: `finetune/outputs/qwen35_4b_full_corpus_draft23974/gguf/qwen3_5_4b_private_analyst_full_corpus_q4_k_m_gguf`
- Private Hugging Face repo: `Mikkkkoooo/qwen35-4b-private-analyst-full-corpus`

## Current Recommendation

- Use the private Hugging Face model for experimentation and evaluation now.
- Use the GGUF artifact for deployment testing.
- For a stronger final house-style model, review the highest-value examples in the full-corpus draft set and rerun the same training pipeline on that curated dataset.
